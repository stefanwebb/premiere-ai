"""Compute the sync offset (and clock-drift estimate) between a camera
recording and a high-quality mic recording of the same speech, by
combining word-level ASR timestamps with local cross-correlation.
"""

import argparse
import difflib
import json
import os
import statistics
import sys
import tempfile

import numpy as np

from premiere_ai.transcribe import VIDEO_EXTENSIONS, _extract_audio, transcribe_file

_STRIP_CHARS = ".,!?;:\"'"


def _normalize_token(text: str) -> str:
    return text.lower().strip(_STRIP_CHARS)


def _align_words(camera_words: list, mic_words: list) -> list:
    """Fuzzy-match two word-timestamp lists via difflib.SequenceMatcher.

    Returns (camera_word, mic_word) pairs, in order, for every word pair
    difflib considers an "equal" match between the two normalized token
    sequences. Mismatched/inserted/deleted words on either side are
    skipped — they never appear in the result.
    """
    camera_tokens = [_normalize_token(w["text"]) for w in camera_words]
    mic_tokens = [_normalize_token(w["text"]) for w in mic_words]

    matcher = difflib.SequenceMatcher(a=camera_tokens, b=mic_tokens, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            continue
        for offset in range(i2 - i1):
            pairs.append((camera_words[i1 + offset], mic_words[j1 + offset]))
    return pairs


def _pick_anchors(matched_pairs: list, anchor_spacing_seconds: float) -> list:
    """Pick one matched pair per anchor_spacing_seconds-sized step across
    the camera timeline (nearest match to each step's target time).

    Bounds the anchor count to roughly duration / anchor_spacing_seconds
    regardless of speech density, keeping cross-correlation cost
    predictable for long recordings.
    """
    if not matched_pairs:
        return []

    duration_seconds = matched_pairs[-1][0]["start"]
    anchors = []
    used_indices = set()
    target = 0.0
    while target <= duration_seconds:
        best_idx = min(
            range(len(matched_pairs)),
            key=lambda i: abs(matched_pairs[i][0]["start"] - target),
        )
        if best_idx not in used_indices:
            used_indices.add(best_idx)
            anchors.append(matched_pairs[best_idx])
        target += anchor_spacing_seconds
    return anchors


def _compute_anchor_offset(
    camera_audio,
    mic_audio,
    sample_rate,
    camera_time_seconds,
    mic_time_approx_seconds,
    window_seconds,
    search_range_seconds,
):
    """Refine an approximate (ASR-derived) offset via local normalized
    cross-correlation.

    Fixes a window_seconds-long slice of camera_audio centered on
    camera_time_seconds, then slides an equal-length window through
    mic_audio across +/-search_range_seconds around
    mic_time_approx_seconds, returning the (offset_seconds, confidence)
    of the best-correlating lag. Returns None if the required window/
    search range would run off the edge of either recording, or either
    window is silent (zero standard deviation, so correlation is
    meaningless).
    """
    window_samples = int(round(window_seconds * sample_rate))
    search_samples = int(round(search_range_seconds * sample_rate))

    cam_center = int(round(camera_time_seconds * sample_rate))
    cam_start = cam_center - window_samples // 2
    cam_end = cam_start + window_samples
    if cam_start < 0 or cam_end > len(camera_audio):
        return None
    camera_window = camera_audio[cam_start:cam_end]

    camera_std = camera_window.std()
    if camera_std == 0:
        return None
    camera_norm = (camera_window - camera_window.mean()) / camera_std

    mic_center = int(round(mic_time_approx_seconds * sample_rate))
    search_start = mic_center - window_samples // 2 - search_samples
    search_end = mic_center + window_samples // 2 + search_samples
    if search_start < 0 or search_end > len(mic_audio):
        return None
    mic_search = mic_audio[search_start:search_end]

    best_lag = None
    best_corr = -2.0
    num_lags = 2 * search_samples + 1
    for lag in range(num_lags):
        mic_window = mic_search[lag:lag + window_samples]
        mic_std = mic_window.std()
        if mic_std == 0:
            continue
        mic_norm = (mic_window - mic_window.mean()) / mic_std
        corr = float(np.dot(camera_norm, mic_norm) / window_samples)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_lag is None:
        return None

    lag_seconds = (best_lag - search_samples) / sample_rate
    offset_seconds = (mic_time_approx_seconds + lag_seconds) - camera_time_seconds
    return offset_seconds, best_corr


def _fit_drift_line(confident_anchors: list) -> "dict | None":
    """Fit offsetSeconds vs. cameraSeconds across confident anchors.

    The slope is the drift rate (catches clock drift between two
    independently-clocked recording devices over a long take); the
    intercept is the recommended constant offset. Returns None if fewer
    than 3 anchors are given — not enough points for a meaningful fit.
    """
    if len(confident_anchors) < 3:
        return None

    camera_times = np.array([a["cameraSeconds"] for a in confident_anchors], dtype=float)
    offsets = np.array([a["offsetSeconds"] for a in confident_anchors], dtype=float)
    slope, intercept = np.polyfit(camera_times, offsets, 1)

    return {
        "slopeSecondsPerSecond": float(slope),
        "interceptSeconds": float(intercept),
        "driftMsPerMinute": float(slope * 60 * 1000),
    }


def _load_audio_for_correlation(media_path: str, high_fidelity: bool):
    """Decode media_path to mono audio samples for cross-correlation.

    Video files are extracted to a temporary WAV first (matching
    transcribe.py's own approach). high_fidelity=False (default) decodes
    at 16kHz mono via the same mlx_audio path transcribe.py already uses
    for ASR — no re-extraction needed downstream. high_fidelity=True
    decodes at the file's native sample rate via torchaudio instead.
    """
    ext = os.path.splitext(media_path)[1].lower()
    tmp_wav_path = None
    try:
        if ext in VIDEO_EXTENSIONS:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            _extract_audio(media_path, tmp.name)
            tmp_wav_path = tmp.name
            audio_path = tmp_wav_path
        else:
            audio_path = media_path

        if high_fidelity:
            import torchaudio

            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze(0).numpy(), sample_rate
        else:
            from mlx_audio.stt.utils import load_audio

            audio_mx = load_audio(audio_path)
            return np.array(audio_mx), 16000
    finally:
        if tmp_wav_path is not None and os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)


def compute_sync_offsets(
    camera_file: str,
    mic_file: str,
    camera_words: "str | None" = None,
    mic_words: "str | None" = None,
    anchor_spacing_seconds: float = 30.0,
    correlation_window_seconds: float = 1.0,
    correlation_search_range_seconds: float = 0.5,
    min_confidence: float = 0.5,
    high_fidelity: bool = False,
    language: str = "English",
) -> dict:
    """Compute the sync offset (and drift estimate) between camera_file
    and mic_file, two recordings of the same speech.

    The offsetSeconds (and recommendedOffsetSeconds) is mic_timestamp −
    camera_timestamp — i.e. the amount you'd shift the mic clip later
    (if positive) to align it with the camera recording.

    Raises FileNotFoundError if either input file is missing. Raises
    RuntimeError if zero words match between the two transcripts, or if
    zero anchors survive confidence filtering — in both cases nothing
    useful can be computed.
    """
    if not os.path.isfile(camera_file):
        raise FileNotFoundError(f"file not found: {camera_file}")
    if not os.path.isfile(mic_file):
        raise FileNotFoundError(f"file not found: {mic_file}")

    camera_words_path = camera_words or transcribe_file(camera_file, language=language)[1]
    mic_words_path = mic_words or transcribe_file(mic_file, language=language)[1]

    with open(camera_words_path, encoding="utf-8") as fh:
        camera_word_list = json.load(fh)
    with open(mic_words_path, encoding="utf-8") as fh:
        mic_word_list = json.load(fh)

    matched_pairs = _align_words(camera_word_list, mic_word_list)
    if not matched_pairs:
        raise RuntimeError(
            "no words matched between the two transcripts — check the language, "
            "or that both recordings capture the same speech"
        )

    anchor_pairs = _pick_anchors(matched_pairs, anchor_spacing_seconds)

    camera_audio, camera_sr = _load_audio_for_correlation(camera_file, high_fidelity)
    mic_audio, mic_sr = _load_audio_for_correlation(mic_file, high_fidelity)

    if mic_sr != camera_sr:
        import torch
        import torchaudio

        mic_audio = torchaudio.functional.resample(
            torch.from_numpy(mic_audio), mic_sr, camera_sr
        ).numpy()

    sample_rate = camera_sr

    anchors = []
    skipped_edge_count = 0
    for camera_word, mic_word in anchor_pairs:
        result = _compute_anchor_offset(
            camera_audio,
            mic_audio,
            sample_rate,
            camera_word["start"],
            mic_word["start"],
            correlation_window_seconds,
            correlation_search_range_seconds,
        )
        if result is None:
            skipped_edge_count += 1
            continue
        offset_seconds, confidence = result
        # micSeconds is the correlation-corrected mic timestamp (camera time + refined offset), not raw ASR mic time
        anchors.append({
            "cameraSeconds": camera_word["start"],
            "micSeconds": camera_word["start"] + offset_seconds,
            "offsetSeconds": offset_seconds,
            "confidence": confidence,
        })

    confident_anchors = [a for a in anchors if a["confidence"] >= min_confidence]
    dropped_low_confidence_count = len(anchors) - len(confident_anchors)

    warnings = []
    if skipped_edge_count > 0:
        warnings.append(
            f"{skipped_edge_count} anchor(s) skipped: too close to a recording's "
            "start/end for the requested window/search range"
        )

    if not confident_anchors:
        raise RuntimeError(
            f"no anchors met the confidence threshold (min_confidence={min_confidence}); "
            "recordings may not contain the same speech, or try --high-fidelity / "
            "a larger --correlation-window-seconds"
        )

    drift_fit = _fit_drift_line(confident_anchors)
    if drift_fit is not None:
        recommended_offset_seconds = drift_fit["interceptSeconds"]
    else:
        recommended_offset_seconds = statistics.median(
            a["offsetSeconds"] for a in confident_anchors
        )
        warnings.append(
            f"only {len(confident_anchors)} confident anchor(s) (fewer than 3) — "
            "using median offset, no drift fit"
        )

    return {
        "cameraFile": camera_file,
        "micFile": mic_file,
        "anchors": confident_anchors,
        "droppedLowConfidenceCount": dropped_low_confidence_count,
        "driftFit": drift_fit,
        "recommendedOffsetSeconds": recommended_offset_seconds,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute the audio-sync offset (and clock-drift estimate) between a "
                    "camera recording and a high-quality mic recording of the same speech."
    )
    parser.add_argument("camera_file", help="Path to the camera recording (video or audio).")
    parser.add_argument("mic_file", help="Path to the external mic recording (audio).")
    parser.add_argument(
        "--camera-words", default=None,
        help="Existing .words.json for camera_file (computed via transcribe if omitted).",
    )
    parser.add_argument(
        "--mic-words", default=None,
        help="Existing .words.json for mic_file (computed via transcribe if omitted).",
    )
    parser.add_argument("--anchor-spacing-seconds", type=float, default=30.0)
    parser.add_argument("--correlation-window-seconds", type=float, default=1.0)
    parser.add_argument("--correlation-search-range-seconds", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--high-fidelity", action="store_true",
        help="Decode at native sample rate instead of reusing 16kHz ASR audio.",
    )
    parser.add_argument("--language", "-l", default="English")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to write the JSON report (default: print to stdout).",
    )
    args = parser.parse_args()

    try:
        report = compute_sync_offsets(
            args.camera_file,
            args.mic_file,
            camera_words=args.camera_words,
            mic_words=args.mic_words,
            anchor_spacing_seconds=args.anchor_spacing_seconds,
            correlation_window_seconds=args.correlation_window_seconds,
            correlation_search_range_seconds=args.correlation_search_range_seconds,
            min_confidence=args.min_confidence,
            high_fidelity=args.high_fidelity,
            language=args.language,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output_json = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output_json)
        print(f"Report written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
