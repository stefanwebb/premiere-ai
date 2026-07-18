"""Detect pauses to cut from an audio/video file: transcribe it, ask Claude
which word-gaps are phrase/sentence boundaries, confirm silence at those
boundaries with Silero VAD, and emit MM:SS:FF cut ranges.
"""

import json
import os
import tempfile

from premiere_ai.pause_cuts import (
    build_boundary_prompt,
    format_duration,
    frames_to_timecode,
    invert_speech_to_silence,
    parse_boundary_response,
    select_cut_candidates,
    total_pause_seconds,
)
from premiere_ai.transcribe import (
    VIDEO_EXTENSIONS,
    _extract_audio,
    transcribe_file,
)

ANTHROPIC_MODEL = "claude-opus-4-8"


def _ensure_words_json(input_file: str, language: str, verbose: bool) -> str:
    """Return the path to <stem>.words.json, producing it via transcribe_file
    if it doesn't already exist."""
    stem = os.path.splitext(input_file)[0]
    words_path = f"{stem}.words.json"
    if os.path.isfile(words_path):
        return words_path
    _, words_path = transcribe_file(input_file, language=language, verbose=verbose)
    return words_path


def _load_words(words_path: str) -> "list[dict]":
    with open(words_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _get_wav_path(input_file: str) -> "tuple[str, str | None]":
    """Return (wav_path, tmp_path_to_clean_up_or_None). Extracts audio from
    video files to a temp WAV; passes audio files through unchanged."""
    ext = os.path.splitext(input_file)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        _extract_audio(input_file, tmp.name)
        return tmp.name, tmp.name
    return input_file, None


def _get_anthropic_client():
    from anthropic import Anthropic
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv())
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — add it to .env or the environment"
        )
    return Anthropic(api_key=api_key)


def _detect_boundaries(words: "list[dict]") -> "list[int]":
    prompt = build_boundary_prompt(words)
    client = _get_anthropic_client()
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        raise RuntimeError(f"Anthropic API call failed: {exc}")
    text = next((block.text for block in response.content if block.type == "text"), "")
    return parse_boundary_response(text, len(words))


def _detect_silence(wav_path: str) -> "tuple[list[tuple[float, float]], float]":
    """Return (silence_intervals, total_duration_seconds)."""
    import torch

    torch.set_num_threads(1)
    try:
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
    except Exception as exc:
        raise RuntimeError(f"failed to load Silero VAD model: {exc}")
    get_speech_timestamps, _, read_audio = utils[0], utils[1], utils[2]
    audio = read_audio(wav_path, sampling_rate=16000)
    total_duration = len(audio) / 16000.0
    speech_segments = get_speech_timestamps(
        audio, model, sampling_rate=16000, return_seconds=True,
        speech_pad_ms=0, min_silence_duration_ms=30,
        # time_resolution controls decimal places on returned timestamps;
        # the default (1 = rounded to 0.1s) is coarser than our 25fps frame
        # grid (40ms/frame) and can round two adjacent silence boundaries
        # to the same value, erasing short silences entirely. 3 = 1ms.
        time_resolution=3,
    )
    return invert_speech_to_silence(speech_segments, total_duration), total_duration


def run(
    input_file: str,
    aggressiveness: float = 0.5,
    min_pause_ms: float = 300,
    fps: int = 25,
    language: str = "English",
    verbose: bool = False,
    restrict_to_boundaries: bool = True,
) -> "tuple[list[str], float]":
    """Run the full pipeline and return (lines, total_pause_seconds):
    formatted "MM:SS:FF - MM:SS:FF" cut-range lines sorted by start time,
    and the summed duration of all cuts in seconds. Each range is
    half-open [start, end): the pause runs up to but not including the end
    timecode, matching the standard NLE in/out-point convention.

    restrict_to_boundaries (default True) gates cuts to word-gaps Claude
    flags as phrase/sentence boundaries. Set False to cut any VAD-confirmed
    silence regardless of grammar — every word-gap becomes a candidate and
    the Claude API call is skipped entirely."""
    words_path = _ensure_words_json(input_file, language, verbose)
    words = _load_words(words_path)
    if len(words) < 2:
        return [], 0.0

    if restrict_to_boundaries:
        boundary_indices = _detect_boundaries(words)
    else:
        boundary_indices = list(range(len(words) - 1))

    wav_path, tmp_wav = _get_wav_path(input_file)
    try:
        silence_intervals, total_duration = _detect_silence(wav_path)
    finally:
        if tmp_wav is not None and os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    cuts = select_cut_candidates(
        words, boundary_indices, silence_intervals,
        aggressiveness, fps, min_pause_ms / 1000.0, total_duration,
    )

    lines = [
        f"{frames_to_timecode(start, fps)} - {frames_to_timecode(end, fps)}"
        for start, end in cuts
    ]
    return lines, total_pause_seconds(cuts, fps)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Detect pauses to cut from an audio or video file, "
                    "using Claude for phrase/sentence boundaries and "
                    "Silero VAD to confirm silence."
    )
    parser.add_argument("input_file", help="Path to an audio or video file.")
    parser.add_argument(
        "--aggressiveness", "-a", type=float, default=0.5,
        help="0=conservative (larger end-of-pause buffer), "
             "1=aggressive (cut tight to the next word). Default: 0.5.",
    )
    parser.add_argument(
        "--min-pause", type=float, default=300,
        help="Minimum pause duration in milliseconds to cut. Default: 300.",
    )
    parser.add_argument("--fps", type=int, default=25, help="Frame rate. Default: 25.")
    parser.add_argument(
        "--language", "-l", default="English",
        help="Language name for the forced aligner (default: English).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to write the cut list (default: <stem>.cuts.txt).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show progress during inference."
    )
    parser.add_argument(
        "--allow-mid-phrase-cuts", action="store_true",
        help="Also cut pauses that aren't at a phrase/sentence boundary "
             "(skips the Claude grammatical-boundary check entirely; every "
             "VAD-confirmed silence becomes a candidate).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Error: file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    if not (0.0 <= args.aggressiveness <= 1.0):
        print("Error: --aggressiveness must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    try:
        lines, total_seconds = run(
            args.input_file,
            aggressiveness=args.aggressiveness,
            min_pause_ms=args.min_pause,
            fps=args.fps,
            language=args.language,
            verbose=args.verbose,
            restrict_to_boundaries=not args.allow_mid_phrase_cuts,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not lines:
        print("No pauses found to cut.")
    else:
        for line in lines:
            print(line)
        print(f"Total pause time removed: {format_duration(total_seconds)}")

    output_path = args.output or f"{os.path.splitext(args.input_file)[0]}.cuts.txt"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))

    print(f"Cut list written to {output_path}")


if __name__ == "__main__":
    main()
