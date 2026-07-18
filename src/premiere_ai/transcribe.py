import argparse
import json
import os
import subprocess
import sys
import tempfile

MODEL_ID = "mlx-community/VibeVoice-ASR-4bit"
ALIGNER_MODEL_ID = "mlx-community/Qwen3-ForcedAligner-0.6B-4bit"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a", ".opus", ".aiff"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mts", ".ts"}


def _hf_cache_dir() -> str:
    return (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    )


def _model_cached(model_id: str) -> bool:
    model_dir = model_id.replace("/", "--")
    return os.path.isdir(os.path.join(_hf_cache_dir(), f"models--{model_dir}"))


def _check_dependencies() -> bool:
    ok = True

    if not _model_cached(MODEL_ID):
        print(
            f"Error: model '{MODEL_ID}' is not downloaded.\n"
            f"Run the following command to download it:\n\n"
            f"  hf download {MODEL_ID}\n",
            file=sys.stderr,
        )
        ok = False

    if not _model_cached(ALIGNER_MODEL_ID):
        print(
            f"Error: model '{ALIGNER_MODEL_ID}' is not downloaded.\n"
            f"Run the following command to download it:\n\n"
            f"  hf download {ALIGNER_MODEL_ID}\n",
            file=sys.stderr,
        )
        ok = False

    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        print(
            "Error: mlx-audio is not installed.\n"
            "Install it with:\n\n"
            "  pip install -U mlx-audio\n",
            file=sys.stderr,
        )
        ok = False

    return ok


def _extract_audio(video_path: str, wav_path: str) -> None:
    """Extract audio track from a video file to a WAV using ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", wav_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: ffmpeg failed to extract audio:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def _align_segments(aligner_model, audio_path: str, segments, language: str) -> list:
    """Run forced alignment on each ASR segment to produce word-level timestamps."""
    import numpy as np
    from mlx_audio.stt.utils import load_audio

    # Load full audio at 16 kHz once, then slice per segment
    audio_mx = load_audio(audio_path)
    audio_np = np.array(audio_mx)
    sample_rate = 16000

    words = []
    seg_list = segments.segments if hasattr(segments, "segments") and segments.segments else []

    for seg in seg_list:
        text = seg["text"].strip()
        if not text:
            continue

        start_s: float = seg["start"]
        end_s: float = seg["end"]
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        audio_slice = audio_np[start_idx:end_idx]

        align_result = aligner_model.generate(audio_slice, text=text, language=language)

        # The aligner returns clean word tokens with punctuation stripped.
        # If its token count matches a plain whitespace split of the
        # original (punctuated) segment text, use those raw tokens instead
        # so punctuation survives into words.json — downstream consumers
        # (e.g. remove-pauses' Claude boundary detection) rely on it to
        # recognize phrase/clause boundaries that capitalization alone
        # can't signal. Falls back to the aligner's own text on mismatch.
        raw_tokens = text.split()
        punctuation_aligned = len(raw_tokens) == len(align_result)

        for idx, item in enumerate(align_result):
            word_text = raw_tokens[idx] if punctuation_aligned else item.text
            words.append({
                "text": word_text,
                "start": round(start_s + item.start_time, 3),
                "end": round(start_s + item.end_time, 3),
            })

    return words


def transcribe_file(
    input_file: str,
    output: "str | None" = None,
    format: str = "txt",
    language: str = "English",
    verbose: bool = False,
) -> "tuple[str, str]":
    """Transcribe input_file and write transcript + word-timestamp outputs.

    Returns (transcript_path, words_path). Raises FileNotFoundError if
    input_file doesn't exist, RuntimeError if dependencies are missing.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"file not found: {input_file}")

    if not _check_dependencies():
        raise RuntimeError("missing transcribe dependencies")

    ext = os.path.splitext(input_file)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS
    is_audio = ext in AUDIO_EXTENSIONS

    if not is_video and not is_audio:
        print(
            f"Warning: unrecognised extension '{ext}', treating as audio.",
            file=sys.stderr,
        )

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    from mlx_audio.stt.generate import generate_transcription
    from mlx_audio.stt.utils import load_model

    print(f"Loading model {MODEL_ID} …")
    model = load_model(MODEL_ID)

    tmp = None
    try:
        if is_video:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            print(f"Extracting audio from {input_file} …")
            _extract_audio(input_file, tmp.name)
            audio_path = tmp.name
        else:
            audio_path = input_file

        # generate_transcription appends the format extension itself,
        # so output_path must be a stem (no extension).
        if output:
            output_path = os.path.splitext(output)[0]
        else:
            output_path = os.path.splitext(input_file)[0]

        print("Transcribing …")
        segments = generate_transcription(
            model=model,
            audio=audio_path,
            output_path=output_path,
            format=format,
            verbose=verbose,
        )

        print(f"Loading aligner model {ALIGNER_MODEL_ID} …")
        aligner = load_model(ALIGNER_MODEL_ID)

        print("Aligning …")
        words = _align_segments(aligner, audio_path, segments, language)

        words_path = f"{output_path}.words.json"
        with open(words_path, "w", encoding="utf-8") as fh:
            json.dump(words, fh, ensure_ascii=False, indent=2)

    finally:
        if tmp is not None and os.path.exists(tmp.name):
            os.unlink(tmp.name)

    transcript_path = f"{output_path}.{format}"
    print(f"Transcript written to {transcript_path}")
    print(f"Word timestamps written to {words_path}")
    return transcript_path, words_path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio or video using VibeVoice-ASR (MLX, 4-bit), "
                    "then obtain word-level timestamps via Qwen3-ForcedAligner."
    )
    parser.add_argument("input_file", help="Path to an audio or video file to transcribe.")
    parser.add_argument(
        "--output", "-o", default=None, help="Path to write the transcript (optional)."
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["txt", "srt", "vtt"],
        default="txt",
        help="Output format (default: txt).",
    )
    parser.add_argument(
        "--language", "-l", default="English",
        help="Language name for the forced aligner (default: English).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show progress during inference."
    )
    args = parser.parse_args()

    try:
        transcribe_file(
            args.input_file,
            output=args.output,
            format=args.format,
            language=args.language,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
