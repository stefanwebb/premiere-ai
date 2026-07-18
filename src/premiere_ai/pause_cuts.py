"""Pure logic for detecting and formatting pause cuts. No I/O, no
external-process calls — everything here is unit-testable with plain data.
"""

import json
import re


def frames_to_timecode(frame: int, fps: int) -> str:
    """Format a frame count as MM:SS:FF."""
    if frame < 0:
        raise ValueError("frame must be non-negative")
    total_seconds, ff = divmod(frame, fps)
    mm, ss = divmod(total_seconds, 60)
    return f"{mm:02d}:{ss:02d}:{ff:02d}"


def seconds_to_frame(seconds: float, fps: int) -> int:
    """Convert a time in seconds to the nearest frame index."""
    return round(seconds * fps)


def total_pause_seconds(cuts: "list[tuple[int, int]]", fps: int) -> float:
    """Sum the duration in seconds of a list of half-open (start_frame,
    end_frame) cut windows."""
    return sum(end - start for start, end in cuts) / fps


def format_duration(seconds: float) -> str:
    """Format a duration in seconds as "M:SS.s" (minutes, seconds rounded
    to the nearest tenth)."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    total_tenths = round(seconds * 10)
    minutes, remainder_tenths = divmod(total_tenths, 600)
    return f"{minutes}:{remainder_tenths / 10:04.1f}"


def compute_pause_window(
    word_end: float,
    next_word_start: float,
    vad_start: float,
    vad_end: float,
    aggressiveness: float,
    fps: int,
    min_pause_seconds: float,
    max_start_margin_frames: int = 2,
    max_end_margin_frames: int = 6,
) -> "tuple[int, int] | None":
    """Compute the cuttable pause window in frames, or None if there is no
    usable window.

    The returned (start_frame, end_frame) is a half-open range
    [start_frame, end_frame): end_frame is the first frame that remains,
    not the last frame cut.

    The window starts at vad_start — including inside the tail of the
    preceding word's labeled span if VAD confirms silence has already begun
    there, since forced-alignment word boundaries are frequently imprecise
    and VAD is the more trustworthy signal for actual silence. word_end is
    used only to confirm the VAD interval genuinely overlaps this gap (not
    some unrelated silence entirely before it). The window is then narrowed
    by two asymmetric margins, both scaled by aggressiveness (0=largest
    margins/most conservative, 1=zero margin/cuts tight to both edges):

    - start margin: pulled forward from the pause start (which follows
      known speech — a buffer at low aggressiveness avoids clipping the
      tail of whatever speech precedes it). Smaller than the end margin by
      default since this edge is less risky to cut close to.
    - end margin: pulled backward from the pause end (which precedes the
      next word — a larger buffer at low aggressiveness avoids clipping
      the start of the next word).
    """
    if vad_end <= word_end or vad_start >= next_word_start:
        return None  # this VAD interval doesn't actually overlap the gap

    raw_start = vad_start
    raw_end = min(next_word_start, vad_end)

    start_margin_frames = round(max_start_margin_frames * (1 - aggressiveness))
    end_margin_frames = round(max_end_margin_frames * (1 - aggressiveness))

    start_frame = seconds_to_frame(raw_start, fps) + start_margin_frames
    end_frame = seconds_to_frame(raw_end, fps) - end_margin_frames

    if end_frame <= start_frame:
        return None

    min_pause_frames = round(min_pause_seconds * fps)
    if end_frame - start_frame < min_pause_frames:
        return None

    return start_frame, end_frame


def invert_speech_to_silence(
    speech_segments: "list[dict]",
    total_duration: float,
) -> "list[tuple[float, float]]":
    """Invert a list of {"start": float, "end": float} speech segments
    (seconds, possibly unsorted or overlapping) into sorted, non-overlapping
    silence intervals covering the gaps between/around speech."""
    if not speech_segments:
        return [(0.0, total_duration)] if total_duration > 0 else []

    segments = sorted(
        ({"start": s["start"], "end": s["end"]} for s in speech_segments),
        key=lambda s: s["start"],
    )

    merged: "list[dict]" = []
    for seg in segments:
        if merged and seg["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
        else:
            merged.append(dict(seg))

    silence: "list[tuple[float, float]]" = []
    cursor = 0.0
    for seg in merged:
        if seg["start"] > cursor:
            silence.append((cursor, seg["start"]))
        cursor = max(cursor, seg["end"])
    if cursor < total_duration:
        silence.append((cursor, total_duration))

    return silence


def build_boundary_prompt(words: "list[dict]") -> str:
    """Build the prompt sent to the `claude` CLI to identify which word gaps
    are phrase/sentence boundaries (candidates for cutting)."""
    numbered = " ".join(f"{i}:{w['text']}" for i, w in enumerate(words))
    max_valid_index = len(words) - 2
    return (
        "Below is a transcript as space-separated \"index:word\" tokens. "
        "A \"gap\" after word i is the space between word i and word i+1. "
        "Identify every gap that falls at a phrase or sentence boundary "
        "(a natural place to pause or cut), as opposed to a gap in the "
        "middle of a phrase (e.g. a mid-thought hesitation). "
        "Punctuation is a strong signal: if word i ends with a comma, "
        "semicolon, or colon, the gap after it is always a boundary. If "
        "word i ends with a period, the gap after it is a boundary too, "
        "unless that period is part of an abbreviation rather than "
        "ending a sentence (e.g. \"Mr.\", \"Dr.\", \"U.S.\", \"etc.\") — "
        "abbreviation periods are not boundaries. "
        f"The transcript has {len(words)} words (indices 0 to {len(words) - 1}); "
        f"the last word, index {len(words) - 1}, has no following gap, so the "
        f"highest valid index to report is {max_valid_index}. "
        "Respond with ONLY a JSON array of integers, each the index i of a "
        "word whose following gap is a boundary. No other text.\n\n"
        f"{numbered}"
    )


def parse_boundary_response(raw_text: str, num_words: int) -> "list[int]":
    """Parse the `claude` CLI's response into a sorted, deduped list of
    boundary word-gap indices. Raises ValueError on malformed responses
    (invalid JSON, wrong shape, non-integer items, negative indices).

    Indices above the valid range are dropped rather than raised: the last
    word (index num_words-1) reads as an obvious sentence boundary to
    Claude — especially once punctuation is visible — so it is flagged
    despite the prompt's explicit instruction not to. That single
    structurally-impossible index (there's no word after the last one) is
    a predictable near-miss, not a sign the rest of the response is
    unreliable, so only it is filtered."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude response is not valid JSON: {raw_text!r}") from exc

    if not isinstance(data, list) or not all(
        isinstance(x, int) and not isinstance(x, bool) for x in data
    ):
        raise ValueError(
            f"Claude response is not a JSON array of integers: {raw_text!r}"
        )

    negative = [i for i in data if i < 0]
    if negative:
        raise ValueError(f"Boundary indices out of range: {negative}")

    max_valid_index = num_words - 2  # gap after word i requires word i+1 to exist
    in_range = [i for i in data if i <= max_valid_index]

    return sorted(set(in_range))


def _merge_overlapping_silence(
    word_end: float,
    next_word_start: float,
    silence_intervals: "list[tuple[float, float]]",
) -> "tuple[float, float] | None":
    """Merge every silence interval overlapping [word_end, next_word_start]
    into a single (vad_start, vad_end) bound, or None if none overlap."""
    overlapping = [
        (vad_start, vad_end)
        for vad_start, vad_end in silence_intervals
        if vad_end > word_end and vad_start < next_word_start
    ]
    if not overlapping:
        return None
    return min(s for s, _ in overlapping), max(e for _, e in overlapping)


def merge_overlapping_cuts(
    cuts: "list[tuple[int, int]]",
) -> "list[tuple[int, int]]":
    """Merge overlapping or touching half-open (start_frame, end_frame)
    cut windows into a sorted, non-overlapping list. Two windows merge if
    the next one's start is <= the current merged window's end (covers
    both genuine overlap and back-to-back adjacency)."""
    if not cuts:
        return []

    sorted_cuts = sorted(cuts)
    merged = [sorted_cuts[0]]
    for start, end in sorted_cuts[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def select_cut_candidates(
    words: "list[dict]",
    boundary_indices: "list[int]",
    silence_intervals: "list[tuple[float, float]]",
    aggressiveness: float,
    fps: int,
    min_pause_seconds: float,
    total_duration: float,
) -> "list[tuple[int, int]]":
    """Combine Claude's boundary gating, VAD silence overlap, and the
    aggressiveness-scaled margin/min-duration filtering (compute_pause_window)
    into the final sorted, non-overlapping list of (start_frame, end_frame)
    cut windows (overlapping/touching windows are merged — see
    merge_overlapping_cuts).

    Each window is a half-open frame range [start_frame, end_frame): the
    pause spans frames start_frame through end_frame - 1, and end_frame is
    the first frame that remains (matches the standard NLE in/out-point
    convention where duration = end - start).

    Leading silence before the first word, and trailing silence after the
    last word (up to total_duration), are always candidates too, without
    boundary gating — there is no preceding/following phrase to protect
    against clipping at either edge of the file."""
    cuts: "list[tuple[int, int]]" = []

    if words:
        first_word_start = words[0]["start"]
        merged = _merge_overlapping_silence(0.0, first_word_start, silence_intervals)
        if merged is not None:
            vad_start, vad_end = merged
            window = compute_pause_window(
                0.0, first_word_start, vad_start, vad_end,
                aggressiveness, fps, min_pause_seconds,
            )
            if window is not None:
                cuts.append(window)

    for i in boundary_indices:
        if i < 0 or i + 1 >= len(words):
            continue
        word_end = words[i]["end"]
        next_word_start = words[i + 1]["start"]

        merged = _merge_overlapping_silence(word_end, next_word_start, silence_intervals)
        if merged is None:
            continue
        vad_start, vad_end = merged

        window = compute_pause_window(
            word_end, next_word_start, vad_start, vad_end,
            aggressiveness, fps, min_pause_seconds,
        )
        if window is not None:
            cuts.append(window)

    if words:
        last_word_end = words[-1]["end"]
        merged = _merge_overlapping_silence(last_word_end, total_duration, silence_intervals)
        if merged is not None:
            vad_start, vad_end = merged
            window = compute_pause_window(
                last_word_end, total_duration, vad_start, vad_end,
                aggressiveness, fps, min_pause_seconds,
            )
            if window is not None:
                cuts.append(window)

    return merge_overlapping_cuts(cuts)
