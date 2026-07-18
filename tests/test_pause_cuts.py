import pytest

from premiere_ai.pause_cuts import frames_to_timecode, seconds_to_frame


def test_frames_to_timecode_zero():
    assert frames_to_timecode(0, fps=25) == "00:00:00"


def test_frames_to_timecode_frames_only():
    assert frames_to_timecode(10, fps=25) == "00:00:10"


def test_frames_to_timecode_rolls_into_seconds():
    assert frames_to_timecode(25, fps=25) == "00:01:00"


def test_frames_to_timecode_rolls_into_minutes():
    assert frames_to_timecode(60 * 25, fps=25) == "01:00:00"


def test_frames_to_timecode_mixed():
    # 1 minute, 2 seconds, 3 frames at 25fps
    frame = 60 * 25 + 2 * 25 + 3
    assert frames_to_timecode(frame, fps=25) == "01:02:03"


def test_frames_to_timecode_negative_raises():
    with pytest.raises(ValueError):
        frames_to_timecode(-1, fps=25)


def test_seconds_to_frame_rounds_to_nearest():
    assert seconds_to_frame(1.0, fps=25) == 25
    assert seconds_to_frame(0.5, fps=25) == 12  # round(12.5) -> 12 (banker's rounding)
    assert seconds_to_frame(0.04, fps=25) == 1  # exactly one frame
    assert seconds_to_frame(0.0, fps=25) == 0


from premiere_ai.pause_cuts import total_pause_seconds


def test_total_pause_seconds_sums_cut_durations():
    cuts = [(0, 25), (50, 75)]  # two 1-second cuts at 25fps
    assert total_pause_seconds(cuts, fps=25) == 2.0


def test_total_pause_seconds_empty_list_is_zero():
    assert total_pause_seconds([], fps=25) == 0.0


def test_total_pause_seconds_fractional_result():
    cuts = [(0, 10)]  # 10 frames at 25fps = 0.4s
    assert total_pause_seconds(cuts, fps=25) == pytest.approx(0.4)


from premiere_ai.pause_cuts import format_duration


def test_format_duration_zero():
    assert format_duration(0.0) == "0:00.0"


def test_format_duration_seconds_only():
    assert format_duration(5.44) == "0:05.4"  # rounds to nearest tenth


def test_format_duration_rolls_into_minutes():
    assert format_duration(65.96) == "1:06.0"


def test_format_duration_rounds_up_to_next_minute():
    # 59.96s rounds to 60.0s, which must carry into the minutes place
    assert format_duration(59.96) == "1:00.0"


def test_format_duration_negative_raises():
    with pytest.raises(ValueError):
        format_duration(-1.0)


from premiere_ai.pause_cuts import compute_pause_window


def test_compute_pause_window_basic_intersection():
    # word ends at 1.0s, next word starts at 2.0s, VAD silence covers 1.0-2.0
    window = compute_pause_window(
        word_end=1.0, next_word_start=2.0,
        vad_start=1.0, vad_end=2.0,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
    )
    assert window is not None
    start, end = window
    # start margin at aggressiveness 0.5 is round(2 * 0.5) = 1 frame forward from 25
    assert start == 25 + 1
    # end margin at aggressiveness 0.5 is round(6 * 0.5) = 3 frames back from 50
    assert end == 50 - 3


def test_compute_pause_window_aggressiveness_zero_keeps_full_margins():
    window = compute_pause_window(
        word_end=0.0, next_word_start=2.0,
        vad_start=0.0, vad_end=2.0,
        aggressiveness=0.0, fps=25, min_pause_seconds=0.1,
    )
    assert window is not None
    start, end = window
    assert start == 0 + 2  # full max_start_margin_frames at aggressiveness 0
    assert end == 50 - 6  # full max_end_margin_frames at aggressiveness 0


def test_compute_pause_window_aggressiveness_one_cuts_tight_to_both_edges():
    window = compute_pause_window(
        word_end=0.0, next_word_start=2.0,
        vad_start=0.0, vad_end=2.0,
        aggressiveness=1.0, fps=25, min_pause_seconds=0.1,
    )
    assert window is not None
    start, end = window
    assert start == 0  # zero start margin at aggressiveness 1
    assert end == 50  # zero end margin at aggressiveness 1


def test_compute_pause_window_uses_tighter_of_word_and_vad_bounds():
    # VAD silence is narrower than the word gap on both sides
    window = compute_pause_window(
        word_end=0.0, next_word_start=3.0,
        vad_start=1.0, vad_end=1.5,
        aggressiveness=0.0, fps=25, min_pause_seconds=0.01,
    )
    assert window is not None
    start, end = window
    assert start == 25 + 2  # from vad_start (1.0s), not word_end (0.0s)
    assert end == round(1.5 * 25) - 6  # from vad_end (1.5s), not next_word_start


def test_compute_pause_window_no_overlap_returns_none():
    # VAD silence entirely before the word gap
    window = compute_pause_window(
        word_end=2.0, next_word_start=3.0,
        vad_start=0.0, vad_end=1.0,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.1,
    )
    assert window is None


def test_compute_pause_window_below_min_duration_returns_none():
    window = compute_pause_window(
        word_end=0.0, next_word_start=0.2,
        vad_start=0.0, vad_end=0.2,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
    )
    assert window is None


def test_compute_pause_window_below_min_duration_after_positive_margins():
    # Post-margin window is positive but duration is still below min_pause_seconds.
    # Trace: raw_start=0.0, raw_end=0.5. start_frame = 0 + 2 = 2.
    # end_margin_frames = round(6*(1-0.0)) = 6. end_frame = 12 - 6 = 6.
    # end_frame(6) > start_frame(2), so margin-collapse guard passes.
    # Duration = 6 - 2 = 4 frames. min_pause_frames = round(0.3*25) = 8.
    # Since 4 < 8, must return None via min_pause_frames comparison.
    window = compute_pause_window(
        word_end=0.0, next_word_start=0.5,
        vad_start=0.0, vad_end=0.5,
        aggressiveness=0.0, fps=25, min_pause_seconds=0.3,
    )
    assert window is None


def test_compute_pause_window_margins_collapse_window_returns_none():
    # Window is nominally positive but margins eat past a min_pause of 0
    window = compute_pause_window(
        word_end=0.0, next_word_start=0.12,
        vad_start=0.0, vad_end=0.12,
        aggressiveness=0.0, fps=25, min_pause_seconds=0.0,
    )
    # raw window is 3 frames (0.12s * 25 = 3), start margin +2, end margin -6
    # start_frame = 2, end_frame = 3 - 6 = -3 -> end <= start -> None
    assert window is None


def test_compute_pause_window_allows_start_before_word_end_when_silence_starts_earlier():
    # Forced-alignment word boundaries are frequently imprecise: VAD may
    # detect silence starting inside the tail of the preceding word's
    # labeled span (word_end=1.0). That silence should be usable as the
    # pause-window start rather than clamped to word_end.
    window = compute_pause_window(
        word_end=1.0, next_word_start=2.0,
        vad_start=0.9, vad_end=2.0,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
    )
    assert window is not None
    start, end = window
    assert start == seconds_to_frame(0.9, 25) + 1  # from vad_start, not word_end
    assert end == seconds_to_frame(2.0, 25) - 3  # aggressiveness-0.5 end margin


from premiere_ai.pause_cuts import invert_speech_to_silence


def test_invert_speech_to_silence_basic_gaps():
    speech = [{"start": 1.0, "end": 2.0}, {"start": 3.0, "end": 4.0}]
    silence = invert_speech_to_silence(speech, total_duration=5.0)
    assert silence == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


def test_invert_speech_to_silence_no_speech():
    assert invert_speech_to_silence([], total_duration=5.0) == [(0.0, 5.0)]


def test_invert_speech_to_silence_speech_covers_everything():
    speech = [{"start": 0.0, "end": 5.0}]
    assert invert_speech_to_silence(speech, total_duration=5.0) == []


def test_invert_speech_to_silence_unsorted_input():
    speech = [{"start": 3.0, "end": 4.0}, {"start": 1.0, "end": 2.0}]
    silence = invert_speech_to_silence(speech, total_duration=5.0)
    assert silence == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


def test_invert_speech_to_silence_overlapping_segments_merged():
    speech = [{"start": 1.0, "end": 2.5}, {"start": 2.0, "end": 3.0}]
    silence = invert_speech_to_silence(speech, total_duration=4.0)
    assert silence == [(0.0, 1.0), (3.0, 4.0)]


def test_invert_speech_to_silence_no_trailing_silence():
    speech = [{"start": 0.0, "end": 3.0}]
    assert invert_speech_to_silence(speech, total_duration=3.0) == []


from premiere_ai.pause_cuts import build_boundary_prompt, parse_boundary_response


def test_build_boundary_prompt_includes_indexed_words():
    words = [{"text": "Hello"}, {"text": "world"}, {"text": "friend"}]
    prompt = build_boundary_prompt(words)
    assert "0:Hello" in prompt
    assert "1:world" in prompt
    assert "2:friend" in prompt
    assert "JSON array" in prompt


def test_build_boundary_prompt_states_highest_valid_index():
    # The last word (index len(words)-1) has no following gap, so the prompt
    # must tell Claude the highest index it may report is len(words)-2 —
    # otherwise Claude naturally flags the transcript-final word as a
    # "sentence boundary," which parse_boundary_response then rejects as
    # out-of-range and crashes the whole run.
    words = [{"text": "Hello"}, {"text": "world"}, {"text": "friend"}]
    prompt = build_boundary_prompt(words)
    assert "highest valid index to report is 1" in prompt


def test_build_boundary_prompt_states_punctuation_rules():
    words = [{"text": "Hello,"}, {"text": "world"}]
    prompt = build_boundary_prompt(words)
    assert "comma, semicolon, or colon" in prompt
    assert "always a boundary" in prompt
    assert "abbreviation" in prompt
    assert "Mr." in prompt


def test_parse_boundary_response_plain_json():
    assert parse_boundary_response("[0, 2, 5]", num_words=10) == [0, 2, 5]


def test_parse_boundary_response_sorts_and_dedupes():
    assert parse_boundary_response("[5, 0, 5, 2]", num_words=10) == [0, 2, 5]


def test_parse_boundary_response_strips_markdown_fence():
    raw = "```json\n[1, 2, 3]\n```"
    assert parse_boundary_response(raw, num_words=10) == [1, 2, 3]


def test_parse_boundary_response_strips_plain_fence():
    raw = "```\n[1, 2]\n```"
    assert parse_boundary_response(raw, num_words=10) == [1, 2]


def test_parse_boundary_response_empty_array():
    assert parse_boundary_response("[]", num_words=10) == []


def test_parse_boundary_response_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_boundary_response("not json at all", num_words=10)


def test_parse_boundary_response_non_array_raises():
    with pytest.raises(ValueError):
        parse_boundary_response('{"indices": [1, 2]}', num_words=10)


def test_parse_boundary_response_non_integer_items_raises():
    with pytest.raises(ValueError):
        parse_boundary_response('[1, "two"]', num_words=10)


def test_parse_boundary_response_drops_indices_above_max_valid():
    # num_words=3 means valid gap indices are 0 and 1 (gap after word i
    # requires word i+1 to exist). Index 2 (the last word) is a predictable
    # near-miss Claude makes despite the prompt — dropped, not raised.
    assert parse_boundary_response("[0, 2]", num_words=3) == [0]


def test_parse_boundary_response_all_indices_above_max_valid_returns_empty():
    assert parse_boundary_response("[2]", num_words=3) == []


def test_parse_boundary_response_negative_index_raises():
    with pytest.raises(ValueError):
        parse_boundary_response("[-1]", num_words=10)


from premiere_ai.pause_cuts import merge_overlapping_cuts


def test_merge_overlapping_cuts_empty_list():
    assert merge_overlapping_cuts([]) == []


def test_merge_overlapping_cuts_no_overlap_unchanged():
    cuts = [(0, 10), (20, 30)]
    assert merge_overlapping_cuts(cuts) == [(0, 10), (20, 30)]


def test_merge_overlapping_cuts_merges_overlapping_pair():
    cuts = [(0, 10), (5, 20)]
    assert merge_overlapping_cuts(cuts) == [(0, 20)]


def test_merge_overlapping_cuts_merges_fully_contained_window():
    # (5, 8) is entirely inside (0, 20) — the exact shape produced when one
    # wide VAD silence interval spans two adjacent word-gaps.
    cuts = [(0, 20), (5, 8)]
    assert merge_overlapping_cuts(cuts) == [(0, 20)]


def test_merge_overlapping_cuts_merges_touching_windows():
    # end of the first == start of the second: adjacent, not overlapping,
    # but still merged into one continuous cut.
    cuts = [(0, 10), (10, 20)]
    assert merge_overlapping_cuts(cuts) == [(0, 20)]


def test_merge_overlapping_cuts_sorts_unsorted_input():
    cuts = [(20, 30), (0, 10)]
    assert merge_overlapping_cuts(cuts) == [(0, 10), (20, 30)]


def test_merge_overlapping_cuts_chains_three_overlapping_windows():
    cuts = [(0, 10), (5, 15), (12, 25)]
    assert merge_overlapping_cuts(cuts) == [(0, 25)]


from premiere_ai.pause_cuts import select_cut_candidates


def _words(*pairs):
    return [{"text": f"w{i}", "start": s, "end": e} for i, (s, e) in enumerate(pairs)]


def test_select_cut_candidates_single_boundary_with_silence():
    words = _words((0.0, 1.0), (2.0, 3.0))  # gap 1.0-2.0 after word 0
    silence = [(1.0, 2.0)]
    cuts = select_cut_candidates(
        words, boundary_indices=[0], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=words[-1]["end"],  # no trailing gap to consider
    )
    assert cuts == [(26, 47)]  # matches compute_pause_window's own math


def test_select_cut_candidates_ignores_non_boundary_gaps():
    words = _words((0.0, 1.0), (2.0, 3.0), (4.0, 5.0))
    silence = [(1.0, 2.0), (3.0, 4.0)]
    # Only gap after word 0 is a boundary; gap after word 1 is not, even
    # though it's silent.
    cuts = select_cut_candidates(
        words, boundary_indices=[0], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=words[-1]["end"],
    )
    assert len(cuts) == 1


def test_select_cut_candidates_boundary_without_silence_is_dropped():
    words = _words((0.0, 1.0), (2.0, 3.0))
    # No silence interval overlaps the 1.0-2.0 gap at all.
    cuts = select_cut_candidates(
        words, boundary_indices=[0], silence_intervals=[(5.0, 6.0)],
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=words[-1]["end"],
    )
    assert cuts == []


def test_select_cut_candidates_out_of_range_boundary_index_ignored():
    words = _words((0.0, 1.0), (2.0, 3.0))
    cuts = select_cut_candidates(
        words, boundary_indices=[5], silence_intervals=[(1.0, 2.0)],
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=words[-1]["end"],
    )
    assert cuts == []


def test_select_cut_candidates_multiple_boundaries_sorted_by_start():
    words = _words((0.0, 1.0), (2.0, 3.0), (4.0, 5.0))
    silence = [(1.0, 2.0), (3.0, 4.0)]
    cuts = select_cut_candidates(
        words, boundary_indices=[1, 0], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=words[-1]["end"],
    )
    assert len(cuts) == 2
    assert cuts == sorted(cuts)
    assert cuts[0][0] < cuts[1][0]


def test_select_cut_candidates_below_min_duration_dropped():
    words = _words((0.0, 1.0), (2.0, 1.35))  # 350ms gap before margins
    cuts = select_cut_candidates(
        words, boundary_indices=[0], silence_intervals=[(1.0, 1.35)],
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=words[-1]["end"],
    )
    assert cuts == []


def test_select_cut_candidates_start_can_precede_word_end_when_vad_confirms_earlier_silence():
    words = _words((0.0, 1.0), (2.0, 3.0))
    # VAD silence starts at 0.9s, inside word 0's labeled span (ends 1.0s).
    silence = [(0.9, 2.0)]
    cuts = select_cut_candidates(
        words, boundary_indices=[0], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=words[-1]["end"],
    )
    assert cuts == [(seconds_to_frame(0.9, 25) + 1, seconds_to_frame(2.0, 25) - 3)]


def test_select_cut_candidates_finds_leading_silence_before_first_word():
    words = _words((1.0, 2.0))
    silence = [(0.0, 1.0)]
    cuts = select_cut_candidates(
        words, boundary_indices=[], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=2.0,
    )
    assert cuts == [(1, 22)]


def test_select_cut_candidates_no_leading_cut_when_first_word_starts_at_zero():
    words = _words((0.0, 1.0))
    silence = []
    cuts = select_cut_candidates(
        words, boundary_indices=[], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=1.0,
    )
    assert cuts == []


def test_select_cut_candidates_no_leading_cut_without_silence():
    words = _words((1.0, 2.0))
    cuts = select_cut_candidates(
        words, boundary_indices=[], silence_intervals=[],
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=2.0,
    )
    assert cuts == []


def test_select_cut_candidates_leading_cut_respects_min_duration():
    words = _words((1.0, 2.0))
    silence = [(0.9, 1.0)]  # 100ms leading silence
    cuts = select_cut_candidates(
        words, boundary_indices=[], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=2.0,
    )
    assert cuts == []


def test_select_cut_candidates_finds_trailing_silence_after_last_word():
    words = _words((0.0, 1.0))
    silence = [(1.0, 2.0)]
    cuts = select_cut_candidates(
        words, boundary_indices=[], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=2.0,
    )
    assert cuts == [(26, 47)]  # same math as the basic-intersection case


def test_select_cut_candidates_no_trailing_cut_without_silence():
    words = _words((0.0, 1.0))
    cuts = select_cut_candidates(
        words, boundary_indices=[], silence_intervals=[],
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=2.0,
    )
    assert cuts == []


def test_select_cut_candidates_trailing_cut_respects_min_duration():
    words = _words((0.0, 1.0))
    silence = [(1.0, 1.1)]  # 100ms trailing silence
    cuts = select_cut_candidates(
        words, boundary_indices=[], silence_intervals=silence,
        aggressiveness=0.5, fps=25, min_pause_seconds=0.3,
        total_duration=1.1,
    )
    assert cuts == []


def test_select_cut_candidates_merges_overlapping_windows_from_shared_silence():
    # A single wide VAD silence interval (1.0-3.0) spans two adjacent
    # word-gaps around a very short middle word (w1, 1.05-1.1). Each gap's
    # window is computed independently and can come out overlapping (gap 1's
    # start is the same vad_start=1.0 used for gap 0, since compute_pause_window
    # trusts VAD over the word timestamps) — this must not surface as two
    # overlapping ranges in the final cut list.
    words = _words((0.0, 1.0), (1.05, 1.1), (3.0, 4.0))
    silence = [(1.0, 3.0)]
    cuts = select_cut_candidates(
        words, boundary_indices=[0, 1], silence_intervals=silence,
        aggressiveness=1.0, fps=25, min_pause_seconds=0.0,
        total_duration=words[-1]["end"],
    )
    assert cuts == [(25, 75)]  # gap 0's (25, 26) is fully absorbed into gap 1's (25, 75)
