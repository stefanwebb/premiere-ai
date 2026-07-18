import json
import sys
import types

import numpy as np
import pytest

from premiere_ai import sync_audio


def test_align_words_matches_common_subsequence_case_and_punctuation_insensitive():
    camera_words = [
        {"text": "Hello,", "start": 0.0, "end": 0.4},
        {"text": "world!", "start": 0.5, "end": 1.0},
        {"text": "foo", "start": 1.1, "end": 1.4},
        {"text": "bar", "start": 1.5, "end": 1.8},
    ]
    mic_words = [
        {"text": "hello", "start": 0.05, "end": 0.45},
        {"text": "there", "start": 0.46, "end": 0.5},
        {"text": "World!", "start": 0.55, "end": 1.05},
        {"text": "foo", "start": 1.15, "end": 1.45},
        {"text": "qux", "start": 1.46, "end": 1.5},
        {"text": "bar", "start": 1.55, "end": 1.85},
    ]

    pairs = sync_audio._align_words(camera_words, mic_words)

    assert pairs == [
        (camera_words[0], mic_words[0]),
        (camera_words[1], mic_words[2]),
        (camera_words[2], mic_words[3]),
        (camera_words[3], mic_words[5]),
    ]


def test_align_words_returns_empty_list_when_nothing_matches():
    camera_words = [{"text": "apple", "start": 0.0, "end": 0.3}]
    mic_words = [{"text": "zebra", "start": 0.0, "end": 0.3}]

    assert sync_audio._align_words(camera_words, mic_words) == []


def _pair(camera_start):
    return ({"text": "w", "start": camera_start, "end": camera_start + 0.2}, {"text": "w", "start": camera_start, "end": camera_start + 0.2})


def test_pick_anchors_selects_nearest_pair_per_fixed_time_step():
    matched_pairs = [_pair(t) for t in [0, 10, 20, 32, 41, 58, 61]]

    anchors = sync_audio._pick_anchors(matched_pairs, anchor_spacing_seconds=25.0)

    camera_starts = [pair[0]["start"] for pair in anchors]
    assert camera_starts == [0, 20, 58]


def test_pick_anchors_returns_empty_list_for_no_matched_pairs():
    assert sync_audio._pick_anchors([], anchor_spacing_seconds=25.0) == []


def test_compute_anchor_offset_recovers_known_sub_frame_shift():
    sample_rate = 1000  # samples/sec — synthetic, for fast/simple test math
    rng = np.random.RandomState(42)
    source = rng.randn(5000).astype(np.float64)

    true_shift_samples = 13  # 13ms at 1000Hz — well under one video frame (40ms @ 25fps)
    camera_audio = source
    mic_audio = np.concatenate([np.zeros(true_shift_samples), source])

    camera_time_seconds = 2.5
    # Deliberately jittered "approximate" mic timestamp (simulating ASR
    # imprecision) — the true offset is 0.013s, but we feed in an approx
    # that's off by 7ms. The search range (0.05s) comfortably covers that.
    mic_time_approx_seconds = 2.5 + 0.013 + 0.007

    result = sync_audio._compute_anchor_offset(
        camera_audio, mic_audio, sample_rate,
        camera_time_seconds, mic_time_approx_seconds,
        window_seconds=1.0, search_range_seconds=0.05,
    )

    assert result is not None
    offset_seconds, confidence = result
    assert abs(offset_seconds - 0.013) < 1e-9
    assert confidence > 0.999


def test_compute_anchor_offset_returns_none_when_window_runs_off_start():
    sample_rate = 1000
    camera_audio = np.zeros(5000)
    mic_audio = np.zeros(5000)

    result = sync_audio._compute_anchor_offset(
        camera_audio, mic_audio, sample_rate,
        camera_time_seconds=0.05, mic_time_approx_seconds=0.06,
        window_seconds=1.0, search_range_seconds=0.05,
    )

    assert result is None


def test_fit_drift_line_recovers_known_linear_drift():
    slope = 0.0002  # seconds of offset drift per second elapsed
    intercept = 0.1
    anchors = [
        {"cameraSeconds": t, "offsetSeconds": intercept + slope * t, "confidence": 0.9}
        for t in [0, 30, 60, 90, 120]
    ]

    fit = sync_audio._fit_drift_line(anchors)

    assert fit is not None
    assert abs(fit["slopeSecondsPerSecond"] - slope) < 1e-9
    assert abs(fit["interceptSeconds"] - intercept) < 1e-9
    assert abs(fit["driftMsPerMinute"] - 12.0) < 1e-6


def test_fit_drift_line_returns_none_below_three_anchors():
    anchors = [
        {"cameraSeconds": 0, "offsetSeconds": 0.1, "confidence": 0.9},
        {"cameraSeconds": 30, "offsetSeconds": 0.1, "confidence": 0.9},
    ]

    assert sync_audio._fit_drift_line(anchors) is None


def _install_fake_mlx_load_audio(monkeypatch, fake_array):
    fake_utils_mod = types.ModuleType("mlx_audio.stt.utils")
    fake_utils_mod.load_audio = lambda path: fake_array
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", types.ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_utils_mod)


def test_load_audio_for_correlation_default_uses_16khz_mlx_path(tmp_path, monkeypatch):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"\x00")
    _install_fake_mlx_load_audio(monkeypatch, np.array([1.0, 2.0, 3.0]))

    samples, sample_rate = sync_audio._load_audio_for_correlation(str(audio_path), high_fidelity=False)

    assert sample_rate == 16000
    assert list(samples) == [1.0, 2.0, 3.0]


def test_load_audio_for_correlation_high_fidelity_uses_torchaudio_native_rate(tmp_path, monkeypatch):
    import torch

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"\x00")

    fake_torchaudio_mod = types.ModuleType("torchaudio")
    fake_torchaudio_mod.load = lambda path: (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 48000)
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio_mod)

    samples, sample_rate = sync_audio._load_audio_for_correlation(str(audio_path), high_fidelity=True)

    assert sample_rate == 48000
    # Two channels [1,2] and [3,4] averaged to mono: [2,3]
    assert list(samples) == [2.0, 3.0]


def test_compute_sync_offsets_missing_camera_file_raises(tmp_path):
    mic_path = tmp_path / "mic.wav"
    mic_path.write_bytes(b"\x00")

    with pytest.raises(FileNotFoundError):
        sync_audio.compute_sync_offsets(str(tmp_path / "no_such_camera.mp4"), str(mic_path))


def test_compute_sync_offsets_zero_word_matches_raises_runtime_error(tmp_path):
    camera_path = tmp_path / "camera.mp4"
    camera_path.write_bytes(b"\x00")
    mic_path = tmp_path / "mic.wav"
    mic_path.write_bytes(b"\x00")

    camera_words_path = tmp_path / "camera.words.json"
    camera_words_path.write_text(json.dumps([{"text": "apple", "start": 0.0, "end": 0.3}]))
    mic_words_path = tmp_path / "mic.words.json"
    mic_words_path.write_text(json.dumps([{"text": "zebra", "start": 0.0, "end": 0.3}]))

    with pytest.raises(RuntimeError):
        sync_audio.compute_sync_offsets(
            str(camera_path), str(mic_path),
            camera_words=str(camera_words_path), mic_words=str(mic_words_path),
        )


def _build_shifted_recordings(sample_rate, total_samples, true_shift_samples, seed=7):
    rng = np.random.RandomState(seed)
    source = rng.randn(total_samples + true_shift_samples).astype(np.float64)
    camera_audio = source[:total_samples]
    # The same content appears true_shift_samples later in mic_audio than in
    # camera_audio -- i.e. mic_audio[i] == camera_audio[i - true_shift_samples].
    mic_audio = np.concatenate([np.zeros(true_shift_samples), source[:total_samples]])
    return camera_audio, mic_audio


def test_compute_sync_offsets_happy_path_recovers_offset_and_drift_fit(tmp_path, monkeypatch):
    sample_rate = 1000
    total_samples = 130_000  # 130 seconds
    true_shift_samples = 20  # constant 0.02s offset, no drift
    camera_audio, mic_audio = _build_shifted_recordings(sample_rate, total_samples, true_shift_samples)

    camera_times = [10.0, 40.0, 70.0, 100.0, 120.0]
    jitters = [0.05, -0.03, 0.02, 0.04, -0.01]
    true_offset_seconds = true_shift_samples / sample_rate

    camera_words = [
        {"text": f"word{i}", "start": t, "end": t + 0.3} for i, t in enumerate(camera_times)
    ]
    mic_words = [
        {"text": f"word{i}", "start": t + true_offset_seconds + jitters[i], "end": t + true_offset_seconds + jitters[i] + 0.3}
        for i, t in enumerate(camera_times)
    ]

    camera_path = tmp_path / "camera.mp4"
    camera_path.write_bytes(b"\x00")
    mic_path = tmp_path / "mic.wav"
    mic_path.write_bytes(b"\x00")
    camera_words_path = tmp_path / "camera.words.json"
    camera_words_path.write_text(json.dumps(camera_words))
    mic_words_path = tmp_path / "mic.words.json"
    mic_words_path.write_text(json.dumps(mic_words))

    def fake_loader(media_path, high_fidelity):
        if media_path == str(camera_path):
            return camera_audio, sample_rate
        if media_path == str(mic_path):
            return mic_audio, sample_rate
        raise AssertionError(f"unexpected media_path: {media_path}")

    monkeypatch.setattr(sync_audio, "_load_audio_for_correlation", fake_loader)

    report = sync_audio.compute_sync_offsets(
        str(camera_path), str(mic_path),
        camera_words=str(camera_words_path), mic_words=str(mic_words_path),
        anchor_spacing_seconds=15.0,
    )

    assert len(report["anchors"]) == 5
    for anchor in report["anchors"]:
        assert abs(anchor["offsetSeconds"] - true_offset_seconds) < 1e-6
        assert anchor["confidence"] > 0.99
    assert report["droppedLowConfidenceCount"] == 0
    assert report["warnings"] == []
    assert report["driftFit"] is not None
    assert abs(report["driftFit"]["slopeSecondsPerSecond"]) < 1e-6
    assert abs(report["driftFit"]["interceptSeconds"] - true_offset_seconds) < 1e-6
    assert abs(report["recommendedOffsetSeconds"] - true_offset_seconds) < 1e-6


def test_compute_sync_offsets_insufficient_anchors_falls_back_to_median(tmp_path, monkeypatch):
    sample_rate = 1000
    total_samples = 60_000
    true_shift_samples = 20
    camera_audio, mic_audio = _build_shifted_recordings(sample_rate, total_samples, true_shift_samples)
    true_offset_seconds = true_shift_samples / sample_rate

    camera_times = [10.0, 40.0]  # only 2 anchors possible
    camera_words = [
        {"text": f"word{i}", "start": t, "end": t + 0.3} for i, t in enumerate(camera_times)
    ]
    mic_words = [
        {"text": f"word{i}", "start": t + true_offset_seconds, "end": t + true_offset_seconds + 0.3}
        for i, t in enumerate(camera_times)
    ]

    camera_path = tmp_path / "camera.mp4"
    camera_path.write_bytes(b"\x00")
    mic_path = tmp_path / "mic.wav"
    mic_path.write_bytes(b"\x00")
    camera_words_path = tmp_path / "camera.words.json"
    camera_words_path.write_text(json.dumps(camera_words))
    mic_words_path = tmp_path / "mic.words.json"
    mic_words_path.write_text(json.dumps(mic_words))

    def fake_loader(media_path, high_fidelity):
        if media_path == str(camera_path):
            return camera_audio, sample_rate
        if media_path == str(mic_path):
            return mic_audio, sample_rate
        raise AssertionError(f"unexpected media_path: {media_path}")

    monkeypatch.setattr(sync_audio, "_load_audio_for_correlation", fake_loader)

    report = sync_audio.compute_sync_offsets(
        str(camera_path), str(mic_path),
        camera_words=str(camera_words_path), mic_words=str(mic_words_path),
        anchor_spacing_seconds=15.0,
    )

    assert len(report["anchors"]) == 2
    assert report["driftFit"] is None
    assert abs(report["recommendedOffsetSeconds"] - true_offset_seconds) < 1e-6
    assert any("median" in w for w in report["warnings"])


def test_compute_sync_offsets_calls_transcribe_file_when_words_omitted(tmp_path, monkeypatch):
    camera_path = tmp_path / "camera.mp4"
    camera_path.write_bytes(b"\x00")
    mic_path = tmp_path / "mic.wav"
    mic_path.write_bytes(b"\x00")

    camera_words_path = tmp_path / "camera.words.json"
    camera_words_path.write_text(json.dumps([{"text": "hello", "start": 0.0, "end": 0.3}]))
    mic_words_path = tmp_path / "mic.words.json"
    mic_words_path.write_text(json.dumps([{"text": "hello", "start": 0.1, "end": 0.4}]))

    calls = []

    def fake_transcribe_file(input_file, language="English"):
        calls.append(input_file)
        if input_file == str(camera_path):
            return (str(tmp_path / "camera.txt"), str(camera_words_path))
        return (str(tmp_path / "mic.txt"), str(mic_words_path))

    monkeypatch.setattr(sync_audio, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(
        sync_audio, "_load_audio_for_correlation",
        lambda media_path, high_fidelity: (np.zeros(1000), 1000),
    )

    with pytest.raises(RuntimeError):
        # Only 1 matched word -> zero anchors after _pick_anchors would
        # still try correlation on silent audio -> RuntimeError from the
        # zero-confident-anchors path. We only care that transcribe_file
        # was invoked for both files before that happens.
        sync_audio.compute_sync_offsets(str(camera_path), str(mic_path))

    assert calls == [str(camera_path), str(mic_path)]
