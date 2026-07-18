import pytest

from premiere_ai import remove_pauses


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeAnthropicMessages:
    def __init__(self, handler):
        self._handler = handler

    def create(self, **kwargs):
        return self._handler(**kwargs)


class _FakeAnthropicClient:
    def __init__(self, handler):
        self.messages = _FakeAnthropicMessages(handler)


def test_detect_boundaries_parses_anthropic_response(monkeypatch):
    def handler(**kwargs):
        assert kwargs["model"] == remove_pauses.ANTHROPIC_MODEL
        assert kwargs["messages"][0]["role"] == "user"
        return _FakeAnthropicResponse("[0, 2]")

    monkeypatch.setattr(
        remove_pauses, "_get_anthropic_client", lambda: _FakeAnthropicClient(handler)
    )
    words = [{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}]
    assert remove_pauses._detect_boundaries(words) == [0, 2]


def test_detect_boundaries_raises_on_api_error(monkeypatch):
    def handler(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        remove_pauses, "_get_anthropic_client", lambda: _FakeAnthropicClient(handler)
    )
    words = [{"text": "a"}, {"text": "b"}]
    with pytest.raises(RuntimeError, match="Anthropic API call failed"):
        remove_pauses._detect_boundaries(words)


def test_get_anthropic_client_raises_when_api_key_missing(monkeypatch):
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not set"):
        remove_pauses._get_anthropic_client()


def test_detect_silence_raises_clean_error_when_vad_load_fails(monkeypatch):
    import torch

    def fake_hub_load(*args, **kwargs):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    with pytest.raises(RuntimeError, match="failed to load Silero VAD model"):
        remove_pauses._detect_silence("/dev/null")


def test_detect_silence_uses_tuned_vad_parameters_and_returns_duration(monkeypatch):
    import torch

    calls = {}

    def fake_get_speech_timestamps(audio, model, sampling_rate, return_seconds, **kwargs):
        calls["kwargs"] = kwargs
        return []  # no speech detected -> entire duration counts as silence

    def fake_read_audio(path, sampling_rate):
        return [0.0] * (sampling_rate * 2)  # 2 seconds of audio

    fake_utils = (fake_get_speech_timestamps, None, fake_read_audio)

    def fake_hub_load(*args, **kwargs):
        return "fake-model", fake_utils

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)

    silence, total_duration = remove_pauses._detect_silence("/dev/null")

    assert total_duration == 2.0
    assert silence == [(0.0, 2.0)]
    # speech_pad_ms=0 and a low min_silence_duration_ms avoid Silero's
    # defaults (30ms padding + 100ms minimum) silently eating into real
    # pauses shorter than that — our own margin/min-pause filtering
    # downstream already gates on duration. time_resolution=3 (1ms) avoids
    # Silero's default 0.1s-rounded timestamps, which can round two
    # adjacent silence boundaries to the same value and erase short
    # silences entirely.
    assert calls["kwargs"] == {
        "speech_pad_ms": 0, "min_silence_duration_ms": 30, "time_resolution": 3,
    }


def test_run_wires_helpers_and_formats_output(monkeypatch, tmp_path):
    input_file = tmp_path / "clip.wav"
    input_file.write_bytes(b"\x00")

    words = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "world", "start": 2.0, "end": 3.0},
    ]

    monkeypatch.setattr(
        remove_pauses, "_ensure_words_json",
        lambda input_file, language, verbose: str(tmp_path / "clip.words.json"),
    )
    monkeypatch.setattr(remove_pauses, "_load_words", lambda words_path: words)
    monkeypatch.setattr(
        remove_pauses, "_get_wav_path", lambda input_file: (str(input_file), None)
    )
    monkeypatch.setattr(remove_pauses, "_detect_boundaries", lambda words: [0])
    monkeypatch.setattr(
        remove_pauses, "_detect_silence", lambda wav_path: ([(1.0, 2.0)], 3.0)
    )

    lines, total_seconds = remove_pauses.run(
        str(input_file), aggressiveness=0.5, min_pause_ms=300, fps=25
    )

    assert lines == ["00:01:01 - 00:01:22"]
    # cut spans frames 26-47 (21 frames) at 25fps = 0.84s
    assert total_seconds == pytest.approx(21 / 25)


def test_run_with_restrict_to_boundaries_false_skips_claude_and_uses_all_gaps(
    monkeypatch, tmp_path
):
    input_file = tmp_path / "clip.wav"
    input_file.write_bytes(b"\x00")

    # Three words -> two gaps (indices 0 and 1); only the gap after word 1
    # (a genuine mid-phrase pause, no punctuation) has silence overlapping it.
    words = [
        {"text": "hello", "start": 0.0, "end": 1.0},
        {"text": "there", "start": 1.2, "end": 2.0},
        {"text": "world", "start": 3.0, "end": 4.0},
    ]

    monkeypatch.setattr(
        remove_pauses, "_ensure_words_json",
        lambda input_file, language, verbose: str(tmp_path / "clip.words.json"),
    )
    monkeypatch.setattr(remove_pauses, "_load_words", lambda words_path: words)
    monkeypatch.setattr(
        remove_pauses, "_get_wav_path", lambda input_file: (str(input_file), None)
    )

    def _fail_if_called(words):
        raise AssertionError("_detect_boundaries should not be called")

    monkeypatch.setattr(remove_pauses, "_detect_boundaries", _fail_if_called)
    monkeypatch.setattr(
        remove_pauses, "_detect_silence", lambda wav_path: ([(2.0, 3.0)], 4.0)
    )

    lines, total_seconds = remove_pauses.run(
        str(input_file), aggressiveness=1.0, min_pause_ms=1,
        restrict_to_boundaries=False,
    )

    assert len(lines) == 1
    assert total_seconds > 0.0


def test_run_returns_empty_list_when_fewer_than_two_words(monkeypatch, tmp_path):
    input_file = tmp_path / "clip.wav"
    input_file.write_bytes(b"\x00")

    monkeypatch.setattr(
        remove_pauses, "_ensure_words_json",
        lambda input_file, language, verbose: str(tmp_path / "clip.words.json"),
    )
    monkeypatch.setattr(remove_pauses, "_load_words", lambda words_path: [{"text": "hi", "start": 0.0, "end": 0.5}])

    lines, total_seconds = remove_pauses.run(str(input_file))
    assert lines == []
    assert total_seconds == 0.0


import sys


def test_main_missing_file_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["remove-pauses", "/no/such/file.wav"])
    with pytest.raises(SystemExit) as exc_info:
        remove_pauses.main()
    assert exc_info.value.code != 0
    assert "not found" in capsys.readouterr().err


def test_main_rejects_out_of_range_aggressiveness(monkeypatch, tmp_path, capsys):
    input_file = tmp_path / "clip.wav"
    input_file.write_bytes(b"\x00")
    monkeypatch.setattr(
        sys, "argv", ["remove-pauses", str(input_file), "--aggressiveness", "1.5"]
    )
    with pytest.raises(SystemExit) as exc_info:
        remove_pauses.main()
    assert exc_info.value.code != 0
    assert "aggressiveness" in capsys.readouterr().err


def test_main_writes_cut_list_and_prints_output(monkeypatch, tmp_path, capsys):
    input_file = tmp_path / "clip.wav"
    input_file.write_bytes(b"\x00")

    monkeypatch.setattr(sys, "argv", ["remove-pauses", str(input_file)])
    monkeypatch.setattr(
        remove_pauses, "run", lambda *args, **kwargs: (["00:01:02 - 00:01:22"], 20.4)
    )

    remove_pauses.main()

    captured = capsys.readouterr()
    assert "00:01:02 - 00:01:22" in captured.out
    assert "Total pause time removed: 0:20.4" in captured.out

    output_path = tmp_path / "clip.cuts.txt"
    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == "00:01:02 - 00:01:22\n"


def test_main_no_cuts_writes_empty_file(monkeypatch, tmp_path, capsys):
    input_file = tmp_path / "clip.wav"
    input_file.write_bytes(b"\x00")

    monkeypatch.setattr(sys, "argv", ["remove-pauses", str(input_file)])
    monkeypatch.setattr(remove_pauses, "run", lambda *args, **kwargs: ([], 0.0))

    remove_pauses.main()

    captured = capsys.readouterr()
    assert "No pauses found" in captured.out
    assert "Total pause time" not in captured.out

    output_path = tmp_path / "clip.cuts.txt"
    assert output_path.read_text(encoding="utf-8") == ""
