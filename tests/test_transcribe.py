import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from premiere_ai import transcribe


class _FakeAlignedItem:
    def __init__(self, text, start_time, end_time):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


def _install_fake_load_audio(monkeypatch):
    """_align_segments does `from mlx_audio.stt.utils import load_audio`
    inside its body, so the fake module must be registered in sys.modules
    before the function runs."""
    fake_utils_mod = types.ModuleType("mlx_audio.stt.utils")
    fake_utils_mod.load_audio = lambda path: np.zeros(16000, dtype=np.float32)
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", types.ModuleType("mlx_audio.stt"))
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_utils_mod)


def test_align_segments_preserves_punctuation_when_token_counts_match(monkeypatch):
    _install_fake_load_audio(monkeypatch)

    class FakeAlignerModel:
        def generate(self, audio_slice, text, language):
            # The real aligner strips punctuation; simulate that here.
            return [
                _FakeAlignedItem("Hello", 0.0, 0.4),
                _FakeAlignedItem("world", 0.5, 1.0),
            ]

    segments = SimpleNamespace(
        segments=[{"text": "Hello, world!", "start": 0.0, "end": 1.0}]
    )

    words = transcribe._align_segments(FakeAlignerModel(), "fake.wav", segments, "English")

    assert words == [
        {"text": "Hello,", "start": 0.0, "end": 0.4},
        {"text": "world!", "start": 0.5, "end": 1.0},
    ]


def test_align_segments_falls_back_to_aligner_text_on_token_count_mismatch(monkeypatch):
    _install_fake_load_audio(monkeypatch)

    class FakeAlignerModel:
        def generate(self, audio_slice, text, language):
            # Only one aligned item, but the raw text has two whitespace
            # tokens — counts don't match, so punctuation reattachment must
            # not be attempted.
            return [_FakeAlignedItem("Hello", 0.0, 0.4)]

    segments = SimpleNamespace(
        segments=[{"text": "Hello, world!", "start": 0.0, "end": 1.0}]
    )

    words = transcribe._align_segments(FakeAlignerModel(), "fake.wav", segments, "English")

    assert words == [{"text": "Hello", "start": 0.0, "end": 0.4}]


def test_transcribe_file_missing_input_raises():
    with pytest.raises(FileNotFoundError):
        transcribe.transcribe_file("/no/such/file.wav")


def test_transcribe_file_missing_dependencies_raises(tmp_path, monkeypatch):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"\x00")
    monkeypatch.setattr(transcribe, "_check_dependencies", lambda: False)
    with pytest.raises(RuntimeError):
        transcribe.transcribe_file(str(audio_path))


def test_transcribe_file_happy_path(tmp_path, monkeypatch):
    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"\x00")

    monkeypatch.setattr(transcribe, "_check_dependencies", lambda: True)

    class FakeSegments:
        segments = [{"text": "hello world", "start": 0.0, "end": 1.0}]

    def fake_load_model(model_id):
        return f"model:{model_id}"

    def fake_generate_transcription(model, audio, output_path, format, verbose):
        with open(f"{output_path}.{format}", "w", encoding="utf-8") as fh:
            fh.write("hello world")
        return FakeSegments()

    def fake_align_segments(aligner_model, audio_path, segments, language):
        return [
            {"text": "hello", "start": 0.0, "end": 0.4},
            {"text": "world", "start": 0.5, "end": 1.0},
        ]

    fake_mlx_audio_mod = types.ModuleType("mlx_audio")
    fake_stt_mod = types.ModuleType("mlx_audio.stt")
    fake_generate_mod = types.ModuleType("mlx_audio.stt.generate")
    fake_generate_mod.generate_transcription = fake_generate_transcription
    fake_utils_mod = types.ModuleType("mlx_audio.stt.utils")
    fake_utils_mod.load_model = fake_load_model

    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio_mod)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt", fake_stt_mod)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.generate", fake_generate_mod)
    monkeypatch.setitem(sys.modules, "mlx_audio.stt.utils", fake_utils_mod)
    monkeypatch.setattr(transcribe, "_align_segments", fake_align_segments)

    transcript_path, words_path = transcribe.transcribe_file(str(audio_path))

    assert transcript_path == str(tmp_path / "clip.txt")
    assert words_path == str(tmp_path / "clip.words.json")

    with open(words_path, encoding="utf-8") as fh:
        words = json.load(fh)
    assert words == [
        {"text": "hello", "start": 0.0, "end": 0.4},
        {"text": "world", "start": 0.5, "end": 1.0},
    ]
    with open(transcript_path, encoding="utf-8") as fh:
        assert fh.read() == "hello world"
