import os
import time

import pytest

from premiere_ai import import_raw_footage as irf


def _touch(path: str, mtime: float, content: bytes = b"data") -> str:
    with open(path, "wb") as f:
        f.write(content)
    os.utime(path, (mtime, mtime))
    return path


def test_find_best_match_prefers_close_mtimes(tmp_path):
    now = time.time()
    cam = _touch(str(tmp_path / "C0001.MP4"), now)
    mic_close = _touch(str(tmp_path / "close.wav"), now + 60)
    mic_far = _touch(str(tmp_path / "far.wav"), now - 5 * 3600)

    result_cam, result_mic = irf.find_best_match([cam], [mic_far, mic_close])

    assert result_cam == cam
    assert result_mic == mic_close


def test_find_best_match_picks_smallest_time_delta_among_multiple_cameras(tmp_path):
    now = time.time()
    mic = _touch(str(tmp_path / "mic.wav"), now)
    cam_far = _touch(str(tmp_path / "far.MP4"), now - 20 * 60)
    cam_close = _touch(str(tmp_path / "close.MP4"), now - 2 * 60)

    result_cam, result_mic = irf.find_best_match([cam_far, cam_close], [mic])

    assert result_cam == cam_close
    assert result_mic == mic


def test_find_best_match_prefers_latest_camera_over_tighter_stale_match(tmp_path):
    now = time.time()
    # An old take where camera and mic mtimes happen to be very close together.
    old_cam = _touch(str(tmp_path / "old_cam.MP4"), now - 10 * 24 * 3600)
    old_mic = _touch(str(tmp_path / "old_mic.wav"), now - 10 * 24 * 3600 + 2)
    # The latest take, mtimes further apart but still well within tolerance.
    new_cam = _touch(str(tmp_path / "new_cam.MP4"), now)
    new_mic = _touch(str(tmp_path / "new_mic.wav"), now + 11)

    result_cam, result_mic = irf.find_best_match([old_cam, new_cam], [old_mic, new_mic])

    assert result_cam == new_cam
    assert result_mic == new_mic


def test_find_best_match_raises_when_no_pair_matches(tmp_path):
    now = time.time()
    cam = _touch(str(tmp_path / "C0001.MP4"), now)
    mic = _touch(str(tmp_path / "mic.wav"), now - 10 * 3600)

    with pytest.raises(RuntimeError, match="No camera/mic pair matched"):
        irf.find_best_match([cam], [mic], duration_tolerance_seconds=0, duration_tolerance_pct=0)


def test_find_best_match_raises_on_empty_camera_list():
    with pytest.raises(RuntimeError, match="No camera recordings found"):
        irf.find_best_match([], ["/tmp/mic.wav"])


def test_find_best_match_raises_on_empty_mic_list():
    with pytest.raises(RuntimeError, match="No microphone recordings found"):
        irf.find_best_match(["/tmp/cam.MP4"], [])


def test_copy_footage_creates_video_and_audio_subdirs(tmp_path):
    cam = _touch(str(tmp_path / "C0001.MP4"), time.time(), b"camera bytes")
    mic = _touch(str(tmp_path / "mic.wav"), time.time(), b"mic bytes")
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    video_dest, audio_dest = irf.copy_footage(cam, mic, str(project_dir))

    assert video_dest == os.path.join(str(project_dir), "assets", "video", "C0001.MP4")
    assert audio_dest == os.path.join(str(project_dir), "assets", "audio", "mic.wav")
    assert open(video_dest, "rb").read() == b"camera bytes"
    assert open(audio_dest, "rb").read() == b"mic bytes"


def test_copy_footage_uses_given_names_when_provided(tmp_path):
    cam = _touch(str(tmp_path / "C0001.MP4"), time.time(), b"camera bytes")
    mic = _touch(str(tmp_path / "mic.wav"), time.time(), b"mic bytes")
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    video_dest, audio_dest = irf.copy_footage(
        cam, mic, str(project_dir), video_name="main-camera.mp4", audio_name="main-microphone.wav",
    )

    assert video_dest == os.path.join(str(project_dir), "assets", "video", "main-camera.mp4")
    assert audio_dest == os.path.join(str(project_dir), "assets", "audio", "main-microphone.wav")
    assert open(video_dest, "rb").read() == b"camera bytes"
    assert open(audio_dest, "rb").read() == b"mic bytes"


def test_import_raw_footage_uses_explicit_files_without_matching(tmp_path):
    cam = _touch(str(tmp_path / "C0001.MP4"), time.time())
    mic = _touch(str(tmp_path / "mic.wav"), time.time() - 100 * 3600)
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    video_dest, audio_dest = irf.import_raw_footage(str(project_dir), camera_file=cam, mic_file=mic)

    assert os.path.isfile(video_dest)
    assert os.path.isfile(audio_dest)


def test_import_raw_footage_raises_on_missing_project_dir(tmp_path):
    cam = _touch(str(tmp_path / "C0001.MP4"), time.time())
    mic = _touch(str(tmp_path / "mic.wav"), time.time())

    with pytest.raises(RuntimeError, match="Project directory not found"):
        irf.import_raw_footage(str(tmp_path / "does-not-exist"), camera_file=cam, mic_file=mic)
