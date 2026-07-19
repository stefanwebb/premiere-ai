"""Locate raw camera and microphone recordings on their mounted capture
volumes, match a camera clip to a mic recording from the same take, and
copy both into a Premiere Pro project directory.

Where to look is entirely environment-driven — this package makes no
assumption about anyone's specific camera model or mic hardware. Any
variable left unset just means that source is skipped, so auto-detection
degrades gracefully to whichever sources are configured; a caller that
passes `--camera-file`/`--mic-file` explicitly doesn't need any of these
set at all.

- `PREMIERE_AI_CAMERA_GLOBS` — comma-separated glob pattern(s) for camera
  clips, e.g. "/Volumes/MyCamera/DCIM/**/*.MP4"
- `PREMIERE_AI_MIC_FLAT_ROOTS` — comma-separated directories to check
  directly for `*.wav`/`*.WAV` files (non-recursive)
- `PREMIERE_AI_MIC_RECURSIVE_ROOTS` — comma-separated directories to
  search recursively for `*.wav`/`*.WAV` files
"""

import glob
import os
import shutil
import subprocess

DEFAULT_TIME_TOLERANCE_SECONDS = 30 * 60
DEFAULT_DURATION_TOLERANCE_SECONDS = 5.0
DEFAULT_DURATION_TOLERANCE_PCT = 0.03


def _split_env_list(name: str) -> list:
    return [p.strip() for p in os.environ.get(name, "").split(",") if p.strip()]


def find_camera_files() -> list:
    """Return camera clip paths matching PREMIERE_AI_CAMERA_GLOBS, newest first."""
    files = []
    for pattern in _split_env_list("PREMIERE_AI_CAMERA_GLOBS"):
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files), key=os.path.getmtime, reverse=True)


def find_mic_files() -> list:
    """Return mic recording paths from configured mic sources, newest first."""
    files = []
    for root in _split_env_list("PREMIERE_AI_MIC_FLAT_ROOTS"):
        files.extend(glob.glob(os.path.join(root, "*.wav")))
        files.extend(glob.glob(os.path.join(root, "*.WAV")))
    for root in _split_env_list("PREMIERE_AI_MIC_RECURSIVE_ROOTS"):
        files.extend(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
        files.extend(glob.glob(os.path.join(root, "**", "*.WAV"), recursive=True))
    return sorted(set(files), key=os.path.getmtime, reverse=True)


def get_duration_seconds(path: str) -> float:
    """Probe a media file's duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def find_best_match(
    camera_files: list,
    mic_files: list,
    time_tolerance_seconds: float = DEFAULT_TIME_TOLERANCE_SECONDS,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    duration_tolerance_pct: float = DEFAULT_DURATION_TOLERANCE_PCT,
) -> tuple:
    """Find the best matching (camera_path, mic_path) pair, anchored on the
    most recent camera clip that has any match at all.

    Camera clips are tried newest first. For each, every mic file is
    checked for a match — mtimes within `time_tolerance_seconds`, or
    (failing that) media durations within `duration_tolerance_seconds` or
    `duration_tolerance_pct` — and the closest matching mic is paired with
    it. The first camera clip (i.e. the newest) with any match wins, so a
    tight but stale match never displaces the latest take; this only
    reaches further back if newer clips have no matching mic recording at
    all. Raises RuntimeError if no camera/mic pair matches by either
    criterion.
    """
    if not camera_files:
        configured = os.environ.get("PREMIERE_AI_CAMERA_GLOBS")
        raise RuntimeError(
            f"No camera recordings found (PREMIERE_AI_CAMERA_GLOBS={configured!r})"
            if configured else
            "No camera recordings found — PREMIERE_AI_CAMERA_GLOBS is not set, and "
            "no --camera-file was given"
        )
    if not mic_files:
        flat = os.environ.get("PREMIERE_AI_MIC_FLAT_ROOTS")
        recursive = os.environ.get("PREMIERE_AI_MIC_RECURSIVE_ROOTS")
        raise RuntimeError(
            f"No microphone recordings found (PREMIERE_AI_MIC_FLAT_ROOTS={flat!r}, "
            f"PREMIERE_AI_MIC_RECURSIVE_ROOTS={recursive!r})"
            if flat or recursive else
            "No microphone recordings found — neither PREMIERE_AI_MIC_FLAT_ROOTS nor "
            "PREMIERE_AI_MIC_RECURSIVE_ROOTS is set, and no --mic-file was given"
        )

    cameras_newest_first = sorted(camera_files, key=os.path.getmtime, reverse=True)
    mics_newest_first = sorted(mic_files, key=os.path.getmtime, reverse=True)

    for cam in cameras_newest_first:
        cam_mtime = os.path.getmtime(cam)
        time_matches = [
            (abs(cam_mtime - os.path.getmtime(mic)), mic)
            for mic in mics_newest_first
        ]
        time_matches = [(delta, mic) for delta, mic in time_matches if delta <= time_tolerance_seconds]
        if time_matches:
            time_matches.sort(key=lambda t: t[0])
            return cam, time_matches[0][1]

    for cam in cameras_newest_first:
        try:
            cam_duration = get_duration_seconds(cam)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            continue
        duration_matches = []
        for mic in mics_newest_first:
            try:
                mic_duration = get_duration_seconds(mic)
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                continue
            duration_delta = abs(cam_duration - mic_duration)
            tolerance = max(duration_tolerance_seconds, duration_tolerance_pct * max(cam_duration, mic_duration))
            if duration_delta <= tolerance:
                duration_matches.append((duration_delta, mic))
        if duration_matches:
            duration_matches.sort(key=lambda t: t[0])
            return cam, duration_matches[0][1]

    raise RuntimeError(
        "No camera/mic pair matched by recording time or duration.\n"
        f"Camera candidates: {camera_files[:5]}\n"
        f"Mic candidates: {mic_files[:5]}"
    )


def copy_footage(
    camera_path: str,
    mic_path: str,
    project_dir: str,
    video_name: str | None = None,
    audio_name: str | None = None,
) -> tuple:
    """Copy the matched camera and mic files into <project_dir>/assets/video
    and <project_dir>/assets/audio. Uses `video_name`/`audio_name` for the
    destination filenames if given, otherwise preserves the original
    filenames. Returns the (video_dest, audio_dest) destination paths."""
    video_dir = os.path.join(project_dir, "assets", "video")
    audio_dir = os.path.join(project_dir, "assets", "audio")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    video_dest = os.path.join(video_dir, video_name or os.path.basename(camera_path))
    audio_dest = os.path.join(audio_dir, audio_name or os.path.basename(mic_path))
    shutil.copy2(camera_path, video_dest)
    shutil.copy2(mic_path, audio_dest)
    return video_dest, audio_dest


def import_raw_footage(
    project_dir: str,
    camera_file: str | None = None,
    mic_file: str | None = None,
    time_tolerance_seconds: float = DEFAULT_TIME_TOLERANCE_SECONDS,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    duration_tolerance_pct: float = DEFAULT_DURATION_TOLERANCE_PCT,
    video_name: str | None = None,
    audio_name: str | None = None,
) -> tuple:
    """Match the latest (or explicitly given) camera/mic recordings and
    copy them into `project_dir`. Returns (video_dest, audio_dest)."""
    if camera_file and mic_file:
        cam, mic = camera_file, mic_file
    else:
        camera_files = [camera_file] if camera_file else find_camera_files()
        mic_files = [mic_file] if mic_file else find_mic_files()
        cam, mic = find_best_match(
            camera_files, mic_files,
            time_tolerance_seconds, duration_tolerance_seconds, duration_tolerance_pct,
        )

    if not os.path.isdir(project_dir):
        raise RuntimeError(f"Project directory not found: {project_dir}")

    return copy_footage(cam, mic, project_dir, video_name=video_name, audio_name=audio_name)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Locate the latest matching raw camera and mic recordings and copy them into a Premiere Pro project directory."
    )
    parser.add_argument("project_dir", help="Destination Premiere Pro project directory.")
    parser.add_argument("--camera-file", default=None, help="Use this camera file instead of auto-detecting.")
    parser.add_argument("--mic-file", default=None, help="Use this mic file instead of auto-detecting.")
    parser.add_argument(
        "--time-tolerance-minutes", type=float, default=DEFAULT_TIME_TOLERANCE_SECONDS / 60,
        help="Max minutes apart camera/mic file mtimes can be to count as a match (default: 30).",
    )
    parser.add_argument(
        "--duration-tolerance-seconds", type=float, default=DEFAULT_DURATION_TOLERANCE_SECONDS,
        help="Max seconds apart camera/mic durations can be to count as a match (default: 5).",
    )
    parser.add_argument(
        "--duration-tolerance-pct", type=float, default=DEFAULT_DURATION_TOLERANCE_PCT,
        help="Max fractional difference between camera/mic durations to count as a match (default: 0.03).",
    )
    parser.add_argument(
        "--video-name", default=None,
        help="Destination filename for the camera footage (default: keep the original filename).",
    )
    parser.add_argument(
        "--audio-name", default=None,
        help="Destination filename for the mic recording (default: keep the original filename).",
    )
    args = parser.parse_args()

    video_dest, audio_dest = import_raw_footage(
        args.project_dir,
        camera_file=args.camera_file,
        mic_file=args.mic_file,
        time_tolerance_seconds=args.time_tolerance_minutes * 60,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
        duration_tolerance_pct=args.duration_tolerance_pct,
        video_name=args.video_name,
        audio_name=args.audio_name,
    )
    print(f"Copied camera footage to: {video_dest}")
    print(f"Copied mic recording to: {audio_dest}")


if __name__ == "__main__":
    main()
