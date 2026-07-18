"""Create a fresh, empty Premiere Pro project by copying the shared empty
template and renaming its .prproj file to match the new project.
"""

import os
import shutil
import sys

TEMPLATE_DIR = "/Volumes/Extreme Pro/video-production/Shared Assets/Empty Premiere Pro Template"
DEFAULT_BASE_DIR = "/Volumes/Extreme Pro/video-production/Generative GameDev"


def create_project(project_name: str, series_name: str | None = None, base_dir: str | None = None) -> str:
    """Copy the empty template into <base_dir>/[<series_name>/]<project_name>
    and rename its .prproj file to '<project_name>.prproj'. Returns the
    destination directory path."""
    if not os.path.isdir(TEMPLATE_DIR):
        raise RuntimeError(f"Template directory not found: {TEMPLATE_DIR}")

    base_dir = base_dir or DEFAULT_BASE_DIR
    dest_dir = os.path.join(base_dir, series_name, project_name) if series_name else os.path.join(base_dir, project_name)

    if os.path.exists(dest_dir):
        raise RuntimeError(f"Destination already exists, refusing to overwrite: {dest_dir}")

    shutil.copytree(TEMPLATE_DIR, dest_dir)

    # Copying across some filesystems (e.g. exFAT external drives) emits
    # AppleDouble sidecar files (e.g. "._Untitled.prproj") for xattrs/resource
    # forks. Strip them from the copy so they don't get mistaken for real files.
    for root, _dirs, files in os.walk(dest_dir):
        for f in files:
            if f.startswith("._"):
                os.remove(os.path.join(root, f))

    prproj_files = [f for f in os.listdir(dest_dir) if f.endswith(".prproj")]
    if len(prproj_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .prproj file in template, found {len(prproj_files)}: {prproj_files}"
        )

    old_path = os.path.join(dest_dir, prproj_files[0])
    new_path = os.path.join(dest_dir, f"{project_name}.prproj")
    os.rename(old_path, new_path)

    return dest_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a fresh, empty Premiere Pro project from the shared empty template."
    )
    parser.add_argument("project_name", help="Name of the new project, e.g. vlog0002.")
    parser.add_argument(
        "--series", "-s", default=None,
        help="Optional series name; the project is nested under <base-dir>/<series>/<project_name>.",
    )
    parser.add_argument(
        "--base-dir", default=None,
        help=f"Override the default base directory (default: {DEFAULT_BASE_DIR}).",
    )
    args = parser.parse_args()

    try:
        dest_dir = create_project(args.project_name, series_name=args.series, base_dir=args.base_dir)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Created empty Premiere Pro project at {dest_dir}")


if __name__ == "__main__":
    main()
