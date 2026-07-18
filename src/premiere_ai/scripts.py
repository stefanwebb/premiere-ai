import os
import sys
import subprocess
from importlib.resources import files


def _run_sh(script_name: str) -> None:
    script_path = files("premiere_ai").joinpath(script_name)
    # Write to a temp file if the resource is inside a zip/wheel; otherwise use directly.
    try:
        path = str(script_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    except (TypeError, FileNotFoundError):
        import tempfile
        data = script_path.read_bytes()
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as tmp:
            tmp.write(data)
            path = tmp.name
        os.chmod(path, 0o755)

    result = subprocess.run(["bash", path] + sys.argv[1:])
    sys.exit(result.returncode)


def zmbv_to_h265_vga() -> None:
    _run_sh("zmbv_to_h265_vga.sh")


def zmbv_to_h265_ega() -> None:
    _run_sh("zmbv_to_h265_ega.sh")
