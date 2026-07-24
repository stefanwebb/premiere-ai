"""Apply a 3D .cube LUT to an image via trilinear interpolation, for
visually evaluating a fitted LUT against the frame it came from (or any
other frame).

Reading and interpolation are delegated to colour-science: `read_LUT`
parses the .cube (and owns the red-fastest axis ordering that the writers
in `lut_io.py` also rely on), and `LUT3D.apply` does the trilinear
interpolation over the 0-1 domain.
"""

import sys
from pathlib import Path

import colour
import numpy as np
from PIL import Image


def apply_lut(image: np.ndarray, lut: colour.LUT3D) -> np.ndarray:
    """Apply `lut` to a uint8 HxWx3 image, returning a uint8 HxWx3 image.

    Output is clipped to 0-255: these LUTs correct a display-referred
    Rec.709 frame, so anything the interpolation pushes out of range is an
    artifact rather than headroom worth keeping.
    """
    corrected = lut.apply(image.astype(float) / 255)
    return np.clip(corrected * 255, 0, 255).astype(np.uint8)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lut = colour.read_LUT(str(args.lut))
    with Image.open(args.image) as img:
        image = np.array(img.convert("RGB"))

    corrected = apply_lut(image, lut)
    Image.fromarray(corrected).save(args.output)
    print(f"Saved corrected image to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
