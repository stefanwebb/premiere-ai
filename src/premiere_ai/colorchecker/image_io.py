"""Color-managed image loading, matched to Premiere's working space.

CRITICAL for any measurement feeding a LUT fit: PIL returns an image's RAW
encoded values and ignores its embedded ICC profile, but Premiere Pro
color-manages on import — so a file tagged Display P3 (which is what macOS
screenshots are by default) shows Premiere *different* RGB numbers than
PIL reports for the same pixel.

Measuring in P3 and then handing Premiere a LUT keyed on those numbers
silently mis-corrects: a hue-indexed curve gets applied to hues that no
longer match the ones it was fit against.

Target space is a Rec.709 sequence. sRGB and Rec.709 share primaries and
white point (they differ only in transfer function), so sRGB stands in as
the conversion target — the primaries change is the part that moves hue
and saturation, which is what these fits key on.

Files ALREADY carrying Rec.709/sRGB primaries are passed through
untouched. Converting those would apply a transfer-function shift that
Premiere itself never performs (e.g. `colorchecker.png` is tagged
Rec. ITU-R BT.709-5 and lands in a Rec.709 sequence needing no conversion
at all), so a blanket "convert everything to sRGB" would introduce error
rather than remove it.
"""

import io

import numpy as np
from PIL import Image, ImageCms

# sRGB / Rec.709 primaries as ICC colorant tags. These are the D50-ADAPTED
# values, not the familiar D65 ones (0.4124, 0.2126, 0.0193 for red): an
# ICC profile's connection space is D50, so every profile's colorants come
# back Bradford-adapted to it. Comparing against D65 numbers here rejects
# even a genuine sRGB profile.
_REC709_COLORANTS_D50 = np.array(
    [
        [0.4360, 0.2225, 0.0139],  # red
        [0.3851, 0.7169, 0.0971],  # green
        [0.1430, 0.0606, 0.7139],  # blue
    ]
)
_COLORANT_TOLERANCE = 0.02


def _has_rec709_primaries(profile: ImageCms.ImageCmsProfile) -> bool:
    try:
        colorants = np.array(
            [
                list(profile.profile.red_colorant[0]),
                list(profile.profile.green_colorant[0]),
                list(profile.profile.blue_colorant[0]),
            ]
        )
    except (AttributeError, TypeError):
        return False
    return bool(np.all(np.abs(colorants - _REC709_COLORANTS_D50) < _COLORANT_TOLERANCE))


def load_srgb(path) -> Image.Image:
    """Open an image and bring it into the Rec.709/sRGB working space.

    Untagged files are assumed to already be there and pass through
    unconverted — that is the assumption every consumer of an untagged
    file makes, Premiere included.
    """
    img = Image.open(path)
    icc = img.info.get("icc_profile")
    img = img.convert("RGB")

    if not icc:
        return img

    source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    description = ImageCms.getProfileDescription(source).strip()

    if _has_rec709_primaries(source):
        print(f"  {description!r} already has Rec.709 primaries — no conversion")
        return img

    target = ImageCms.createProfile("sRGB")
    converted = ImageCms.profileToProfile(img, source, target, outputMode="RGB")
    print(f"  color-managed {description!r} -> sRGB (Rec.709 primaries)")
    return converted


def load_srgb_array(path) -> np.ndarray:
    return np.array(load_srgb(path))
