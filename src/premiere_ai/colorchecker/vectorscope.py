"""Rec.709 Y'CbCr vectorscope geometry — used to derive hue targets that
Premiere's Lumetri "Vectorscope YUV" actually plots, computed analytically
from the standard broadcast matrices rather than reusing degree figures
quoted (inconsistently, across different zero-reference conventions) by
various web sources.

Hue angle convention here: atan2(Cr, Cb) in degrees, wrapped to [0, 360).
This is self-consistent for everything computed in this module — it is
NOT guaranteed to match the literal on-screen rotation of Premiere's
graticule (which may place its zero/orientation differently), so absolute
angles printed by callers should be read as "this many degrees apart from
each other," and visually cross-checked against Premiere's scope before
being trusted as absolute.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

# Standard Rec.709 luma coefficients (Kr, Kb); Kg = 1 - Kr - Kb.
REC709_KR = 0.2126
REC709_KB = 0.0722


def rgb_to_ycbcr(rgb_unit: np.ndarray) -> np.ndarray:
    """rgb_unit: (..., 3) in [0, 1]. Returns (..., 3) Y, Cb, Cr with Cb/Cr
    in [-0.5, 0.5]."""
    r, g, b = rgb_unit[..., 0], rgb_unit[..., 1], rgb_unit[..., 2]
    kr, kb = REC709_KR, REC709_KB
    y = kr * r + (1 - kr - kb) * g + kb * b
    cb = (b - y) / (2 * (1 - kb))
    cr = (r - y) / (2 * (1 - kr))
    return np.stack([y, cb, cr], axis=-1)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    kr, kb = REC709_KR, REC709_KB
    kg = 1 - kr - kb
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.stack([r, g, b], axis=-1)


def hue_angle_deg(cb: float, cr: float) -> float:
    return float(np.degrees(np.arctan2(cr, cb)) % 360)


def circular_mean_deg(angles_deg: list[float]) -> float:
    radians = np.radians(angles_deg)
    mean = np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return float(mean % 360)


def angular_diff_deg(target: float, measured: float) -> float:
    """Shortest signed rotation from `measured` to `target`, in (-180, 180]."""
    return float(((target - measured) + 180) % 360 - 180)


# Vectorscope target hues for the six 100%-saturated primaries/secondaries,
# computed directly from the Rec.709 matrix (exact, not looked up).
_PRIMARY_RGB = {
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow": (1, 1, 0),
}
PRIMARY_TARGET_HUES = {
    name: hue_angle_deg(*rgb_to_ycbcr(np.array(rgb, dtype=float))[1:])
    for name, rgb in _PRIMARY_RGB.items()
}


def skin_tone_target_hue() -> float:
    """The classic NTSC "skin tone line" is documented as running along the
    -I axis of the YIQ color model, sitting between the Yellow and Red
    vectorscope targets (closer to Red) — see conversation for sources. I
    and Q are U/V (i.e. Cb/Cr-family) rotated by the well-established
    33-degree constant, so -I expressed in our Cb/Cr basis is a +/-33-degree
    offset from the Q axis; which sign is correct is disambiguated here by
    checking which one actually falls between our computed Yellow and Red
    hues, rather than trusting a recalled rotation sign.

    This is a derived construction, not a literal figure taken from Adobe
    documentation of Premiere's exact on-screen graticule — treat the
    result as a well-reasoned estimate and confirm visually against
    Premiere's own skin-tone-line toggle before trusting it as exact.
    """
    red = PRIMARY_TARGET_HUES["red"]
    yellow = PRIMARY_TARGET_HUES["yellow"]

    def between(angle, a, b):
        # shortest-arc "is angle within the arc from a to b" check
        span = angular_diff_deg(b, a)
        pos = angular_diff_deg(angle, a)
        return (0 <= pos <= span) if span >= 0 else (span <= pos <= 0)

    candidates = [(yellow + 33) % 360, (yellow - 33) % 360]
    for candidate in candidates:
        if between(candidate, yellow, red):
            return candidate
    # Fallback: neither landed exactly between them (arc direction/rounding
    # edge case) — pick whichever is angularly closer to Red, since both
    # sources agree it's "closer to Red."
    return min(candidates, key=lambda c: abs(angular_diff_deg(red, c)))


def primary_target_point(name: str, amplitude: float = 0.75) -> np.ndarray:
    """(Cb, Cr) of a primary/secondary at a given amplitude — Premiere's
    vectorscope graticule target boxes are conventionally calibrated for
    75%-amplitude color bars, not 100%-saturated primaries, so this is what
    "the square" actually corresponds to. Returns a (2,) array [Cb, Cr]."""
    rgb = np.array(_PRIMARY_RGB[name], dtype=float)
    return rgb_to_ycbcr(rgb * amplitude)[1:]


def fit_periodic_hue_curve(hues_deg: list[float], values: list[float]) -> PchipInterpolator:
    """A smooth curve over the 0-360 degree hue circle passing through each
    (hue, value) point, wrapping seamlessly at the boundary — this is the
    same shape of tool as Lumetri's "Hue vs Hue" / "Hue vs Sat" curves
    (each keyed by ORIGINAL hue, unlike a single global matrix/rotation,
    which forces one shared correction on every hue at once).

    Uses PCHIP (shape-preserving / non-overshooting), not a natural cubic
    spline — with only 6 anchors that can end up very unevenly spaced
    around the circle (a heavy color cast can bunch several measured hues
    together and leave one huge gap elsewhere), a plain cubic spline
    overshoots badly across the big gap (confirmed: >3x saturation swing
    for a real captured frame with a 162-degree gap between two anchors).
    PCHIP is built specifically to never overshoot beyond the local data
    range, which is exactly the safety property needed here.

    Periodicity is approximated (scipy has no periodic PCHIP) by fitting
    over three wrapped copies of the data (hue-360, hue, hue+360) and only
    ever evaluating in the middle copy's domain — this keeps the curve
    continuous and consistent across the wrap without needing an exact
    periodic boundary condition.
    """
    order = np.argsort(hues_deg)
    hues_sorted = np.array(hues_deg)[order]
    values_sorted = np.array(values)[order]
    hues_ext = np.concatenate([hues_sorted - 360, hues_sorted, hues_sorted + 360])
    values_ext = np.concatenate([values_sorted, values_sorted, values_sorted])
    return PchipInterpolator(hues_ext, values_ext)


def eval_periodic_hue_curve(curve: PchipInterpolator, hues_deg: np.ndarray) -> np.ndarray:
    return curve(np.asarray(hues_deg) % 360)


def rotate_hue(rgb_unit: np.ndarray, delta_deg: float) -> np.ndarray:
    """Rotate Cb/Cr by delta_deg, leaving Y (and Cb/Cr magnitude) unchanged."""
    ycbcr = rgb_to_ycbcr(rgb_unit)
    y, cb, cr = ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]
    theta = np.radians(delta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    new_cb = cb * cos_t - cr * sin_t
    new_cr = cb * sin_t + cr * cos_t
    return ycbcr_to_rgb(np.stack([y, new_cb, new_cr], axis=-1))
