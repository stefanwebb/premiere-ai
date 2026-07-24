"""Render a Premiere-Lumetri-style YUV vectorscope from an image file.

    vectorscope frame.png -o scope.png
    vectorscope a.png b.png -o compare.png --labels before after

Draws the same graticule Lumetri does: a circular bounding trace, the six
primary/secondary targets (R, Mg, B, Cy, G, Yl) each with a 75%-bar box
and a 100% box, the skin-tone line, and a centre crosshair. The trace is a
log-scaled 2D histogram of the frame's Cb/Cr, so dense regions read bright
the way a real scope's persistence does.

Images are loaded through image_io.load_srgb so an ICC-tagged file (e.g. a
Display P3 screenshot) is measured in the same space Premiere would show
it in, not in its raw encoded values.

--colorchecker-patches {chromatic,skin} restricts the trace to just the
named category of patches on a Calibrite ColorChecker Passport Video 2's
video page (chromatic = the six green/cyan/blue/magenta/red/yellow chips,
skin = the six skin-tone chips — see patch_mask.py), detected with the
same SAM3 pipeline calibrate-lut uses. --debug-mask-output saves the
image with everything outside the detected patches dimmed, so a bad
detection (wrong chart location, wrong column) is visible before trusting
the resulting trace.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Circle, Rectangle

from premiere_ai.colorchecker.image_io import load_srgb
from premiere_ai.colorchecker.patch_mask import PATCH_CATEGORY_COLUMNS, detect_patch_mask, load_predictor
from premiere_ai.colorchecker.vectorscope import _PRIMARY_RGB, primary_target_point, rgb_to_ycbcr, skin_tone_target_hue

# Lumetri labels its targets with these abbreviations, at these colours.
LABELS = {"red": "R", "magenta": "Mg", "blue": "B", "cyan": "Cy", "green": "G", "yellow": "Yl"}
COLORS = {
    "red": "#d94141", "magenta": "#c341c3", "blue": "#4147d9",
    "cyan": "#41b0b0", "green": "#41a541", "yellow": "#b0b041",
}
BARS_AMPLITUDE = 0.75  # standard colour-bars amplitude the inner box marks


def chroma_histogram(
    image: np.ndarray, bins: int, extent: float, max_pixels: int, mask: np.ndarray | None = None
) -> np.ndarray:
    """2D Cb/Cr histogram of an image's pixels, subsampled for speed.

    `mask`, if given, restricts the histogram to just the True pixels
    (e.g. a detected ColorChecker patch category) instead of the whole
    frame."""
    px = image[mask] if mask is not None else image.reshape(-1, 3)
    px = px.astype(np.float32) / 255.0
    if len(px) > max_pixels:
        idx = np.random.default_rng(0).choice(len(px), max_pixels, replace=False)
        px = px[idx]
    ycbcr = rgb_to_ycbcr(px)
    cb, cr = ycbcr[:, 1], ycbcr[:, 2]
    hist, _, _ = np.histogram2d(
        cb, cr, bins=bins, range=[[-extent, extent], [-extent, extent]]
    )
    return hist.T  # histogram2d puts the first arg on axis 0; we want cb on x


def draw_graticule(ax, extent: float, show_skin_line: bool) -> None:
    ax.add_patch(Circle((0, 0), extent * 0.92, fill=False, ec="#8a8a8a", lw=1.0))
    tick = extent * 0.022
    ax.plot([-tick, tick], [0, 0], color="#c8c8c8", lw=1.0)
    ax.plot([0, 0], [-tick, tick], color="#c8c8c8", lw=1.0)

    box = extent * 0.038
    for name in _PRIMARY_RGB:
        colour = COLORS[name]
        for amplitude, lw in ((BARS_AMPLITUDE, 1.2), (1.0, 1.2)):
            cb, cr = primary_target_point(name, amplitude)
            ax.add_patch(
                Rectangle((cb - box, cr - box), 2 * box, 2 * box,
                          fill=False, ec=colour, lw=lw)
            )
        # label just outside the 100% box, pushed radially outward
        cb, cr = primary_target_point(name, 1.0)
        norm = np.hypot(cb, cr) or 1.0
        ax.text(cb + cb / norm * box * 2.2, cr + cr / norm * box * 2.2, LABELS[name],
                color="#d0d0d0", fontsize=11, ha="center", va="center")

    if show_skin_line:
        theta = np.radians(skin_tone_target_hue())
        r = extent * 0.92
        ax.plot([0, r * np.cos(theta)], [0, r * np.sin(theta)],
                color="#9a9a9a", lw=0.9, alpha=0.9)


def save_debug_mask(image: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    """Save `image` with everything outside `mask` dimmed, so a detected
    ColorChecker patch region can be visually confirmed before trusting
    the vectorscope trace built from it."""
    debug = image.copy()
    debug[~mask] = (debug[~mask].astype(np.float32) * 0.25).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(debug).save(out_path)
    print(f"wrote mask debug image to {out_path}")


def debug_mask_path_for(base: Path, index: int, total: int) -> Path:
    """base with an `_N` suffix inserted before the extension when
    rendering more than one image, so each input's debug mask gets its
    own file instead of overwriting the last one."""
    if total <= 1:
        return base
    return base.with_name(f"{base.stem}_{index}{base.suffix}")


def render(paths, out_path, labels=None, bins=600, gain=3.0, max_pixels=2_000_000,
           show_skin_line=True, colorchecker_patches=None, debug_mask_output=None):
    # Scale so the 100% primaries sit just inside the bounding circle, as
    # Lumetri's graticule does.
    extent = max(np.hypot(*primary_target_point(n, 1.0)) for n in _PRIMARY_RGB) * 1.18

    predictor = load_predictor() if colorchecker_patches else None

    fig, axes = plt.subplots(1, len(paths), figsize=(6.2 * len(paths), 6.4),
                             facecolor="black", squeeze=False)
    for i, path in enumerate(paths):
        ax = axes[0][i]
        image = np.array(load_srgb(path))

        mask = None
        if colorchecker_patches:
            mask = detect_patch_mask(Image.fromarray(image), colorchecker_patches, predictor=predictor)
            if debug_mask_output:
                save_debug_mask(image, mask, debug_mask_path_for(debug_mask_output, i, len(paths)))

        hist = chroma_histogram(image, bins, extent, max_pixels, mask=mask)
        shown = np.log1p(hist)
        if shown.max() > 0:
            shown = np.clip(shown / shown.max() * gain, 0, 1)
        ax.imshow(shown, origin="lower", extent=[-extent, extent, -extent, extent],
                  cmap="bone", interpolation="bilinear", zorder=0)
        draw_graticule(ax, extent, show_skin_line)
        ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor("black")
        title = labels[i] if labels and i < len(labels) else Path(path).name
        ax.set_title(title, color="#d0d0d0", fontsize=12, pad=10)

    fig.patch.set_facecolor("black")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="black", dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--bins", type=int, default=600)
    p.add_argument("--gain", type=float, default=3.0,
                   help="brightness of the trace (higher shows sparser pixels)")
    p.add_argument("--no-skin-line", action="store_true")
    p.add_argument(
        "--colorchecker-patches", choices=sorted(PATCH_CATEGORY_COLUMNS),
        default=None,
        help="Detect a ColorChecker Passport Video 2's video page in each image and "
             "restrict the trace to just this category of its 24-patch grid.",
    )
    p.add_argument(
        "--debug-mask-output", type=Path, default=None,
        help="With --colorchecker-patches, also save the image with everything "
             "outside the detected patches dimmed, to confirm the detection landed "
             "in the right place. When rendering more than one image, an index is "
             "inserted before the extension of each one's debug file.",
    )
    args = p.parse_args()
    if args.debug_mask_output and not args.colorchecker_patches:
        p.error("--debug-mask-output requires --colorchecker-patches")
    render(args.images, args.output, labels=args.labels, bins=args.bins,
           gain=args.gain, show_skin_line=not args.no_skin_line,
           colorchecker_patches=args.colorchecker_patches,
           debug_mask_output=args.debug_mask_output)


if __name__ == "__main__":
    sys.exit(main())
