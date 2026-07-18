#!/usr/bin/env python3
"""
Calculate CSS-relevant font metrics for a given font file and size.

Usage:
    python3 font_metrics.py <font_file> <size_px>

Example:
    python3 font_metrics.py "public/CoFo Sans Pixel Regular.otf" 36
"""

import sys
from fontTools.ttLib import TTFont


def font_metrics(font_path: str, size_px: float) -> None:
    font = TTFont(font_path)

    units_per_em = font["head"].unitsPerEm
    scale = size_px / units_per_em

    # hhea table — what browsers (Chrome/WebKit) use for CSS line box calculations
    hhea = font["hhea"]
    hhea_ascender_px = hhea.ascent * scale
    hhea_descender_px = abs(hhea.descent) * scale
    hhea_line_gap_px = hhea.lineGap * scale
    content_area_px = (hhea.ascent + abs(hhea.descent)) * scale

    # OS/2 table — typographic (design-intent) metrics
    os2 = font["OS/2"]
    typo_ascender_px = os2.sTypoAscender * scale
    typo_descender_px = abs(os2.sTypoDescender) * scale
    typo_line_gap_px = os2.sTypoLineGap * scale

    print(f"Font:          {font_path}")
    print(f"Size:          {size_px}px")
    print(f"Units per em:  {units_per_em}")
    print()
    print("── hhea (used by browsers for CSS line box) ──────────────────")
    print(f"  Ascender:      {hhea_ascender_px:.2f}px  ({hhea.ascent} units)")
    print(f"  Descender:     {hhea_descender_px:.2f}px  ({abs(hhea.descent)} units)")
    print(f"  Line gap:      {hhea_line_gap_px:.2f}px  ({hhea.lineGap} units)")
    print(f"  Content area:  {content_area_px:.2f}px  (ascender + |descender|)")
    print()
    print("── OS/2 typographic ──────────────────────────────────────────")
    print(f"  Ascender:      {typo_ascender_px:.2f}px  ({os2.sTypoAscender} units)")
    print(f"  Descender:     {typo_descender_px:.2f}px  ({abs(os2.sTypoDescender)} units)")
    print(f"  Line gap:      {typo_line_gap_px:.2f}px  ({os2.sTypoLineGap} units)")
    print()
    print("── CSS baseline alignment ────────────────────────────────────")
    line_height = size_px  # baseline case: line-height == font-size
    half_leading = (line_height - content_area_px) / 2
    baseline_from_line_box_top = half_leading + hhea_ascender_px
    print(f"  With line-height = font-size ({size_px}px):")
    print(f"    half-leading:              {half_leading:.2f}px")
    print(f"    baseline from line-box top: {baseline_from_line_box_top:.2f}px")
    print()

    # Show for a custom line-height if it differs from size_px
    target_line_height = 45.0
    if target_line_height != size_px:
        half_leading_45 = (target_line_height - content_area_px) / 2
        baseline_45 = half_leading_45 + hhea_ascender_px
        print(f"  With line-height = {target_line_height}px:")
        print(f"    half-leading:              {half_leading_45:.2f}px")
        print(f"    baseline from line-box top: {baseline_45:.2f}px")
        print()
        print("  To align first baseline to a 45px grid with element top on a gridline:")
        import math
        # We want element_top + paddingTop + baseline_45 = N * 45
        # paddingTop = N*45 - element_top - baseline_45
        # Assuming element_top = 90:
        element_top = 90
        raw_baseline = element_top + baseline_45
        target = math.ceil(raw_baseline / 45) * 45
        padding_top = target - element_top - baseline_45
        print(f"    element top = {element_top}px, raw first baseline ≈ {raw_baseline:.2f}px")
        print(f"    nearest gridline above = {target:.0f}px")
        print(f"    required paddingTop ≈ {padding_top:.2f}px")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    font_path = sys.argv[1]
    size_px = float(sys.argv[2])
    font_metrics(font_path, size_px)
