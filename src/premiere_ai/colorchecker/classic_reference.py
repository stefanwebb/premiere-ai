"""Published nominal sRGB values for the 24-patch X-Rite/Calibrite
ColorChecker Classic, in its standard 4-row x 6-col layout (row 0 = top,
col 0 = left, matching how every vendor/reference source numbers patches
1-24 row-major). These are widely published values (e.g. via BabelColor /
Danny Pascale's measurements), not the physical unit's own calibrated
values — good enough to verify grid identity/orientation, not for a final
color-managed LUT (use the unit's own reference file for that, once we
have it).
"""

ORIG_SHAPE = (4, 6)  # (rows, cols)

# (name, orig_row, orig_col, srgb)
CLASSIC_PATCHES = [
    ("dark_skin", 0, 0, (115, 82, 68)),
    ("light_skin", 0, 1, (194, 150, 130)),
    ("blue_sky", 0, 2, (98, 122, 157)),
    ("foliage", 0, 3, (87, 108, 67)),
    ("blue_flower", 0, 4, (133, 128, 177)),
    ("bluish_green", 0, 5, (103, 189, 170)),
    ("orange", 1, 0, (214, 126, 44)),
    ("purplish_blue", 1, 1, (80, 91, 166)),
    ("moderate_red", 1, 2, (193, 90, 99)),
    ("purple", 1, 3, (94, 60, 108)),
    ("yellow_green", 1, 4, (157, 188, 64)),
    ("orange_yellow", 1, 5, (224, 163, 46)),
    ("blue", 2, 0, (56, 61, 150)),
    ("green", 2, 1, (70, 148, 73)),
    ("red", 2, 2, (175, 54, 60)),
    ("yellow", 2, 3, (231, 199, 31)),
    ("magenta", 2, 4, (187, 86, 149)),
    ("cyan", 2, 5, (8, 133, 161)),
    ("white_95", 3, 0, (243, 243, 242)),
    ("neutral_8", 3, 1, (200, 200, 200)),
    ("neutral_65", 3, 2, (160, 160, 160)),
    ("neutral_5", 3, 3, (122, 122, 121)),
    ("neutral_35", 3, 4, (85, 85, 85)),
    ("black_2", 3, 5, (52, 52, 52)),
]

REFERENCE_NAMES = [p[0] for p in CLASSIC_PATCHES]
REFERENCE_POSITIONS = [(p[1], p[2]) for p in CLASSIC_PATCHES]
REFERENCE_RGBS = [p[3] for p in CLASSIC_PATCHES]
