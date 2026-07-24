"""ColorChecker Passport Video 2 -> Lumetri correction-LUT pipeline.

Detects the chart in a photo (VLM + SAM3), measures its patches, and fits
a 3D .cube LUT — the file premiere-cli's `apply-lut`/`desktop-set-input-lut`
then applies to a clip. `build_lut.py` (the `calibrate-lut` command) is the
production entry point; the rest are library/diagnostic modules it depends
on or that are useful when troubleshooting chart detection on new footage.
"""
