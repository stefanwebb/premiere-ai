# Changelog

## 0.3.0 — 2026-07-23

### New features

- **`calibrate-lut` command** — builds a 3D `.cube` correction LUT from a
  photo of a ColorChecker Passport Video 2 chart's video page (chart
  detection via SAM3, per-channel tone-curve + hue-rotation fitting). The
  resulting LUT is what `premiere-cli`'s `apply-lut` (or, until it's
  rebuilt, a manual Lumetri Input LUT load) then applies to a clip. Ships
  with a `premiere_ai.colorchecker` module of supporting diagnostic tools
  for troubleshooting chart detection on new footage. See the README for
  caveats — several fitted targets are reasoned estimates, not
  vendor-confirmed figures.
- **`calibrate-lut-classic` command** — same pipeline for the traditional
  24-patch (classic) ColorChecker page: fits a color-correction matrix
  (Cheung 2004 or Finlayson 2015, via colour-science) mapping measured
  patches to their published reference sRGB values, cross-checked by grid
  orientation consistency.
- **`vectorscope` command** — promotes the `vectorscope_render` diagnostic
  module to an installed console script for rendering a Lumetri-style YUV
  vectorscope trace from one or more images. Adds
  `--colorchecker-patches {chromatic,skin}`, which detects a Calibrite
  ColorChecker Passport Video 2's video page (same SAM3 pipeline as
  `calibrate-lut`) and restricts the trace to just that category of its
  24-patch grid, plus `--debug-mask-output` to save the detected patch
  region as a dimmed overlay for confirming the detection landed
  correctly before trusting the trace. Detection logic lives in the new
  `premiere_ai.colorchecker.patch_mask` module.

## 0.2.0 — 2026-07-20

### Improvements

- `import-raw-footage`'s camera/mic sources are now env-var configurable

### Infrastructure / Documentation

- Removed `create-empty-premiere-project`, superseded by `premiere-cli init-project`
- Added CC-BY-SA-4.0 license, matching `premiere-cli`

## 0.1.0 — 2026-07-18

Initial release: `premiere-ai` extracted from `video-production`.
