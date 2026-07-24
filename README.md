# premiere-ai

AI-assisted video-production workflows for Adobe Premiere Pro, built on top
of [premiere-cli](https://github.com/stefanwebb/premiere-cli) (which
provides the `premiere-cli`/`premiere-log` CLIs and the Premiere Bridge CEP
panel).

## Installation

```bash
pip install -e .
```

## Commands

### `transcribe <input_file>`

Transcribes an audio or video file in two passes:

1. **VibeVoice-ASR** (`mlx-community/VibeVoice-ASR-4bit`) — produces a transcript in `.txt`, `.srt`, or `.vtt` format
2. **Qwen3-ForcedAligner** (`mlx-community/Qwen3-ForcedAligner-0.6B-4bit`) — aligns each ASR segment to produce word-level timestamps saved as `.words.json`

Video files have their audio extracted automatically via ffmpeg before transcription.

```
transcribe video.mp4
transcribe audio.wav --format srt --language English
transcribe audio.m4a --output transcript --verbose
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output / -o` | input filename stem | Output path (without extension) |
| `--format / -f` | `txt` | Transcript format: `txt`, `srt`, or `vtt` |
| `--language / -l` | `English` | Language name for the forced aligner |
| `--verbose / -v` | off | Show inference progress |

**Output files:**
- `<stem>.txt` / `.srt` / `.vtt` — transcript
- `<stem>.words.json` — word-level timestamps as `[{"text": "word", "start": 0.123, "end": 0.456}, …]`

**Prerequisites** — download models before first use:
```
hf download mlx-community/VibeVoice-ASR-4bit
hf download mlx-community/Qwen3-ForcedAligner-0.6B-4bit
```

---

### `remove-pauses <input_file>`

Detects pauses to remove from an audio or video file:

1. Transcribes it (reusing `transcribe`'s output if a matching
   `.words.json` already exists next to the input).
2. Asks Claude (`claude-opus-4-8`, via the Anthropic API) which word-gaps
   are phrase/sentence boundaries — by default, only these are considered
   candidates for cutting. Pass `--allow-mid-phrase-cuts` to skip this
   check entirely and consider every word-gap a candidate.
3. Runs Silero VAD (via `torch.hub`) to confirm which candidate gaps
   are actually silent.
4. Narrows each confirmed pause with asymmetric frame margins controlled by
   `--aggressiveness`: a small fixed buffer after the preceding word, and a
   larger buffer before the next word that shrinks as aggressiveness
   increases (since cutting too close to the next word risks clipping it).

Each cut range is a half-open `[start, end)` interval: `start` is the first
frame to delete, `end` is the first frame that remains — matching the
standard NLE in/out-point convention.

```
remove-pauses recording.wav
remove-pauses video.mp4 --aggressiveness 0.8 --min-pause 250
remove-pauses interview.mp4 --allow-mid-phrase-cuts
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--aggressiveness / -a` | `0.5` | `0`=conservative, `1`=aggressive (cuts tighter to the next word) |
| `--min-pause` | `300` | Minimum pause duration in ms worth cutting |
| `--fps` | `25` | Frame rate used for `MM:SS:FF` timecodes |
| `--language / -l` | `English` | Language name for the forced aligner (when transcribing) |
| `--output / -o` | `<stem>.cuts.txt` | Path to write the cut list |
| `--allow-mid-phrase-cuts` | off | Also cut pauses that aren't at a phrase/sentence boundary (skips the Claude boundary check entirely) |
| `--verbose / -v` | off | Show inference progress |

**Output:**
- Printed to stdout and written to `<stem>.cuts.txt`: one `MM:SS:FF - MM:SS:FF` pause range per line, plus a final "Total pause time removed: M:SS.s" line (when any cuts are found).

**Prerequisites:**
- `ANTHROPIC_API_KEY` must be set (in the environment or a `.env` file) unless `--allow-mid-phrase-cuts` is used, since that flag skips the Claude API call entirely.
- First run downloads the Silero VAD model via `torch.hub` (requires network access).

---

### `zmbv-to-h265-vga <input_file> [output_file]`

Converts a DOSBox ZMBV screen recording (VGA palette) to H.265/HEVC. Scales the frame up 2× using nearest-neighbour to preserve pixel-art crispness.

---

### `zmbv-to-h265-ega <input_file> [output_file]`

Same as above for EGA palette recordings.

---

### `import-raw-footage <project_dir>`

Locates the latest matching raw camera recording and mic recording,
matches them by capture time (falling back to media duration), and
copies both into `<project_dir>/assets/video/` and
`<project_dir>/assets/audio/`.

```
import-raw-footage /path/to/project
import-raw-footage /path/to/project --camera-file cam.mp4 --mic-file mic.wav
```

This package makes no assumption about your camera or mic hardware —
auto-detection is entirely driven by environment variables, each
optional and comma-separated for multiple locations:

| Variable | Purpose |
|----------|---------|
| `PREMIERE_AI_CAMERA_GLOBS` | Glob pattern(s) for camera clips, e.g. `/Volumes/MyCamera/DCIM/**/*.MP4` |
| `PREMIERE_AI_MIC_FLAT_ROOTS` | Directories checked directly (non-recursive) for `*.wav`/`*.WAV` |
| `PREMIERE_AI_MIC_RECURSIVE_ROOTS` | Directories searched recursively for `*.wav`/`*.WAV` |

Any variable left unset just means that source is skipped; passing both
`--camera-file` and `--mic-file` explicitly needs none of them set.
Full flag reference: `import-raw-footage --help`.

---

### `calibrate-lut <image>`

Builds a 3D `.cube` correction LUT from a single photo of the **video
page** of a Calibrite ColorChecker Passport Video 2 chart: detects the
chart (SAM3), measures its patches, and fits

- per-channel (R, G, B) tone curves for exposure/white-balance, from the
  left-panel 3-step strip + grid columns 2 and 3 against IRE targets
- a single global hue rotation aligning the six chromatic chips (col 0)
  to their Rec.709 vectorscope target hues, folding in the skin-tone chips
  (col 1) toward the classic "-I axis" skin-tone line
- highlight/shadow anchors (col 3) folded into the same tone curves, at
  IRE targets you can override

```
calibrate-lut chart_photo.png
calibrate-lut chart_photo.png --output corrected.cube --lut-size 33
calibrate-lut chart_photo.png --highlight-shadow-ire 100,97,94,6,3,0
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output / -o` | `<image>.cube` | Output `.cube` path |
| `--lut-size` | `17` | LUT lattice size (per axis) |
| `--highlight-shadow-ire` | `100,97,94,6,3,0` | 6 comma-separated IRE percentages for the highlight/shadow column, brightest row first — Calibrite doesn't publish exact values for this column, so override with real ones if you obtain them |

**Prerequisites** — download the segmentation model before first use:
```
hf download mlx-community/sam3-4bit
```

**Caveats** (see the module docstring in `build_lut.py` for the full
reasoning): per-channel WB/exposure uses real IRE targets, and the
chromatic hue targets are exact (computed analytically from Rec.709) —
but the skin-tone target and default highlight/shadow IRE values are
reasoned estimates, not vendor-confirmed figures, so verify visually
against Premiere's own skin-tone-line toggle before trusting them as
exact. This only ever measures the chart's **video** page — see
`calibrate-lut-classic` below for the traditional 24-patch page.

The resulting `.cube` is exactly the input `premiere-cli`'s
[`apply-lut`/`desktop-set-input-lut`](https://github.com/stefanwebb/premiere-cli)
commands consume to apply the correction to a clip's Lumetri Color effect.

---

### `calibrate-lut-classic <image>`

Builds a 3D `.cube` correction LUT from a photo of the **classic**
(traditional 24-patch) page of an X-Rite/Calibrite ColorChecker: locates
the chart (SAM3), segments its 24 squares, matches each to its published
reference sRGB value (Hungarian assignment across the whole grid, so no
single ambiguous patch can steal another's slot — cross-checked by finding
which single rotate/flip orientation is consistent across the most
matched patches, excluding any that disagree from the fit), and fits a
color-correction matrix from measured colors to those references.

```
calibrate-lut-classic chart_photo.png
calibrate-lut-classic chart_photo.png --output corrected.cube --lut-size 33
calibrate-lut-classic chart_photo.png --matrix-method "Finlayson 2015" --matrix-degree 2
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--output / -o` | `<image>.cube` | Output `.cube` path |
| `--lut-size` | `17` | LUT lattice size (per axis) |
| `--matrix-method` | `Cheung 2004` | `Cheung 2004` (plain 3x3 + offset affine fit) or `Finlayson 2015` (exposure-invariant root-polynomial expansion, more fitted terms) |
| `--matrix-terms` | `4` | Cheung 2004 augmentation terms (4 = 3x3 + offset) |
| `--matrix-degree` | `2` | Finlayson 2015 root-polynomial degree |

**Prerequisites** — same as `calibrate-lut`:
```
hf download mlx-community/sam3-4bit
```

**Caveats**: reference values are BabelColor/Danny Pascale's published
nominal sRGB measurements for the Classic chart, not the physical unit's
own individually-calibrated values — good enough to verify grid
orientation and fit a real hue/saturation/white-balance correction
against, but this fits **only** a color matrix, no separate exposure/tone
curve. Unlike `calibrate-lut`, this is a single-photo, single-page
pipeline throughout — it does not attempt to combine anchors from a
separate video-page shot (an earlier exploration combining both pages
from two unrelated photos is preserved in `video-production`'s scratch
history as a cautionary example: that combination is only physically valid
when both pages are photographed together, under the same lighting, in
the same frame).

#### Diagnostics (`premiere_ai.colorchecker`)

`calibrate-lut`'s supporting modules also include standalone diagnostic
tools for troubleshooting chart detection/segmentation on new footage —
not installed as console scripts (debug tools, not end-user commands), run
via `python -m`:

| Module | Purpose |
|--------|---------|
| `segment` | Confirms VLM chart-location + SAM3 segmentation agree |
| `subsegment` | Visualizes the 24 individual patch segments |
| `segment_left_target` | Visualizes the left target panel's segmentation |
| `extract_targets` | Extracts just the chart's target regions from a photo |
| `identify_video_patches` | Grid-recovery + colorimetric sanity checks on the video page |
| `identify_classic_patches` | Grid-recovery + reference-matched identity checks on the classic page |
| `apply_cube_lut` | Applies a `.cube` to an image, for visual before/after comparison |
| `vectorscope_render` | Renders a Lumetri-style YUV vectorscope from an image |

```
python -m premiere_ai.colorchecker.vectorscope_render before.png after.png -o compare.png --labels before after
```

Several of these default their `--image` argument to a fixture image
(`colorchecker.png`) that isn't bundled in this package — pass `--image`
explicitly.

The chart-location step (`detect.py`) is currently **stubbed** to a fixed
simulated response rather than querying a live VLM server, so downstream
work isn't blocked on one being available — see its module docstring
before relying on it for a real detection.

---

### `premiere-log` / `premiere-cli` (from the premiere-cli package)

The Premiere-driving CLIs — `premiere-log` (send a message to the
Premiere Bridge panel's log view) and `premiere-cli` (execute
ExtendScript-backed commands against the open project) — live in the
separate [premiere-cli](https://github.com/stefanwebb/premiere-cli)
package, installed automatically as a dependency of this one. See that
repo's README and `docs/COMMANDS.md` for the full command reference,
including `premiere-cli init-project`, which creates a fresh empty
project from a bundled template (formerly this package's
`create-empty-premiere-project`).

---

## File structure

```
src/premiere_ai/
    transcribe.py          transcription + forced alignment CLI
    pause_cuts.py          pure pause-detection logic (timecodes, margins, VAD inversion, Claude parsing)
    remove_pauses.py       remove-pauses CLI orchestration
    scripts.py             thin Python wrappers for the shell scripts
    sync_audio.py          sync-audio / audio-offset CLI
    import_raw_footage.py  import-raw-footage CLI
    zmbv_to_h265_vga.sh    VGA ZMBV → H.265 conversion
    zmbv_to_h265_ega.sh    EGA ZMBV → H.265 conversion
    font_metrics.py        font measurement utilities
    colorchecker/           ColorChecker chart -> Lumetri correction-LUT pipeline
        build_lut.py            calibrate-lut CLI (video-page production command)
        build_lut_classic.py    calibrate-lut-classic CLI (classic-page production command)
        pages.py                per-page (classic/video) chart config
        patch_grid.py           grid recovery + colorimetric self-consistency checks
        classic_reference.py    published reference sRGB values for the classic page
        vectorscope.py          Rec.709 vectorscope geometry / hue targets
        tone_curve.py           per-channel tone-curve fitting
        lut_io.py / image_io.py .cube read/write; color-managed image loading
        detect.py               VLM chart-location query (currently stubbed)
        segment.py, subsegment.py, segment_left_target.py,
        extract_targets.py, identify_video_patches.py,
        identify_classic_patches.py, apply_cube_lut.py, vectorscope_render.py
                                 diagnostic tools, not installed as console scripts
tests/                     pytest suite mirroring the modules above
```

## Development

```bash
git clone https://github.com/stefanwebb/premiere-ai
cd premiere-ai
pip install -e ".[dev]"
pytest
```

## Scope

This package contains the AI-assisted workflow layer for video production.
Everything that drives Premiere Pro itself (ExtendScript-backed commands,
the CEP bridge panel) lives in the separate
[premiere-cli](https://github.com/stefanwebb/premiere-cli) package, which
this one depends on.

## License

[CC-BY-SA-4.0](LICENSE).
