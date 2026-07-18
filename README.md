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

### `create-empty-premiere-project <project_name> [--series <series_name>]`

Creates a fresh, empty Premiere Pro project by copying the shared empty
template and renaming its `.prproj` file to match the new project.

```
create-empty-premiere-project vlog0002
create-empty-premiere-project "episode 2" --series vlog
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--series / -s` | none | Nests the project under `<base-dir>/<series>/<project_name>` instead of directly under `<base-dir>` |
| `--base-dir` | `/Volumes/Extreme Pro/video-production/Generative GameDev` | Override the destination base directory |

**Template:** `/Volumes/Extreme Pro/video-production/Shared Assets/Empty Premiere Pro Template`

Refuses to run if the destination directory already exists.

---

### `premiere-log` / `premiere-cli` (from the premiere-cli package)

The Premiere-driving CLIs — `premiere-log` (send a message to the
Premiere Bridge panel's log view) and `premiere-cli` (execute
ExtendScript-backed commands against the open project) — live in the
separate [premiere-cli](https://github.com/stefanwebb/premiere-cli)
package, installed automatically as a dependency of this one. See that
repo's README and `docs/COMMANDS.md` for the full command reference.

---

## File structure

```
src/premiere_ai/
    transcribe.py          transcription + forced alignment CLI
    pause_cuts.py          pure pause-detection logic (timecodes, margins, VAD inversion, Claude parsing)
    remove_pauses.py       remove-pauses CLI orchestration
    scripts.py             thin Python wrappers for the shell scripts
    create_empty_premiere_project.py  create-empty-premiere-project CLI
    sync_audio.py          sync-audio / audio-offset CLI
    import_raw_footage.py  import-raw-footage CLI
    zmbv_to_h265_vga.sh    VGA ZMBV → H.265 conversion
    zmbv_to_h265_ega.sh    EGA ZMBV → H.265 conversion
    font_metrics.py        font measurement utilities
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
