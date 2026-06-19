# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

audiblez converts e-books (EPUB) into audiobooks (`.wav` / `.m4b`) using the Kokoro-82M TTS model.
CLI (`audiblez`) + optional wxPython GUI (`audiblez-ui`). Offline, single-process, Python 3.10–3.12.
This is a **personal fork** of santinic/audiblez, extended with preflight checks (`--doctor`), an
opt-in sentence cache, a book-scoped pronunciation lexicon, sentence-level resilience, measured-ETA,
and GPU tuning. Preserve that offline single-process simplicity (no daemon / web rewrite).

## Commands (non-obvious ones)

- **Tests:** `unittest`, NOT pytest — see `.claude/rules/testing.md`, or just `/test`.
- **CLI:** `audiblez book.epub` auto-selects the best *working* backend (probes the GPU, falls back
  to CPU); force with `-b cpu`. Preflight a long run with `audiblez --doctor [--deep]`.
- **GUI:** `audiblez-ui` (needs the `[ui]` extra).
- **Lint:** `ruff check .` (config in `pyproject.toml [tool.ruff]`; `ruff check --fix` to autofix). CI gates on it.
- **Dependency check:** `deptry .`.
- The main dev env is a ROCm `.venv` (`torch 2.12.1+rocm7.2`, gfx1101) auto-activated by direnv; alt
  TTS models live in **isolated** `.venv-<model>` envs — never install into the main `.venv`.

## Architecture

`EPUB → chapters → per-sentence synth (batched) → chapter .wav → single ffmpeg pass → .m4b`. The
engine seam is `core.build_synthesizer`; three callers (`cli.py`, `ui.py`, `core.py`) must stay in
sync on any signature change. Modules `backends`/`doctor`/`cache`/`lexicon`/`gpu` are deliberately
import-light so `--doctor` works on a broken install. Details in the rules below.

## Project rules

Focused, binding rules live in `.claude/rules/` and load automatically:
- **architecture.md** — the synth seam & the 3 entry points to keep in sync
- **imports-and-deps.md** — import-light modules; indirectly-used deps not to "clean up"
- **testing.md** — how to actually run the tests
- **cache-and-resilience.md** — cache-key correctness & degraded-run invariants (silent-wrongness landmines)
- **tts-and-runtime.md** — model selection (ignore licenses) & the AMD/ROCm runtime reality

## Persistent project context

Long-form planning/research is on disk (read it when picking up ongoing work): `task_plan.md`,
`findings.md`, `progress.md` (planning-with-files), `TTS_MODELS_COMPARISON.md` (model scoreboard),
`CODE_REVIEW_next-wave.md` (deep code review of the current branch's changes).

## Gotchas

- `--doctor` must stay fast and dependency-light — never add a heavy top-level import to its path.
- Don't commit generated artifacts: `.npy` / `.csv` / `.log` / `.sig` / `.failed.jsonl` /
  `.audiblez_cache/` / `test/e2e_out/` are gitignored — keep it that way.
