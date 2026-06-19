# Audiblez — Remediation Plan

Derived from a deep multi-agent review (58 verified findings) + a doc-grounded research pass
(wxPython 4.x, ffmpeg m4b, Kokoro 0.9.4, spaCy 3.8, Poetry/PEP-621). Each batch is a
self-contained, PR-sized unit with: scope → exact changes → verification → findings closed.

Severity legend: 🔴 critical · 🟠 high · 🟡 medium · ⚪ low

---

## Batch 0 — Verification spikes (do FIRST, ~30 min, no production code)

These four runtime checks de-risk the plan. Two research claims are suspect and must be
confirmed before the code that depends on them is written.

| # | Check | Why it matters | Command / method |
|---|-------|----------------|------------------|
| 0.1 | `print(spacy.load('xx_ent_wiki_sm').pipe_names)` | Research claims the model ships a sentencizer (doubtful). Decides whether Batch 4 **removes** or **guards** `add_pipe('sentencizer')`. Guessing wrong → `doc.sents` crashes OR `E966` crash. | run in venv |
| 0.2 | Inspect type of `audio` yielded by `pipeline(...)` in kokoro 0.9.4 (numpy vs `torch.Tensor`; CPU vs CUDA) | `np.concatenate` on CUDA tensors fails. Decides whether we add `.cpu().numpy()`. The latest commit is literally "kokoro 0.9.4 attempt", so this is live. | `type(audio)`, `audio.device` |
| 0.3 | `deptry .` | Authoritative missing/unused dependency list to ground Batch 2. | `deptry .` |
| 0.4 | Confirm native `aac` builds a valid m4b on a sample epub | Validates Batch 1's encoder switch end-to-end. | run patched CLI on `epub/mini.epub` |

**Decision gates produced:** spaCy fix shape (0.1), audio-conversion need (0.2), exact dep edits (0.3).

---

## Batch 1 — 🔴 Unbreak m4b generation (the headline; ships first)

**Root problem:** `concat_wavs_with_ffmpeg` uses `-c:a libfdk_aac`, absent from standard ffmpeg
(confirmed: this Mac + Ubuntu CI both lack it). It fails silently (no `check`), then a second
ffmpeg fails on the missing intermediate with a misleading error — *after* ~1h of synthesis.

**Changes (`audiblez/core.py`):**
1. **Collapse to a single ffmpeg pass.** Replace `concat_wavs_with_ffmpeg` + `create_m4b` with one
   invocation: concat demuxer (input 0) + FFMETADATA1 (input 1) + cover (input 2), mapping
   `-map 0:a -map_metadata 1 -map 2:v -disposition:v attached_pic -c:v copy -c:a aac -b:a 64k`.
   Eliminates the double lossy transcode (192k→64k) and one temp file.
2. **Native encoder:** `libfdk_aac` → `aac` (universal). Keep 64k; consider `-ac 1` (mono) for speech.
3. **Real error handling:** capture stderr; `if proc.returncode != 0: raise RuntimeError(stderr)`.
   No more silent false-success / `CORE_FINISHED` over a corrupt file.
4. **Escape concat paths:** `str(wav).replace("'", "'\\''")` before writing `file '...'`.
   Fixes breakage on chapter names with apostrophes.
5. **Guard `probe_duration`:** wrap per-file calls; on failure, warn + skip that chapter's index
   entry instead of crashing the whole job post-synthesis.
6. **Fix chapter index file:** number from the real chapter position (parse from filename or pass
   index), use the **actual chapter title** instead of `Chapter {i}` starting at 0; escape
   newlines in `title=`/`artist=` (FFMETADATA injection hardening).
7. **Temp cleanup in `try/finally`** (wav list, cover, any intermediate).

**Reference (validated, from research):** single-pass arg order
`ffmpeg -y -f concat -safe 0 -i list.txt -i chapters.txt -i cover -map 0:a -map_metadata 1 -map 2:v -disposition:v attached_pic -c:v copy -c:a aac -b:a 64k -f mp4 out.m4b`

**Verify:** run `audiblez <sample>.epub`; confirm `.m4b` exists, plays in VLC, shows chapters + cover;
re-run to confirm error path raises loudly if ffmpeg is sabotaged.

**Closes:** libfdk_aac (🔴), missing concat error handling (🔴), README encoder gap (🔴 → moot),
double-encode (🟡), apostrophe concat break (🟡), silent m4b failure (🟡), temp cleanup (🟡),
ffprobe crash (🟠), chapter numbering (🟡), FFMETADATA injection (🟡).

---

## Batch 2 — 🟠 Make the GUI installable + packaging hygiene

**Root problem:** `audiblez-ui` imports `wx`, `PIL`, `torch` — none declared. Clean install → `ImportError`.

**Changes (`pyproject.toml`, PEP-621 format — drop the Poetry parenthesized version style):**
- Add hard deps: `torch>=2.0.0` (needed by kokoro core synthesis).
- Fix `bs4 (>=0.0.2,<0.0.3)` → canonical `beautifulsoup4>=4.12,<5`.
- Add optional extra:
  ```toml
  [project.optional-dependencies]
  ui = ["wxPython>=4.2.0", "Pillow>=10.0.0"]
  ```
- Remove genuinely unused deps **after `deptry` + grep confirm**: `qwen-tts`, `epub-toc`.
  ⚠️ **Do NOT blind-remove `misaki[zh]`** — it's kokoro's Chinese g2p backend (enables 🇨🇳 voices).
  Verify by running a `zf_*` voice before deciding; keep if it breaks.
- README: document `pip install audiblez[ui]` for the GUI; reconcile "v4" vs `0.4.9`.

**Optional resilience:** wrap the GUI entry point so a missing `[ui]` extra prints a friendly
"install audiblez[ui]" message instead of a raw traceback.

**Verify:** `deptry .` clean; fresh venv `pip install .` (CLI works) and `pip install .[ui]`
(`python -c "import wx, PIL"` works, `audiblez-ui` launches).

**Closes:** missing wxPython/Pillow/torch deps (🟠×3), unused deps (🟡), version mismatch (⚪).

---

## Batch 3 — 🟠 GUI correctness (wxPython 4.x)

**Changes (`audiblez/ui.py`), all research-confirmed:**
1. **`wx.EmptyImage(...)` → `wx.Image(...)`** at `ui.py:406` — the *only* hard-broken call (removed in
   Phoenix). Minimal one-word fix; rest of the SetData/Rescale/ConvertToBitmap chain is still valid.
2. **Thread-safe GUI updates:** wrap off-thread widget mutations in `wx.CallAfter` — preview's
   `button.SetLabel`/`Enable` (`ui.py:502-503`). (Synthesis path already uses `wx.PostEvent` correctly.)
3. **Re-enable controls when done:** `on_core_finished` must `Enable()` `start_button` + `params_panel`
   (currently disabled in `on_start` and never restored → can't run a 2nd audiobook). Remove the
   per-chapter `start_button.Show()` in `on_core_chapter_finished`.
4. **Remove `thread.join()` on the UI thread** (`ui.py:505-508`) — it freezes the event loop. Track the
   preview thread without blocking (e.g. is_alive guard, or disable the button while running).
5. **Guard `on_select_speed`** with `try/except ValueError`, revert to last valid value.
6. **Clean the preview temp file** after `ffplay` returns (`NamedTemporaryFile(delete=False)` leak).
7. **Fix bare imports:** `import core` → `from audiblez.core import main` in `CoreThread.run` (and
   `cli.py:38`), enabling Batch 5's removal of the `sys.path` hack.

**Verify:** launch GUI, open epub *with cover* (exercises #1), preview a chapter (exercises #2/#4/#6),
run synthesis to completion, then start a **second** run (exercises #3).

**Closes:** wx.EmptyImage crash (🟠), off-thread GUI (🟠), join() freeze (🟠), start-button-not-reenabled
(🟠), speed crash (🟡), temp leak (🟡), bare import (🟡).

---

## Batch 4 — 🟠 Core correctness + performance

**Changes (`audiblez/core.py`):**
1. **Cache spaCy once** (per research, but per Batch 0.1 decision): module-level `get_nlp()` lazy
   singleton that loads the model **and** adds the sentencizer **once, guarded by `has_pipe`**:
   ```python
   _nlp = None
   def get_nlp():
       global _nlp
       if _nlp is None:
           if not spacy.util.is_package("xx_ent_wiki_sm"):
               spacy.cli.download("xx_ent_wiki_sm")
           _nlp = spacy.load("xx_ent_wiki_sm")
           if "sentencizer" not in _nlp.pipe_names:
               _nlp.add_pipe("sentencizer")
       return _nlp
   ```
   Replace the per-chapter `spacy.load(...)` + `add_pipe(...)` in `gen_audio_segments`. Removes a full
   model load **per chapter** (30+ loads on a 30-chapter book).
2. **Off-by-one:** `i > max_sentences` → `i >= max_sentences` (`core.py:229`).
3. **Intro text:** introduce `intro_added` flag; attach the "Title – Author" intro to the first chapter
   **actually synthesized**, not hardcoded `i == 1` (currently lost if ch.1 is skipped).
4. **Guard empty audio:** in `gen_text`, mirror `main`'s `if audio_segments:` check before
   `np.concatenate`.
5. **Audio type safety (per Batch 0.2):** if 0.9.4 yields torch tensors, append `audio.cpu().numpy()`
   so `np.concatenate` works on CPU and CUDA alike.
6. **Remove dead code:** delete `unmark()` / `unmark_element()` (reference undefined `Markdown`).
7. **Consistency:** pass `repo_id='hexgrad/Kokoro-82M'` to `KPipeline` in both call sites (silences the
   0.9.4 default-repo warning).

**Verify:** unit tests for `max_sentences`, intro logic, empty-input; time a multi-chapter run before/after
to confirm the spaCy speedup.

**Closes:** spaCy reload (🟠), off-by-one (🟠), intro-skip (🟠), empty concat (🟡), Markdown dead code (🟡).

---

## Batch 5 — 🟡 Maintainability & hygiene

- Split `main()` into `parse_epub` / `select_chapters` / `synthesize_chapters` / `finalize_audiobook`.
- Module-level constants: `SAMPLE_RATE=24000`, `GPU/CPU_CHARS_PER_SEC`, `MAX_SENTENCE_LEN=400`.
- Introduce `logging` (replace ~40 `print`s; levels for info/warning/error).
- Extract shared `extract_book_metadata(book)` used by `core.main` + `ui.open_epub` (dedup).
- Add type hints to public functions.
- Remove the `sys.path` hack in `__init__.py` (now safe after Batch 3's import fixes).
- Remove commented-out dead UI blocks (md/txt/pdf/stop buttons).
- Fix `split_long_sentence` docstring (500 → 400); convert O(n²) `in list` checks to `set`.

**Verify:** full test suite green; CLI + GUI smoke test unchanged.

---

## Batch 6 — 🟠 Testing & CI

- **Bundle a tiny fixture epub** under `test/fixtures/`; make `test_find_chapters`, `test_cli`,
  `test_main` hermetic (no network for unit tests).
- **Fix `test_cli.py`:** `python` → `python3`, replace `os.popen`+`cd ..` with `subprocess.run(..., cwd=root)`.
- **Re-enable unit tests in CI**; split into a hermetic **unit** job and an optional **integration**
  (network-download) job.
- **Add unit tests** for the now-isolated functions: `strfdelta`, `split_long_sentence`, `is_chapter`,
  `find_cover`, the new FFMETADATA/chapter builder, `get_nlp`, off-by-one, intro logic.
- Add a UI **import** smoke test gated on the `[ui]` extra.

**Closes:** CI tests disabled (🟠), broken test_cli (🟠×2), missing fixtures (🟡), empty/skip tests (⚪),
no unit coverage for helpers (⚪), no UI tests (🟡).

---

## Already-correct (no action) — verified by the review's adversarial stage
- **Subprocess usage is safe** (list-form, no `shell=True`).
- **No path traversal** (`/`,`\` sanitized; `Path() / str` keeps one component).
- **No XXE** (lxml disables external entities by default).

---

## Sequencing & parallelism

```
Batch 0 (spikes) ─► gates everything
Batch 1 (ffmpeg)  ┐
Batch 2 (packaging)┤ independent of each other → can run in parallel
Batch 3 (GUI)      ┘ (Batch 3 best validated after Batch 2 installs ui deps)
Batch 4 (core)    ─► touches core.py like Batch 1 → run AFTER Batch 1 to avoid conflicts
Batch 5 (refactor)─► after Batch 4 (also core.py); largest churn
Batch 6 (tests/CI)─► last, validates everything
```

⚠️ Batches 1, 4, 5 all edit `core.py` — keep them **sequential** (or in separate worktrees merged in
order) to avoid conflicts. Batch 2 (pyproject) and Batch 3 (ui.py) are conflict-free with the core work
and can proceed in parallel.

**Recommended ship order:** 0 → **1** (unblocks users) → **2** (unblocks GUI installs) → 3 → 4 → 6 → 5.
Batch 1 alone is the highest-value, lowest-risk PR and should go out on its own immediately.
