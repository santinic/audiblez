# Audiblez — Deep Codebase Analysis

**Date:** 2026-06-02
**Method:** 13-dimension multi-agent static audit (93 agents, ~2.7M tokens). Each dimension reviewer's
findings were independently re-checked by an adversarial verifier instructed to *refute* them; only
confirmed/adjusted findings survive. A completeness critic then searched for cross-module gaps the
per-file reviewers missed. **78 findings confirmed** (1 rejected as a false positive), plus 6
cross-cutting findings and 5 optimization opportunities.

> Scope note: this is a *read-only* audit of the current working tree. Nothing was modified. The
> recently-applied `FIX_PLAN.md` fixes (single-pass ffmpeg/native aac, spaCy caching, blend system,
> backend registry, FFMETADATA/concat escaping) were independently re-verified — most are **correct
> and complete**; the residual issues below are either in unchanged code or are gaps the fixes left open.

## Severity distribution

| Severity | Count | Meaning |
|----------|-------|---------|
| **P0** | 1 | Breaks a documented core flow |
| **P1** | 12 | High-impact defect hit in normal use |
| **P2** | 27 | Meaningful edge case / perf / maintainability trap |
| **P3** | 38 | Minor / narrow / advisory |

Verifier rejected 1 finding (a claim that `misaki[zh]` is a redundant dependency — refuted: kokoro
declares only `misaki[en]`, so the direct `[zh]` dependency is *required* for Chinese voices).

---

## P0 — Critical (fix before next release)

### #1 · Relative `-o` output folder breaks m4b assembly — `core.py:489`
`audiblez book.epub -o subdir` (a documented CLI flag) **fails the entire m4b pass**. The ffmpeg
concat demuxer resolves relative entry paths *relative to the list file's directory*, not the cwd.
Chapter paths are written as `Path(output_folder)/'..._.wav'` (`core.py:162`) into a list file that
*also* lives in `output_folder`, producing a doubled prefix (`subdir/subdir/book_chapter_1.wav`) →
"No such file or directory" → `RuntimeError`. Works for `-o .` and for absolute paths (the GUI always
passes `os.path.abspath('.')`), which is why it slipped through.
**Fix:** write absolute paths into the concat list — `_escape_concat_path(Path(wav_file).resolve())`.

---

## P1 — High (should fix)

Grouped by theme. All independently verified; line numbers confirmed against current code.

### Audio pipeline correctness
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 6 | `core.py:196` | **GUI hangs forever when ffmpeg is missing.** `CORE_FINISHED` is emitted only inside `if has_ffmpeg:`. No ffmpeg → WAVs synthesize fine but the event never fires → progress bar stuck <100%, Start button stays disabled, `synthesis_in_progress` stuck True. | Move `post_event('CORE_FINISHED')` out of the `has_ffmpeg` block; optionally pass a "m4b skipped" flag. |
| 9 | `core.py:164` | **Resume re-speaks the book intro.** The skip-existing branch `continue`s without setting `intro_added=True`, so on a resumed run the "Title – Author." intro is prepended *again* to the next synthesized chapter — spoken twice in the final m4b. | Set `intro_added=True` for the first non-empty chapter even when its wav is skipped. |

### Security
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 10 | `core.py:160` | **Concat-list injection via EPUB chapter name.** `get_name()` is attacker-controlled (ebooklib `unquote`s the manifest href). The sanitizer strips only space/`/`/`\` — **not `\n`/`\r`**. A crafted href decodes to a filename with a literal newline; `_escape_concat_path` escapes quotes but is powerless against a newline, which splits the `file '...'` directive so ffmpeg (`-safe 0`) interprets injected lines as new directives → can splice arbitrary local audio into the m4b. | Whitelist-sanitize: `re.sub(r'[^A-Za-z0-9._-]', '_', name)`. |

### Input validation
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 7 | `voices.py:107` | **`-v 'af_heart:inf,af_bella:1'` crashes with a raw traceback.** `float('inf')` parses fine, then `Fraction(inf).limit_denominator(100)` raises **`OverflowError`** — but cli.py:42 and ui.py:410 catch only `ValueError`. In the GUI the `EVT_TEXT` handler raises mid-typing and breaks the event loop. | After `float(raw)`: `if not math.isfinite(weight): raise ValueError(...)`. |

### Backend UX (project's primary target hardware)
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 8 | `cli.py:59` | **`--backend cuda` on an AMD/ROCm box silently downgrades to CPU** (~10× slower) even though the GPU is fully usable as `rocm`. The deprecated `--cuda` alias *does* cross-map to rocm, so the modern flag is strictly worse than the legacy one. Directly affects the gfx1101/RX-7800-XT ROCm user. | Before the generic fallback, map `cuda↔rocm` when the sibling is available. |

### Portability
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 5 | `core.py:87` | **Windows espeak path never globs its `*`.** Linux/Darwin pass their wildcard through `glob()`; the Windows branch assigns the literal string `C:\Program Files*\eSpeak NG\libespeak-ng.dll` straight to `set_library()` → non-existent path → phonemization fails on every standard Windows install. CI only runs `--help`, so it's invisible. | `hits = glob(pattern); library = hits[0] if hits else None`. |

### Extraction completeness
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 3 | `core.py:350` | **Text in `<div>`/`<span>` (and direct `<blockquote>` text) is silently dropped.** `html_content_tags` is `['title','p','h1','h2','h3','h4','li']`. Calibre/InDesign exports and poetry/quote-heavy books that wrap prose in `<div>` produce empty chapters → silently omitted from the audiobook, no warning. | Add `div`/`blockquote`, or fall back to `soup.get_text('\n')` when the tag list yields nothing. |

### GUI robustness
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 12 | `ui.py:420` | **`open_epub` destroys the layout *before* any fallible work**, with no try/except. A malformed EPUB, corrupt cover, or empty chapter set (`good_chapters[0]` IndexError) leaves a **permanently blank window** — the exception is swallowed by `wx.CallAfter`. | Validate + decode into locals first, or wrap in try/except that restores the start hint + shows a MessageBox. |
| 13 | `ui.py:606` | **App exit during synthesis = hang / use-after-free.** `on_exit` just `Close()`s with no `EVT_CLOSE` guard. `CoreThread` is non-daemon, so it keeps posting events to the destroyed window (`RuntimeError: wrapped C/C++ object deleted`) and the interpreter won't exit until synthesis finishes. The `on_open` path *was* hardened against in-progress synthesis; the exit path was not. | Bind `EVT_CLOSE` → veto-or-flag while in progress; null-check `GetTopWindow()` in `post_event`; make `CoreThread` a daemon. |

### Throughput
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 11 | `core.py:315` | **Serial, per-sentence synthesis is the dominant bottleneck.** Each spaCy sentence is a separate `pipeline()` call — thousands of round-trips per book, GPU mostly idle between tiny calls. Chapters are also strictly serial despite being independent. | Batch paragraph-sized blocks into one Kokoro call (let `split_pattern` chunk internally); pipeline/parallelize independent chapters. See Optimizations. |

### Testing / CI
| # | File:line | Issue | Fix shape |
|---|-----------|-------|-----------|
| 2 | `.github/workflows/git-clone-and-run.yml:26` | **`test_voices.py` (25 hermetic tests, the headline blend system) is run by no CI workflow.** A regression in blend math ships green. | Add `test_voices.py` to the unit-test step (or switch to `test_*.py` discovery once fixture-bound files are skip-clean). |
| 4 | `core.py:448` | **The headline ffmpeg/m4b rewrite has zero unit coverage** (`create_m4b`/`create_index_file`/`probe_duration`). `create_index_file` is trivially testable by mocking `probe_duration` — assert START/END accumulation, 1..N numbering, FFMETADATA escaping, RuntimeError-on-failure, temp cleanup. | Add hermetic `test_m4b.py`; wire into CI. |

---

## P2 — Moderate (27)

### Correctness
- **#18** `core.py:351` — Nested content tags extracted **twice** → duplicated narration.
- **#19** `core.py:352` — Spurious `.` appended to every heading/title and to empty paragraphs.
- **#30** `core.py:351` — Internal newlines / whitespace runs preserved in extracted text.
- **#31** `core.py:364` — `is_chapter` regex matches **substrings** → false-positive chapters.
- **#32** `core.py:211` — `find_cover` raises `KeyError` on OPF `<meta name="cover">` lacking a `content` attr.
- **#29** `core.py:461` — `probe_duration` None→0 silently corrupts cumulative chapter-marker offsets.
- **#33** `core.py:58` — Unguarded lazy-init **race on the global spaCy `_nlp`**, reachable from concurrent GUI preview/synth threads.
- **#37** `ui.py:437` — `good_chapters[0]` IndexErrors on an EPUB with no usable document.
- **#28** `backends.py:53` — Broad `except Exception: pass` hides real torch failures behind a silent CPU-only result.
- **#39** `voices.py:180` — Unbounded blend weights build a multi-MB voice string → OOM at synthesis.

### UX
- **#15** `cli.py:25` — `-s/--speed` accepts any float (0, negative, 5.0) despite help claiming 0.5–2.0.
- **#16** `cli.py:65` — Missing/typo'd epub path fails *deep in the heavy stack* (after loading kokoro+torch+spaCy) instead of at the CLI.
- **#17** `core.py:196` — All-skipped/empty chapter set → cryptic ffmpeg error instead of a clear message.
- **#24** `ui.py:637` — `CoreThread` error path leaves the GUI half-reset (checkboxes + Start visibility not restored).
- **#38** `ui.py:575` — Unchecking *all* chapters in the GUI is silently overridden by core's auto-select.

### Performance
- **#20** `core.py:299` — **spaCy NER runs on every chapter but its output is discarded** (only `doc.sents` is used). ~163× slower splitting than needed. Pure win to disable.
- **#21 / #40** `core.py:184/181` — Whole-chapter audio held in a list then `np.concatenate`'d → doubled peak RAM (~600MB/chapter, GB for unsplit books). Stream to `soundfile.SoundFile` instead.

### Testing / CI / packaging
- **#14** `pip-install.yaml:15` — **CI installs audiblez from PyPI**, so it never exercises working-tree code.
- **#22** `core.py:296` — No coverage for `gen_audio_segments` non-english splitting, `max_sentences` boundary, or the `max_sentences=0` falsy trap.
- **#26** `test_find_chapters.py` — Entirely un-runnable (all 7 tests read missing `../epub/*.epub`).
- **#27** `test_main.py` — Network-dependent, slow, one test bound to a missing local file + live audio playback.
- **#25** `poetry.lock:936` — Stale: pins kokoro **0.9.2** while pyproject requires **0.9.4**.

### Maintainability
- **#35** `core.py:111` — `main()` is a 90-line god-function (parse / select / estimate / synth / finalize).
- **#36** `core.py:141` — ~40 bare `print()` calls instead of `logging` (no levels, GUI can't surface them).
- **#34** `core.py:85` — Linux espeak `glob(...)[0]` IndexErrors on miss → swallowed into a misleading "you haven't installed espeak-ng" message even when it *is* installed elsewhere.

---

## P3 — Low (38, condensed)

Notable ones (full list in the audit JSON):

- **#42** `core.py:321` — Intro chars inflate `processed_chars` past `total_chars` → **negative ETA** near the end.
- **#43** `core.py:455` — `chapters.txt` uses a fixed name in the output folder (collision / silent overwrite); **also never cleaned up** (left behind every run — see cross-cutting #6).
- **#45** `core.py:328` — `gen_text()` is **dead code** (unreachable from CLI/GUI/tests).
- **#47/#48** `voices.py:154/156` — Single voice with explicit weight skips the positivity check; duplicate ids in a blend aren't coalesced.
- **#49** `pyproject.toml:23` — Invalid PEP-621 `exclude` key under `[project]` is silently ignored by the build backend.
- **#54** `core.py:463` — Chapter markers labeled generic `Chapter {i}`, discarding real chapter titles.
- **#60** `ui.py:110` — `start_button.Show()` in `on_core_finished` is dead (the button is never hidden).
- **#62** `ui.py:289` — `default_backend()` re-invokes `available_backends()` right after the caller already did.
- **#64** `voices.py:164` — Mixed-language blend uses only the first component's lang_code for G2P while averaging cross-language voicepacks.
- **#65** `README.md:26` — Understates the Python ceiling (pyproject is `>=3.10,<3.13`; README says only "Python 3").
- **#66** `backends.py:44` — GPU dropped when `torch.cuda.is_available()` is True but both `torch.version.hip` and `.cuda` are None.
- **#72** `core.py:505` — Stereo 64k AAC for speech: no `-ac 1`, output larger/slower than needed.
- **#76** `ui.py:453` — Cover decode can divide by zero and drops alpha onto a black background.
- **#56** `core.py:375` — `chapter_beginning_one_liner` truncates before stripping → only-whitespace previews.

---

## Cross-module findings (completeness critic)

These are interaction bugs each per-file review missed because each file looks correct in isolation.

1. **[P2] Cross-thread data race on synthesis input** — `ui.py:189` binds the editor's `EVT_TEXT` to
   `setattr(selected_chapter, 'extracted_text', ...)`, and `on_start` passes those *same* chapter objects
   to `CoreThread`, which reads `chapter.extracted_text` on the worker thread (`core.py:159`). The text
   area is **not** disabled during synthesis → typing mid-run yields torn/truncated narration and
   corrupts `total_chars`/ETA. **Fix:** snapshot the text, or `SetEditable(False)` in `on_start`.
2. **[P3] `split_pattern=r'\n\n\n'` is dead code** (`core.py:276,292`) — each `synth()` call already
   receives a single spaCy sentence, so Kokoro's internal chunking never engages. This is the mechanism
   that *locks in* the per-sentence perf problem (#11).
3. **[P3] Unicode/emoji status prints crash legacy Windows consoles** — `voices.py` carefully provides
   `flags_win` and `print_selected_chapters` guards its checkmark, proving the authors know cp1252 can't
   encode emoji — yet `core.py`/`ui.py` emit raw `✅⏳🚀` and an en-dash unconditionally → `UnicodeEncodeError`.
4. **[P3] MLX engine path is untested and dtype-divergent** — `_build_mlx_synth` (`core.py:290`) does
   `np.asarray(seg.audio).reshape(-1)` with no float32 coercion (torch path normalizes via `to_numpy`);
   `test_synth` mocks only `KPipeline`, never the mlx branch.
5. **[P3] `chapters.txt` is never cleaned up** — `create_m4b`'s `finally` unlinks `wav_list`/`cover` but
   not `chapters.txt` (`core.py:515-518`), so it's left in the user's output folder every run.
6. **[P3] Empty `chapter_wav_files` still calls `create_m4b`** → empty concat list + header-only metadata
   → opaque ffmpeg error (compounds #17).

---

## Optimization opportunities (the explicit ask)

Ranked by leverage:

1. **Batch synthesis at the Kokoro layer** (`core.py:315`) — biggest win. Join sentences into
   paragraph-sized `\n\n\n`-separated blocks and pass *one* block per `pipeline()` call; let Kokoro's
   (currently inert) `split_pattern` do the chunking. Amortizes g2p + kernel launches across a paragraph
   instead of per sentence. Large constant-factor speedup, no quality change. Keep the manual spaCy split
   only for the non-english `>MAX_SENTENCE_LENGTH` truncation case.
2. **Disable spaCy NER** (`core.py:297`) — load with `exclude=['ner']` (or `spacy.blank('xx')` + the
   sentencizer you already add). Sentence boundaries are unchanged; the heavy NER component (the
   measured ~163× cost) disappears. Pure win.
3. **Stream audio to disk** (`core.py:181-185`) — open `soundfile.SoundFile(path,'w')` once per chapter
   and `.write(seg)` per segment, dropping each segment. Peak RAM falls from O(whole chapter) to
   O(one segment); enables huge unsplit chapters.
4. **Compute chapter durations at write time** (`core.py:184` / `create_index_file`) — you write the WAVs
   yourself at a known 24000 Hz, so `len(final_audio)/sample_rate` gives sample-accurate durations and
   eliminates N `ffprobe` subprocess spawns *and* the None→0 corruption risk (#29).
5. **`-ac 1` mono speech output** (`core.py:505`) — input is mono 24 kHz; without `-ac 1` AAC may go
   stereo, ~doubling size for no benefit.
6. **Parallelize independent chapters** — chapters write distinct WAVs and are independent; a worker pool
   (or pipelining synth of chapter N+1 while ffmpeg/IO of N runs) gives near-linear CPU speedup / better
   GPU saturation.

---

## Verified safe (no action)

The adversarial pass confirmed these are *handled*, refuting common worries:
- **No XXE** (lxml disables external entities), **no zip-slip**, **no subprocess arg-injection**
  (all list-form, no `shell=True`), **basename path-traversal sanitized**, **FFMETADATA injection escaped**
  (`[`/`]` unescaped but not exploitable), **PIL decompression-bomb** has the default guard. (#44, #55, #78)
- The blend repetition-encoding correctly matches Kokoro's `torch.mean` voicepack semantics. (#46)
- The single-pass ffmpeg rewrite, spaCy caching, `id()`-based dedup, and sequential chapter numbering
  fixes are all correct and complete.
- **Repo is *not* bloated** by binary artifacts — the working-tree `.wav`/`.m4b` files are gitignored;
  the largest committed blob is a ~600KB PNG. (#50)

---

## Recommended remediation sequence

1. **Ship #1 (P0) + #6 + #9 immediately** — small, isolated, high-impact (`-o`, GUI hang, double intro).
2. **Security #10** — whitelist filename sanitization (one-line, defense-in-depth).
3. **Quick-win batch:** #5 (Windows espeak glob), #7 (inf weight), #8 (cuda→rocm), #2 (CI test_voices),
   #20 (disable NER), #15/#16 (CLI validation). All small and independent.
4. **Test backfill:** #4 (`test_m4b.py`) + #22 + #14 (CI installs local code) — close the coverage gap on
   the very fixes that were just shipped.
5. **GUI hardening:** #12, #13, cross-module #1 (text race).
6. **Perf project:** Optimizations 1–4 (batch synth, NER off, streaming, duration-at-write) as a
   measured before/after.
7. **Maintainability (largest churn, last):** split `main()` (#35), introduce logging (#36), delete
   `gen_text()` (#45).
