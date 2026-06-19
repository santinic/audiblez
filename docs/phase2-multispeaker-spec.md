# Phase-2 multi-speaker — impl spec (alignment + cast)

Design spec for phase-2 multi-speaker (ADR 0001 demoted it here; ADR 0004 records the approach).
**After MOSS lands.** Grounded against real booknlp (`/home/caio/Projects/booknlp`) + audiblez
seams; the alignment strategy was adversarially confirmed **with an empirical run on real booknlp
output** (2026-06-18). booknlp stays an external, English-only preprocessor — never an audiblez dep.

## Chapter alignment — strategy (d), recorded-boundary bucketing
**The doc called this "unsolved." It is solvable EXACTLY, not heuristically.**

- booknlp emits a **flat whole-book** `.book.txt` (`[speaker] sentence [/]` lines, sorted by
  `sentence_id`, **no chapter boundaries**) + `characters_simple.json` (english_booknlp.py:766-796).
- **The enabler:** `.tokens` carries a per-token char offset, and `.book.txt` ↔ `.tokens` join **1:1
  by `sentence_id`**. So every flat-script line has a recoverable offset range into the fed text.
- **audiblez OWNS the booknlp input:** concatenate the selected chapters' **pre-lexicon**
  `extracted_text` into one string, **record each chapter's `[start_char, end_char)` range** at concat
  time, feed that string to booknlp. Then bucket each booknlp sentence (min token char-offset of its
  `sentence_id`) into the chapter whose recorded range contains it. **Zero fuzzy matching.** Per-chapter
  `.wav`/`.sig`/`.failed.jsonl`/m4b markers are untouched — alignment only *attaches a speaker* to
  each existing synth unit.

### Two non-negotiable constraints (silent-wrongness landmines)
1. **CHARACTER offsets, not bytes.** The `.tokens` columns are *named* `byte_onset`/`byte_offset` but
   hold **character** offsets (spaCy `tok.idx` into the decoded str; pipelines.py:140). Drift was
   *proven* at the first smart-quote (char 426 — `data[446]=='”'` but `data.encode()[446]==b't'`).
   Smart quotes are ubiquitous in EPUBs. **Add a fixture test where a chapter boundary lands right
   after a multibyte char.**
2. **Key on the `sentence_id` VALUE, not the line ordinal.** `sentence_id`s are non-contiguous
   (booknlp skips whitespace-only spaCy sentences). Joining on line position desyncs.
3. **Run alignment on PRE-LEXICON text** — `lexicon.apply_lexicon` can change length and runs
   downstream of chapter assignment.

### Layer 2 — necessary, NOT automatic
Bucketing into chapters is layer 1. **Attaching a speaker to each audiblez synth-unit** is layer 2 and
needs **new plumbing**: audiblez segments sentences independently (its own spaCy + 400-char hard
split), so each synth-unit's char-span must map to the overlapping booknlp sentence via the same
offsets. Today `_split_into_sentences` returns bare strings (core.py:634-645) and `split_long_sentence`
`.strip()`s (drops offsets, core.py:363-376). **Must carry `sent.start_char` through both.** Phase-2
trips here if it declares "alignment solved" after only bucketing.

### Upstream-lossy (accept)
booknlp collapses any sentence with ≠1 detected speaker to **Narrator** (english_booknlp.py:782-785) —
dialogue+tag in one sentence loses the non-narrator voice **before audiblez sees it**. audiblez's
finer split cannot recover it.

### Rejected alternatives
- **Fuzzy-match** — two independently mutated texts, ambiguous short repeats, no stable anchor.
- **Per-chapter booknlp** — loses cross-chapter character identity (coref IDs are per-run,
  english_booknlp.py:899-905); re-merging IDs is as hard as the original problem.
- **User sidecar** — manual, doesn't scale, and strictly redundant once audiblez owns input.

## Cast — `book.cast.json`, MOSS-default / Kokoro-ids-optional
Sidecar next to the epub (mirrors lexicon scoping). **Default per character = MOSS clone**; a Kokoro id
is an **explicit override**.

```json
{
  "version": 1,
  "default_character": "Narrator",
  "fallback": {"engine": "moss", "ref": "refs/narrator.wav"},
  "cast": {
    "Narrator": {"engine": "moss",   "ref": "refs/narrator.wav"},
    "Bob":      {"engine": "kokoro", "voice": "am_adam"}
  }
}
```
- `engine` is the **explicit discriminator** — never sniff path-vs-id. Validate: `ref` exists+readable
  for moss; `voice` parses via `voicelib.parse_voice_spec` for kokoro.
- `default_character` speaks unattributed text: the intro line (core.py:273) and "Chapter N." labels.
- `fallback` = MOSS-unavailable / ref-missing policy (data-driven). A character with no ref and no
  kokoro override resolves to `fallback`.

### Threading the seams
- **Closure contract** grows to `synth(text, speed, voice_spec)` (`build_synthesizer:393` binds one
  voice today). Build **one multiplexing synth** owning an optional Kokoro pipeline + the resident
  MOSS handle; dispatch per-call on `voice_spec`; on MOSS-unavailable/ref-missing → `fallback`, and
  **remember which engine actually ran** (it sets the cache key). No cast = degenerate case (every
  sentence → the one `-v` voice; zero behavior change).
- Thread `voice_spec` through `_synth_one_or_silence:556` and `_synth_batch:572`.
- **Batching is speaker-UNSAFE as-is:** `_synth_batch` joins with `\n\n\n` into one synth call = one
  voice. `pack_sentences` (core.py:487) must break batches on **(char-budget OR speaker change)** →
  speaker-contiguous batches only (erodes the batching win in dialogue-heavy chapters — accept).

### Cache + `.sig`
- Per-sentence `cache_key_for(voice_spec)`: `engine` = **actual engine after fallback** (else a
  MOSS→Kokoro fallback render is served forever as MOSS), `repo_id` = (MOSS model-set id |
  `TORCH_REPO_ID`), voice slot = (`"moss:"+sha256(wav_bytes)` | kokoro id). `make_key` unchanged.
  **No `CACHE_VERSION` bump** — per-sentence voice/engine is already in the key (a re-cast changes the
  voice slot → natural miss), the same logic the CROSS_PROJECT_ANALYSIS multi-speaker verdict used. The
  clone-ref folds into the voice-slot string (`moss:<hash>`), not a new field, so no global invalidation.
- `clone_ref_hash` = WAV **content** hash (a path string wouldn't reflect edits — same principle as
  `lexicon.fingerprint`).
- **`.sig`:** add `cast=<sha256 of canonicalized cast_map, each clone-ref replaced by its WAV-content
  hash>` to `_render_signature:877` — else re-casting a character serves a **stale chapter wav** on
  resume (silent-wrongness). Put a short cast-tag in the `_chapter_wav_name` voice slot when a cast is
  active; update `find_chapter_wavs` in lockstep.

### Lifecycle + scope
- `build_synthesizer` returns `(synth, close)`; `core.main` tears MOSS down in `finally`.
- **v1 = CLI-only** (mirror the cache). Defer: GUI cast editor; per-character speed/precision;
  cross-lang Kokoro casts (KPipeline binds one `lang_code` — v1 warns on mixed `a`/`b`).

## The real product risk (not the schema)
**Attribution.** The whole design is correct-but-inert if the upstream (span → character) attribution
is wrong. cast_map + alignment are solved; *who said this line* is the open quality axis. Also: MOSS
(~9.6GB) + Kokoro co-resident on 16GB fits on paper but is **unverified under interleaved load** — a
hybrid chapter alternating engines keeps both hot; measure **interleaved** RTF, not just steady-state.
