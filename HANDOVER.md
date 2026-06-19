# audiblez — Handover note (2026-06-18)

Pick-up doc for continuing the "next wave" work + GPU setup. Read this, then
`task_plan.md` / `findings.md` / `progress.md` for the deeper record.

## TL;DR

- Branch **`feature/next-wave`** — all 8 planned phases implemented, tested, committed
  (10 commits ahead of `main`). See `task_plan.md` (every phase `Status: complete`).
- The heavy TTS stack is now installed in a **`.venv`** (uv, Python 3.12) with a
  **ROCm** torch build, so the full suite and real synthesis run locally on the AMD GPU.
- ⚠️ **There are 5 uncommitted edits** (GPU default + bug fixes + test guards). They are
  safe but were made after the last commit — **commit them first when you resume** (see
  "Uncommitted work" below). They may already be committed if the resume step ran.

## Environment (already set up)

- `.venv/` — created by `uv venv --python 3.12` (Python 3.12.13). Not in git (gitignored).
- `torch==2.12.1+rocm7.2` (HIP 7.2), installed via
  `uv pip install --torch-backend=rocm7.2 --reinstall-package torch torch`.
  Replaced the auto-installed CUDA wheel (this box has no NVIDIA GPU).
- spaCy models installed: `xx_ent_wiki_sm` (used by audiblez) + `en_core_web_sm`.
- System: **AMD Radeon RX 7800 XT** (Navi 32, gfx1101), ROCm 7.2.4 at `/opt/rocm`,
  espeak-ng at `/usr/lib/libespeak-ng.so`, ffmpeg + ffprobe on PATH.
- `torch.cuda.is_available()` → **True** (ROCm rides the cuda API);
  `backends.available_backends()` → `['cpu', 'rocm']`; `default_backend()` → `rocm`.

Activate with: `source .venv/bin/activate` (or call `.venv/bin/python` directly).

## GPU configuration (done)

- **CLI now auto-selects the best backend.** A bare `audiblez book.epub` picks `rocm`
  here (was hardcoded to `cpu`). Force CPU with `-b cpu`. This is the `cli.py` edit below.
- Verified working: `audiblez --doctor --deep -b rocm` → "deep: synth one word: produced
  audio", all checks green. Real book synth confirmed running on GPU (`gpu.log`).
- **Benign noise** during GPU synth (ignore): `(null): No such file or directory`,
  `MIOpen(HIP) Warning [IsEnoughWorkspace]`, AOTRITON "experimental attention" warnings.
- **First-run is slow**: MIOpen compiles gfx1101 kernels on first use (chapter 1 of poe
  measured 43s / ~11 cps purely from kernel compilation). Steady-state throughput is the
  *second* chapter's number — re-measure after kernels are cached (`~/.cache/miopen`).
- Optional speedup to try: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables flash/mem
  efficient attention on RDNA3 (currently flagged experimental and disabled).

## Uncommitted work — COMMIT THIS FIRST

`git status` shows these modified (all safe; verified by reasoning + the e2e that found them):

| File | Change | Why |
|---|---|---|
| `audiblez/cli.py` | bare invocation → `backends.default_backend()` (auto GPU); create output dir early before `--merge`/`--seed-lexicon`/`--trailer` run | "use the GPU"; those subcommands wrote into `-o <dir>` before `main()` (which used to be what created it) → crashed on a missing dir |
| `audiblez/core.py` | `create_m4b` writes **absolute** paths into the ffmpeg concat list | **Real bug**: ffmpeg's concat demuxer resolves relative entries against the *list file's* dir, so a relative `-o out/` doubled to `out/out/chapter.wav` and failed. Affected any relative `-o`, not just `--merge`. |
| `audiblez/lexicon.py` | `save_lexicon` mkdirs parent | `--seed-lexicon -o newdir` and the GUI editor must not crash on a missing dir |
| `test/test_find_chapters.py` | skip when the `../epub/*.epub` fixture is absent | turns 7 spurious collection ERRORs into SKIPs (fixtures aren't in the repo) |
| `test/test_main.py` | skip when download fails / fixture absent; `ffplay` behind `AUDIBLEZ_TEST_PLAY` env | makes the networked e2e tests not ERROR offline and not block on interactive playback |

Suggested commit (two commits is cleaner):
```
git add audiblez/cli.py audiblez/core.py audiblez/lexicon.py
git commit -m "fix: auto-select GPU backend; absolute concat paths; create output dir for subcommands"
git add test/test_find_chapters.py test/test_main.py
git commit -m "test: skip fixture/network-dependent integration tests when assets absent"
```

## Outstanding TODO (resume here)

1. ✅ **DONE — committed** (commits `38080ac` + `8d51aa8` on the branch).
2. ⏳ **GPU run throughput** — see "GPU default decision" below; the real chapter-2 cps is
   the open number. (`test/gpu.log`, task `bduqgfam0`.)
3. ✅ **DONE — `create_m4b` fix VERIFIED**: `audiblez poe.epub --merge -o test/e2e_out`
   produced `test/e2e_out/poe.m4b` (8.2M), no "Impossible to open". The committed fix works.
4. **Run the full suite on the real stack**: `.venv/bin/python -m unittest discover -s test`
   — 8 new + 4 existing hermetic modules PASS; test_find_chapters SKIPs (verified:
   `OK (skipped=7)`); test_main SKIPs without fixtures/network. Target: 0 errors.
5. **GPU default decision — REVISIT (see below).**
6. Decide whether to **push / open a PR** (nothing pushed yet) and whether to commit the
   planning files (`task_plan.md`, `findings.md`, `progress.md`, this `HANDOVER.md`) —
   currently untracked working memory.

## ⚠️ GPU default decision — needs the throughput number

I flipped the CLI default from `cpu` (the repo had it hardcoded with a deliberate
"unchanged default behavior" comment) to auto-select the GPU, per "use the GPU". **But on
this box ROCm may be SLOWER than CPU:**
- CPU baseline (measured this session): **~236 chars/sec**.
- GPU chapter 1: 11 cps (pure MIOpen kernel-compile warm-up — ignore).
- GPU chapter 2 (real synth): the live ETA was *climbing* as progress rose (rate falling) —
  MIOpen is on its fallback path and AOTRITON flash/mem-efficient attention is DISABLED on
  gfx1101. **Get the definitive "Chapter 2 read in … (… cps)" line from `test/gpu.log`.**
- If GPU < CPU: either (a) revert the default to `cpu` and keep GPU opt-in via `-b rocm`,
  or (b) try `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` and re-measure before deciding.
  Note: the GUI already defaulted to the best backend before this change; only the CLI
  default changed.

## Verification status

- 91 hermetic tests pass on the real stack (`.venv/bin/python -m unittest <module>`).
- Running the suite caught two real defects, both fixed: a too-short-text trailer test
  (committed), and the `create_m4b` relative-path doubling (uncommitted, item 1 above).
- New-feature e2e on real `poe.epub` (CPU + GPU): `--doctor`, `--seed-lexicon` (20 terms),
  `--trailer` (1020K wav, 2 chapters), `--cache` (41% intra-run hit from repeated
  boilerplate), `--merge`. Full epub→m4b produced `poe.m4b`.

## Handy commands

```bash
PY=.venv/bin/python
$PY -m audiblez.cli --doctor                 # preflight (all green here)
$PY -m audiblez.cli --doctor --deep -b rocm  # GPU smoke: synth one word
$PY -m audiblez.cli book.epub                # auto-uses the GPU now
$PY -m audiblez.cli book.epub -b cpu         # force CPU
$PY -m unittest discover -s test             # full suite (real stack)
```

## Test artifacts to clean (all gitignored)

`test/poe.epub`, `test/poe.m4b`, `test/gpu_out2/`, `test/e2e_out/`, `test/gpu.log`,
`test/*.wav` — safe to delete; regenerated by the e2e/tests.
