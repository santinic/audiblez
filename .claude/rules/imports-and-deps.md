# Imports & dependencies

**Keep `backends.py`, `doctor.py`, `cache.py`, `lexicon.py`, `gpu.py` import-light.** At module
scope they import only stdlib (and each other). `torch`, `kokoro`, `spacy`, `phonemizer`,
`mlx_audio` are imported LAZILY inside the functions that need them. This is load-bearing:
`audiblez --doctor` must run and diagnose a half-installed/broken TTS stack, so the preflight
path cannot trigger a heavy import. Do not hoist a heavy import to module scope.

**Do not "remove unused" `lxml`, `misaki`, or `phonemizer`** — they're used indirectly:
- `lxml` via `BeautifulSoup(features='lxml')`
- `misaki[zh]` is Kokoro's Chinese g2p backend
- `phonemizer` via a guarded import in `core.set_espeak_library`

`deptry` is configured to ignore exactly these (`pyproject.toml [tool.deptry.per_rule_ignores]`).
Run the dependency check with `deptry .`.

**Optional dependency groups:** `[ui]` (wxPython + Pillow) and `[mlx]` (mlx-audio, Apple Silicon
only). Core synthesis must not depend on either.

**Linting:** `ruff check .` (config in `pyproject.toml [tool.ruff]`; CI gates on it). The rule set is
deliberately focused — `E,F,W,B` minus a few stylistic codes the codebase uses on purpose
(one-line `if x: y`, long lines, `lambda` assignment). Don't widen `select` to import-sorting (`I`) or
`pyupgrade` (`UP`) — that just churns the upstream-fork code. `ruff format` is NOT enforced (it would
reformat the whole tree); leave formatting alone unless asked.
