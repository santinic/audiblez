---
name: test
description: Run the audiblez test suite (Python unittest). Use whenever asked to run tests, check that tests pass, or verify a change didn't regress. Handles the non-obvious invocation — pytest is NOT installed and bare `unittest discover` fails here.
---

Run the audiblez test suite. Notes that matter:
- **pytest is NOT installed** — use `unittest`.
- There is no `test/__init__.py`, so `python -m unittest discover -s test` fails ("Start directory
  is not importable"). Run by module list, or per-file with `-p`.
- Expect **~10 skips** locally (heavy TTS stack absent / network fixtures / Mac-only). A clean run
  ends with `OK (skipped=N)`. Report any FAIL/ERROR with its traceback.

**If `$ARGUMENTS` is given**, run only those modules (map each word to `test.test_<word>`), e.g.
`/test cache lexicon` →

```bash
python -m unittest test.test_cache test.test_lexicon
```

**Otherwise run the whole suite** (auto-discovers every `test/test_*.py`):

```bash
python -m unittest $(ls test/test_*.py | sed 's#test/##;s#\.py$##;s#^#test.#' | tr '\n' ' ')
```

To reproduce CI exactly (per-file discover, hermetic files only), see the list in
`.github/workflows/git-clone-and-run.yml`.

New tests belong in the existing per-area test file — see `.claude/rules/testing.md`.
