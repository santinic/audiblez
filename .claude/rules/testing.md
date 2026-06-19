# Testing

**Run tests with `unittest`, NOT pytest** (pytest isn't installed). There is no
`test/__init__.py`, so bare `python -m unittest discover -s test` fails with *"Start directory is
not importable"*. Use one of:

- per-file, how CI runs it: `python -m unittest discover -s test -p 'test_cache.py' -v`
- by module: `python -m unittest test.test_cache test.test_lexicon ...`
- the whole suite at once: the `/test` skill.

Test files are **feature-named**, not mirror-named (`test_cache.py`, `test_lexicon.py`,
`test_validation.py`, `test_resilience.py`, `test_eta.py`, `test_batching.py`, `test_trailer.py`,
`test_doctor.py`, `test_backends.py`, `test_gpu.py`, `test_synth.py`, `test_cli.py`,
`test_main.py`, `test_find_chapters.py`, `test_text_utils.py`). **Add new tests to the existing
per-area file.** CI **auto-discovers** every `test/test_*.py` (a loop in
`.github/workflows/git-clone-and-run.yml`, which fails if zero files are found), so a brand-new
test file is picked up automatically — no enumeration to maintain. Two carve-outs the loop applies:
`test_main.py` is the network/integration test (downloads a whole book) and is skipped in the loop
(covered by the dedicated e2e step); keep new *hermetic* tests hermetic so the loop stays fast.

**Hermetic-test pattern** — guard the heavy import, then `skipIf`, and fake the heavy bits so the
logic runs with no model / ffmpeg / network:

```python
try:
    import audiblez.core as core
    _ERR = None
except Exception as e:
    _ERR = e

@unittest.skipIf(_ERR is not None, f"audiblez.core unavailable: {_ERR}")
class FooTest(unittest.TestCase): ...
```

Fake spaCy / the synth closure / `soundfile` / `probe_duration` with `mock.patch.object`. ~10
tests skip locally (heavy stack absent / network fixtures / Mac-only). `test_main.py` runs a real
end-to-end conversion only when an epub fixture is present locally, else it skips.
