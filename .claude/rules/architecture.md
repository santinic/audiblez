# Architecture — synth pipeline & entry points

audiblez converts EPUB → M4B audiobooks with the Kokoro-82M TTS model. The pipeline is
`EPUB → chapters → per-sentence synth (batched) → chapter .wav → single ffmpeg pass → .m4b`.

**The engine seam:** `core.build_synthesizer(voice, backend, precision)` returns a closure
`synth(text, speed) -> list[np.ndarray] @ 24 kHz`, dispatched on `backends.BackendInfo.engine`
(`torch` / `mlx`). Adding a backend = one dispatch branch + one lazy-import closure (the mlx
branch is the template).

**Three entry points reach synthesis — a change to a `core` function signature or the synth
path must update ALL of them, and the GUI is the one that gets missed:**

- `cli.py` (`cli_main`) → `core.main` / `make_trailer` / `merge_chapters`
- `ui.py` (wxPython GUI) → `CoreThread` runs `core.main`; preview/audition/trailer go through
  `_synth_to_temp` / `make_trailer`
- `core.py` functions themselves

Before finishing a `core` signature change, grep the symbol across `cli.py`, `ui.py`, `core.py`.
(Real regression: `make_trailer` gained `output_folder=`, wired in `cli.py` but missed in
`ui.py`, so the GUI trailer read the lexicon from the wrong folder.)
