# voice_dataset — podcast → clean single-speaker TTS dataset

A one-time, offline pipeline that turns **many hours of podcast audio** (with intro/outro music,
ads, and occasional guests) into a **clean single-speaker dataset** ready to fine-tune a custom
narrator voice. It is the data-prep stage of the cloning plan in
[`docs/voice-cloning-rocm-plan.md`](../../docs/voice-cloning-rocm-plan.md).

The output is a *superset* dataset that feeds **StyleTTS2 / Stylish-TTS** (the recommended model),
**Piper**, or **Coqui XTTS** unchanged — build it once, train any of them.

This tool is **independent of audiblez** (its own deps, its own venv). It runs anywhere torch runs:
on your Mac now (CPU), and on the Arch + RX 7800 XT (ROCm) box later — the same command, the device
is picked automatically.

---

## What it does

```
podcast files ──▶ ffmpeg → 24 kHz mono  ──▶ WhisperX (transcribe → word-align → diarize → assign)
            (optional Demucs vocal-isolation)        │
                                                     ▼
   keep ONLY the narrator  ◀── pick narrator (reference-clip cosine match, or dominant speaker)
                 │
                 ▼
   segment on sentence boundaries (1–15 s, never mid-word, break on gaps)
                 │
                 ▼
   loudness-normalize each clip ──▶ wavs/ + metadata.csv + train/val_list.txt + metadata_xtts.csv
```

Key behaviours (and why):
- **Single speaker, reliably.** With `--reference narrator.wav` it embeds each diarized speaker
  (SpeechBrain ECAPA) and keeps only the one matching your reference — robust across episodes with
  rotating guests. Without a reference it falls back to the dominant speaker per file.
- **No glued-together audio.** Clips break on a time gap (`--max-gap`) so a guest's removed turn never
  joins two non-contiguous narrator snippets into one clip.
- **Terminal punctuation enforced.** Clips that don't end in `. ? !` are dropped/fixed — without it a
  TTS model never learns to stop and rambles.
- **Loudness-normalized** to −23 LUFS with a peak ceiling (avoids the classic `pyloudnorm` clip-on-write).
- **Resumable & idempotent.** Processed files are recorded in `state.json`; clips append to
  `records.jsonl`; manifests regenerate every run. Re-run anytime to add more episodes.

---

## Prerequisites

1. **ffmpeg** on PATH — `pacman -S ffmpeg` (Arch) / `brew install ffmpeg` (Mac).
2. **A HuggingFace token** for diarization, and you must **accept the model conditions** while logged
   in to HF:
   - https://huggingface.co/pyannote/speaker-diarization-3.1  → "Agree"
   - https://huggingface.co/pyannote/segmentation-3.0  → "Agree"
   - Create a read token at https://huggingface.co/settings/tokens → export `HF_TOKEN=hf_...`
3. **A reference clip** (recommended): 10–30 s of clean speech of just your narrator, any format.

---

## Install

Use a **separate venv** (these deps are heavy and unrelated to audiblez):

```bash
python3.12 -m venv .venv-dataset
source .venv-dataset/bin/activate

# torch FIRST, matching your hardware:
#   Arch + RX 7800 XT (ROCm):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
#   Mac / CPU:
pip install torch torchaudio

pip install -r tools/voice_dataset/requirements.txt
```

> On Arch/ROCm, also confirm `python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"`
> prints `True <hipversion>` (ROCm exposes the GPU through the CUDA API). On Mac this prints `False` and
> the pipeline runs on CPU — slower, but it's a one-time build.

---

## Usage

**1. Smoke-test on a couple of episodes first** (verifies your token, models, and GPU before a long run):

```bash
export HF_TOKEN=hf_xxx
python tools/voice_dataset/build_dataset.py /path/to/podcast/ \
    --reference narrator.wav --out dataset --limit 2
```

Listen to a few `dataset/wavs/*.wav` and open `dataset/metadata.csv` — confirm it's your narrator,
clean, and the transcripts match.

**2. Full run** (resumes where the smoke test stopped; safe to re-run as you download more):

```bash
python tools/voice_dataset/build_dataset.py /path/to/podcast/ \
    --reference narrator.wav --out dataset
```

**3. Music-heavy episodes** — add Demucs vocal isolation (slower; only where music sits under speech):

```bash
python tools/voice_dataset/build_dataset.py /path/to/episode_with_music.mp3 \
    --reference narrator.wav --out dataset --demucs
```

**No reference clip** (solo podcast where the host always dominates): omit `--reference` and it keeps
the most-spoken speaker per file.

Run `python tools/voice_dataset/build_dataset.py --help` for all options.

---

## Calibrating `--sim-threshold`

The reference matcher keeps a speaker only if its ECAPA cosine similarity to your reference is
≥ `--sim-threshold` (default `0.50`). If you're getting guest speech leaking in, raise it (0.6–0.75);
if your narrator is being dropped, lower it. The per-file log prints the matched speaker and its
cosine score — use those numbers to pick a clean cut-off.

---

## Output

```
dataset/
  wavs/                 24 kHz mono PCM-16 clips, 1–15 s, loudness-normalized
  metadata.csv          LJSpeech:  id|raw|normalized            (Piper, Coqui ljspeech)
  train_list.txt        StyleTTS2: id.wav|text|speaker_id       (+ val_list.txt)
  val_list.txt
  metadata_xtts.csv     Coqui XTTS: audio_file|text|speaker_name (with header)
  records.jsonl         append-only clip log (drives resume + manifest regen)
  state.json            processed source files (resume marker)
```

**How many hours?** For *fine-tuning*, clip quality beats raw hours — a few clean hours is plenty for
StyleTTS2/Piper. Note **Stylish-TTS specifically wants ≥ 25 h** of clean pairs, so if you go that
route, check `records.jsonl` total after cleaning. The script prints total hours when it finishes.

---

## Next step

Hand the dataset to the trainer per the plan
([`docs/voice-cloning-rocm-plan.md`](../../docs/voice-cloning-rocm-plan.md) §"End-to-end plan"):
StyleTTS2/Stylish-TTS for best fidelity, or Piper first as the low-risk baseline. Then the fine-tuned
voice is wired back into audiblez through the `build_synthesizer` engine seam.

## Caveats

- **Demucs degrades clean speech** — only use `--demucs` where music actually overlaps dialogue.
- **Whisper can hallucinate text over music/silence.** The diarization + reference-match + terminal-
  punctuation + duration filters remove most of it, but spot-check `metadata.csv`.
- **Personal use only.** Cloning a real person's voice is fine for your own listening; don't
  redistribute the model, the clips, or the generated audio. See the plan's "Risks & caveats".
