# Audiblez: Generate  audiobooks from e-books

[![Installing via pip and running](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml)
[![Git clone and run](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/git-clone-and-run.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiblez)
![PyPI - Version](https://img.shields.io/pypi/v/audiblez)

### v4 Now with Graphical interface, GPU acceleration (CUDA / AMD ROCm / Apple Silicon), and many languages!

![Audiblez GUI on MacOSX](./imgs/mac.png)

Audiblez generates `.m4b` audiobooks from regular `.epub` e-books,
using Kokoro's high-quality speech synthesis.

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a recently published text-to-speech model with just 82M params and very natural sounding output.
It's released under Apache licence and it was trained on < 100 hours of audio.
It currently supports these languages: 🇺🇸 🇬🇧 🇪🇸 🇫🇷 🇮🇳 🇮🇹 🇯🇵 🇧🇷 🇨🇳

On a Google Colab's T4 GPU via Cuda, **it takes about 5 minutes to convert "Animal's Farm" by Orwell** (which is about 160,000 characters) to audiobook, at a rate of about 600 characters per second.

On my M2 MacBook Pro, on CPU, it takes about 1 hour, at a rate of about 60 characters per second.


## How to install the Command Line tool

If you have Python 3 on your computer, you can install it with pip.
You also need `espeak-ng` and `ffmpeg` installed on your machine:

```bash
sudo apt install ffmpeg espeak-ng                   # on Ubuntu/Debian 🐧
pip install audiblez
```

```bash
brew install ffmpeg espeak-ng                       # on Mac 🍏
pip install audiblez
pip install "audiblez[mlx]"                          # optional: native Apple Silicon (MLX) engine, fastest on Mac
```

Then you can convert an .epub directly with:

```
audiblez book.epub
```

This uses the default voice `af_heart` (the highest-quality English narrator). Pick a different
voice, a curated blend, or your own mix with `-v` (see [Supported Voices](#supported-voices)):

```
audiblez book.epub -v af_bella                 # a different high-quality voice
audiblez book.epub -v af_warm                  # a curated "house narrator" blend
audiblez book.epub -v 'af_bella:60,af_heart:40' # your own weighted blend
```

It will first create a bunch of `book_chapter_1.wav`, `book_chapter_2.wav`, etc. files in the same directory,
and at the end it will produce a `book.m4b` file with the whole book you can listen with VLC or any
audiobook player.
It will only produce the `.m4b` file if you have `ffmpeg` installed on your machine.

## How to run the GUI

The GUI is a simple graphical interface to use audiblez.
You need some extra dependencies to run the GUI:

```
sudo apt install ffmpeg espeak-ng 
sudo apt install libgtk-3-dev        # just for Ubuntu/Debian 🐧, Windows/Mac don't need this
  
pip install "audiblez[ui]"
```

Then you can run the GUI with:
```
audiblez-ui
```

## How to run on Windows

After many trials, on Windows we recommend to install audiblez in a Python venv:

1. Open a Windows terminal
2. Create anew folder: `mkdir audiblez`
3. Enter the folder: `cd audiblez`
4. Create a venv: `python -m venv venv`
5. Activate the venv: `.\venv\Scripts\Activate.ps1`
6. Install the dependencies: `pip install "audiblez[ui]"`
7. Now you can run `audiblez` or `audiblez-ui`
8. For Cuda support, you need to install Pytorch accordingly: https://pytorch.org/get-started/locally/


## Speed

By default the audio is generated using a normal speed, but you can make it up to twice slower or faster by specifying a speed argument between 0.5 to 2.0:

```
audiblez book.epub -v af_heart -s 1.5
```

## Supported Voices

Use `-v` option to specify the voice to use. Available voices are listed here.
The first letter is the language code and the second is the gender of the speaker e.g. `im_nicola` is an italian male voice.

The default voice is **`af_heart`** — the highest-graded English narrator (Kokoro grades it **A**).
Voices vary a lot in quality (see the grades in `audiblez --help`); the best English ones are
`af_heart` (A), `af_bella` (A-), `bf_emma` (B-, British) and `af_nicole` (B-).

[For hearing samples of Kokoro-82M voices, go here](https://claudio.uk/posts/audiblez-v4.html)

| Language                  | Voices                                                                                                                                                                                                                                     |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🇺🇸 American English     | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| 🇬🇧 British English      | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`                                                                                                                                          |
| 🇪🇸 Spanish              | `ef_dora`, `em_alex`, `em_santa`                                                                                                                                                                                                           |
| 🇫🇷 French               | `ff_siwis`                                                                                                                                                                                                                                 |
| 🇮🇳 Hindi                | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`                                                                                                                                                                                                |
| 🇮🇹 Italian              | `if_sara`, `im_nicola`                                                                                                                                                                                                                     |
| 🇯🇵 Japanese             | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`                                                                                                                                                                         |
| 🇧🇷 Brazilian Portuguese | `pf_dora`, `pm_alex`, `pm_santa`                                                                                                                                                                                                           |
| 🇨🇳 Mandarin Chinese     | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`                                                                                                                                 |

For more details about voice quality, check this document: [Kokoro-82M voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)

### Voice blends ("house narrators")

You can blend voices into a single, consistent narrator. Kokoro averages the voices, so a blend is
still a fixed preset (no voice cloning, no reference clip) and is just as stable over a whole book —
it works on every backend, including the Apple-Silicon MLX engine.

**Curated presets** (use the name directly with `-v`):

| Preset | Blend | Character |
|---|---|---|
| `af_warm` | Heart + Bella | warm, smooth natural female |
| `af_expressive` | Heart + Nicole (2:1) | expressive, intimate female |
| `ab_storyteller` | Heart + Emma | neutral US/UK female storyteller |
| `am_warm` | Michael + Fenrir | warm male |
| `am_deep` | Puck + Onyx (2:1) | rich, low male baritone (~100 Hz) |

**Custom blends** — mix any voices with optional weights:

```
audiblez book.epub -v 'af_bella,af_heart'        # 50/50 blend
audiblez book.epub -v 'af_bella:60,af_heart:40'  # weighted 60/40 blend
audiblez book.epub -v 'af_heart:3,am_michael:1'  # mostly Heart with a touch of Michael
```

In the GUI, the recommended voices and curated blends appear at the top of the Voice dropdown
(each annotated with its quality grade), and you can type a custom blend like `af_bella:60,af_heart:40`
directly into the box. For an in-depth comparison of voices, blends, and alternative TTS models, see
[`docs/tts-naturalness-deep-dive.md`](docs/tts-naturalness-deep-dive.md).

## Choosing a backend (GPU acceleration)

By default audiblez runs on **CPU**. Use `-b/--backend` to pick a faster engine:

| Backend | Flag      | Hardware        | Notes                                                                       |
|---------|-----------|-----------------|-----------------------------------------------------------------------------|
| CPU     | `-b cpu`  | any             | default                                                                     |
| CUDA    | `-b cuda` | NVIDIA GPU      | requires a CUDA build of PyTorch                                            |
| ROCm    | `-b rocm` | AMD GPU         | Linux only; requires a ROCm build of PyTorch                                |
| MPS     | `-b mps`  | Apple Silicon   | PyTorch Metal backend                                                       |
| MLX     | `-b mlx`  | Apple Silicon   | native Apple engine, fastest on Mac — needs `pip install "audiblez[mlx]"`   |
| MOSS    | `-b moss` | GPU (llama.cpp) | higher-quality MOSS-TTS engine; needs the `llama-moss-tts` binary + GGUFs   |

```
audiblez book.epub -v af_heart -b mlx     # Apple Silicon (fastest)
audiblez book.epub -v af_heart -b cuda    # NVIDIA
audiblez book.epub -v af_heart -b rocm    # AMD (Linux)
audiblez book.epub -b moss                # higher-quality MOSS engine (voice picked by the engine)
```

The GUI exposes the same choices as radio buttons — only the backends actually available on your
machine are shown. If you request a backend that isn't available, audiblez warns and falls back to CPU.
`--cuda` is still accepted as a deprecated alias for `-b cuda` (and maps to `-b rocm` on a ROCm build).

### MOSS — the higher-quality engine

**MOSS** (`-b moss`) is a higher-quality narration engine that runs as a resident `llama.cpp`
co-process (AMD via Vulkan). When its binary and model weights are present it is **auto-selected as
the default**; otherwise audiblez falls back to Kokoro. It is **not installed by default** — you need:

- the patched `llama-moss-tts` binary on `PATH` (or set `AUDIBLEZ_MOSS_BIN=/path/to/llama-moss-tts`), and
- the MOSS GGUF weights (backbone + decoder; the encoder is needed only for voice cloning), discovered
  from `~/Projects/moss-work/gguf` or overridden via `AUDIBLEZ_MOSS_BACKBONE` / `_DECODER` / `_ENCODER`.

Run `audiblez --doctor` to confirm MOSS is detected. MOSS ignores the Kokoro `-v` voice (it uses its
own built-in voice unless you clone one) and ignores speed at synth time — `--speed` is applied
afterwards by re-stretching, pitch-preserved.

**Voice cloning (`--clone-ref`).** Narrate the whole book in a voice cloned from a short reference WAV:

```
audiblez book.epub -b moss --clone-ref my_voice.wav
```

The reference clip *is* the voice, so it enters the cache key — two different clips never reuse each
other's audio. Cloning needs the MOSS encoder GGUF present.

**Coarse-chunk mode (`--coarse`, experimental, MOSS only).** Pack consecutive sentences into one
utterance (capped at ~16 s) so MOSS does fewer, larger requests — faster, at the cost of
sentence-level edit/cache granularity. Ignored for non-MOSS backends. Audition the output: verify
the quality by ear before committing a long run.

```
audiblez book.epub -b moss --coarse
```

**In the GUI**, MOSS shows up as an Engine radio (disabled, with the reason, when not fully
installed). Selecting it greys out the Kokoro voice dropdown and reveals a **Clone voice** control:
pick a reference WAV (or leave it for the built-in voice), then **Audition** to hear it — the first
audition shows a "loading model" state while the co-process warms up.

**Apple Silicon** is fully supported via both **MLX** (native, fastest — `pip install "audiblez[mlx]"`)
and **MPS** (PyTorch Metal). On an M-series Mac, MLX narrates several times faster than CPU.

**NVIDIA Cuda example:** [Audiblez on a Google Colab Notebook with Cuda](https://colab.research.google.com/drive/164PQLowogprWQpRjKk33e-8IORAvqXKI?usp=sharing).

## Manually pick chapters to convert

Sometimes you want to manually select which chapters/sections in the e-book to read out loud.
To do so, you can use `--pick` to interactively choose the chapters to convert (without running the GUI).


## Preview, recover & fine-tune

A handful of commands help you trust a long run before (and after) committing the hour:

```bash
audiblez --doctor                       # preflight: ffmpeg, espeak-ng, spaCy, backend (add --deep to synth one word)
audiblez book.epub --trailer            # render trailer.wav: opening lines of each chapter — audition voice + detection
audiblez book.epub --seed-lexicon       # write book.lexicon.json of names/acronyms to respell; applied next run
audiblez book.epub --cache              # cache synthesized sentences and reuse them on re-runs (CLI; opt-in)
audiblez book.epub --merge              # rebuild a playable .m4b from chapters already synthesized (crash recovery)
```

- **`--doctor`** fails fast with a red/green report instead of dying 40 minutes in on a missing dependency.
- **`--trailer`** lets you hear whether chapter detection, voice, and pronunciation are right in ~2 minutes.
- **`--seed-lexicon`** seeds a per-book pronunciation sidecar (`<book>.lexicon.json`); edit the values to fix
  recurring names/acronyms once. The GUI has a "📖 Pronunciations" editor and a "🗣️ Audition voice" button.
- **`--cache`** stores each sentence's audio under `<output>/.audiblez_cache` and reuses it on the next run
  (CLI only; the GUI does not use the cache yet). `--cache-clear` wipes that cache.
- **`--merge`** stitches the chapter `.wav`s that already exist into an `.m4b`, so an interrupted run still yields
  a playable audiobook.

A bad sentence no longer aborts a chapter: synthesis retries, then splices a short silence and logs the failure
to `<chapter>.failed.jsonl`. The ETA is measured from real throughput (not a flat guess) and prints a heartbeat.

## Help page

For all the options available, you can check the help page `audiblez --help`:

```
usage: audiblez [-h] [-v VOICE] [-p] [-s SPEED] [-b {cpu,cuda,rocm,mps,mlx,moss}] [-o FOLDER]
                [--doctor] [--deep] [--merge] [--trailer] [--seed-lexicon] [--cache]
                [--cache-clear] [--tune] [--precision {fp32,fp16,bf16}]
                [--clone-ref WAV] [--coarse]
                [epub_file_path]

positional arguments:
  epub_file_path        Path to the epub file

options:
  -h, --help            show this help message and exit
  -v, --voice VOICE     Narrating voice (default: af_heart). A voice id (e.g. af_heart),
                        a preset blend (af_warm, af_expressive, ab_storyteller, am_warm,
                        am_deep), or a custom blend like 'af_bella:60,af_heart:40'.
  -p, --pick            Interactively select which chapters to read in the audiobook
  -s, --speed SPEED     Set speed from 0.5 to 2.0
  -b, --backend {cpu,cuda,rocm,mps,mlx,moss}
                        Narration backend: cpu, cuda (NVIDIA), rocm (AMD), mps
                        (Apple Silicon), mlx (Apple Silicon native), moss (higher-quality
                        llama.cpp engine). Default: auto-select the best working backend
                        (MOSS when fully installed, else a GPU when usable, else CPU).
  -o, --output FOLDER   Output folder for the audiobook and temporary files
  --doctor              Run preflight checks (ffmpeg, espeak-ng, spaCy, backend) and exit
  --deep                With --doctor: also load the model and synthesize one word (slow)
  --merge               Assemble already-synthesized chapter wavs into an m4b and exit
  --trailer             Render a short trailer.wav sampling the opening of each chapter and exit
  --seed-lexicon        Write a <book>.lexicon.json of candidate names/acronyms to edit, then exit
  --cache               Cache synthesized sentences and reuse them on re-runs (opt-in)
  --cache-clear         Delete the sentence cache under <output>/.audiblez_cache and exit
  --chapters CHAPTERS   Comma-separated list or N-M range of chapter indices to convert (e.g. '1,3,5' or '2-6')
  --chapter-text-dir DIR
                        Directory containing chapter_<i>.txt files that override the extracted chapter text
  --clone-ref WAV       Reference WAV for zero-shot voice cloning (MOSS engine); narrate in that voice
  --coarse              (EXPERIMENTAL, MOSS engine only) Pack sentences into ~16 s chunks for more speed

examples:
  audiblez book.epub -v af_heart           # best-quality default voice
  audiblez book.epub -v af_warm -b mlx     # a curated "house narrator" blend
  audiblez book.epub -v 'af_bella:60,af_heart:40'  # your own weighted blend

to use the GUI, run:
  audiblez-ui
```

## Author

by [Claudio Santini](https://claudio.uk) in 2025, distributed under MIT licence.

Related Article: [Audiblez v4: Generate Audiobooks from E-books](https://claudio.uk/posts/audiblez-v4.html)
