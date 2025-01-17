# Audiblez: Generate  audiobooks from e-books
[![Installing via pip and running](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml/badge.svg)](https://github.com/santinic/audiblez/actions/workflows/pip-install.yaml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/audiblez)
![PyPI - Version](https://img.shields.io/pypi/v/audiblez)

Audiblez generates `.m4b` audiobooks from regular `.epub` e-books, 
using Kokoro's high-quality speech synthesis.

[Kokoro v0.19](https://huggingface.co/hexgrad/Kokoro-82M) is a recently published text-to-speech model with just 82M params and very natural sounding output.
It's released under Apache licence and it was trained on < 100 hours of audio.
It currently supports American, British English, French, Korean, Japanese and Mandarin, and a bunch of very good voices.

On my M2 MacBook Pro, **it takes about 2 hours to convert to mp3 the Selfish Gene by Richard Dawkins**, which is about 100,000 words (or 600,000 characters),
at a rate of about 80 characters per second.

## How to install and run

If you have Python 3 on your computer, you can install it with pip.
Be aware that it won't work with Python 3.13.
Then you also need to download a couple of additional files in the same folder, which are about ~360MB:

```bash
pip install audiblez
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json
```

Then, to convert an epub file into an audiobook, just run:

```bash
audiblez book.epub -l en-gb -v af_sky
```

It will first create a bunch of `book_chapter_1.wav`, `book_chapter_2.wav`, etc. files in the same directory,
and at the end it will produce a `book.m4b` file with the whole book you can listen with VLC or any
audiobook player.
It will only produce the `.m4b` file if you have `ffmpeg` installed on your machine.

## Supported Languages
Use `-l` option to specify the language, available language codes are:
🇺🇸 `en-us`, 🇬🇧 `en-gb`, 🇫🇷 `fr-fr`, 🇯🇵 `ja`, 🇰🇷 `kr` and 🇨🇳 `cmn`.

## Speed
By default the audio is generated using a normal speed, but you can make it up to twice slower or faster by specifying a speed argument between 0.5 to 2.0:

```bash
audiblez book.epub -l en-gb -v af_sky -s 1.5
```

## Supported Voices
Use `-v` option to specify the voice:
available voices are `af`, `af_bella`, `af_nicole`, `af_sarah`, `af_sky`, `am_adam`, `am_michael`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`.
You can try them here: [https://huggingface.co/spaces/hexgrad/Kokoro-TTS](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)


## How to run on GPU
By default audiblez runs on CPU. If you want to use a GPU for faster performance, install the GPU-enabled ONNX Runtime and specify a runtime provider with the `--providers` flag. By default, the CPU-enabled ONNX Runtime is installed. The GPU runtime must be installed manually.

```bash
pip install onnxruntime-gpu
```

To specify ONNX providers, such as using an NVIDIA GPU, use the `--providers` tag. For example:

```bash
audiblez book.epub -l en-gb -v af_sky --providers CUDAExecutionProvider
```

To see the list of available providers on your system, run the following:

```bash
audiblez --help
```

or

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

This will display the ONNX providers that can be used, such as `CUDAExecutionProvider` for NVIDIA GPUs or `CPUExecutionProvider` for CPU-only execution.

You can specify a provider hierarchy by providing multiple hierarchies separated by spaces.

```bash
audiblez book.epub -l en-gb -v af_sky --providers CUDAExecutionProvider CPUExecutionProvider
```

## Author
by [Claudio Santini](https://claudio.uk) in 2025, distributed under MIT licence.

Related article: [Convert E-books into audiobooks with Kokoro](https://claudio.uk/posts/epub-to-audiobook.html)
