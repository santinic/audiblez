# Changelog

## [Unreleased]

### Added — Apple Silicon (Metal / MPS) GPU support

Audiblez now runs the Kokoro TTS model on Apple's Metal GPU on M-series Macs.
The optimization work along the way also produced device-independent speedups
that benefit CPU and CUDA users.

#### New CLI/UI surface

- `audiblez --device {auto,cpu,cuda,mps}` (CLI flag, with `auto` defaulting to
  MPS on Apple Silicon, then CUDA, then CPU).
- `audiblez --precision {fp32,bf16,fp16}` (CLI flag; CUDA-only useful — see
  limitations below).
- New "Metal (MPS)" radio in the GUI's engine selector, plus a precision
  radio group. Radios for unavailable backends are disabled.
- `-c/--cuda` kept as a deprecated alias for `--device cuda`.

#### Performance

Measured on an Apple M-series chip with `voice=af_sky`, full Poe EPUB
(15.8k chars across 3 chapters, source: Project Gutenberg #1064). Chapter 2
is the longest (~13.7k chars) and most representative of steady-state perf.

| Config                                      | ch2 cps | wall (full EPUB) |
| ------------------------------------------- | ------- | ---------------- |
| CPU, no batching (pre-change baseline)      | 161     | 102.2s           |
| MPS, no batching (naive `.to('mps')`)       | 128     | 128.8s ⚠ slower  |
| CPU, batching `max_length=400`              | 186     |  88.1s           |
| MPS, batching `max_length=400`              | 186     |  88.3s           |
| **MPS, batching `max_length=1200` (ship)**  | **228** | **74.0s**        |

Net: ~2.6× speedup on chapter 2 vs the pre-change baseline. MPS is roughly
25-35% faster than CPU at steady state, and the batching alone gives ~30%
to both devices.

#### How MPS was made to work

Naive `.to('mps')` was *slower* than CPU (128 vs 161 cps). Four changes were
needed to get a real speedup:

1. **Don't call `torch.set_default_device('mps')`.** Doing so forces unrelated
   tensors (tokenizer state, numpy bridges) onto the GPU and causes constant
   CPU↔GPU ping-pong on every step. We move the model with `.to('mps')` only.

2. **Build the vocoder with `KModel(disable_complex=True)`.** Kokoro's default
   vocoder uses `torch.stft`/`torch.istft` with complex tensors, whose MPS
   support is poor in torch 2.12 (slow, plus a flood of `_VF.stft` resize
   warnings). Kokoro ships a conv1d-based `CustomSTFT` (originally for ONNX
   export); on MPS this is significantly faster and warning-free. Audio
   quality is indistinguishable in listen tests (`replicate` vs `reflect`
   padding is the only theoretical difference).

3. **Batch short sentences before calling the pipeline.** EPUB extraction
   produces many short "sentences" (titles, list items, paragraph fragments).
   Each pipeline call has fixed per-call overhead (kernel launch, tensor
   allocation), and at small sentence sizes MPS overhead exceeds the math
   savings — short sentences ran ~20% *slower* on MPS than CPU pre-batching.
   `batch_sentences()` packs adjacent sentences up to `max_length=1200` chars
   before invoking Kokoro; the pipeline then re-chunks at its own
   phoneme-aware boundaries (≤510 phonemes per forward pass). This was the
   single biggest perf win and benefits CPU too.

4. **Keep weights in fp32 on MPS.** See limitations.

#### Implementation notes

- New helpers in `audiblez/core.py`: `pick_device()`, `set_device()`,
  `chars_per_sec_estimate()`, `build_pipeline()`, `batch_sentences()`.
- `main()` and `gen_text()` take `device` and `precision` parameters.
- `precision != 'fp32'` casts the model with `.to(dtype=...)` and monkey-patches
  `pipeline.load_voice` so voice tensors loaded from `.pt` files are also cast.
  No effect on CPU (precision argument ignored).
- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` when `device='mps'` is selected. With
  `disable_complex=True` no ops are expected to fall back, but the env var
  costs nothing if unused.

#### Limitations

**bf16 / fp16 do not currently work on MPS (PyTorch 2.12).** Three approaches
were tested and none ship-worthy:

- `model.bfloat16()` and `model.half()` crash with a hard Metal driver
  assertion: `MPSNDArrayMatrixMultiplication.mm: Destination NDArray and
  Accumulator NDArray cannot have different datatype`. The assertion kills
  the process — it can't be caught.
- `torch.autocast(device_type='mps', dtype=bfloat16)` over fp32 weights gets
  further but fails inside Kokoro's `F0Ntrain` because `torch.nn.LSTM` is
  not part of autocast's op-coverage list and raises on dtype mismatch.
- Autocast + a monkey-patch forcing LSTM into `autocast(enabled=False)` runs
  end-to-end but is 26% slower than fp32 (139 vs 189 cps): dtype-boundary
  casts cost more than bf16 saves on this small (82M) sequential workload.

The `--precision` flag is plumbed through anyway because it works on CUDA.
On MPS it defaults to fp32 and the README discourages switching.

**Other pre-existing issues uncovered but not fixed in this change:**

- `core.create_m4b()` invokes `ffmpeg -c:a libfdk_aac`, which is not in
  Homebrew's default ffmpeg build. The WAV files generate correctly but
  the final m4b assembly fails with `FileNotFoundError`. Unrelated to MPS.
- `cli.py` and `ui.py` had `from core import main` (no package prefix),
  which only resolved when run from inside the `audiblez/` directory.
  Fixed to `from audiblez.core import ...` as part of this change.

#### New scripts

Three benchmark scripts (kept in the repo root for reproducibility of the
numbers above):

- `bench_device.py` — CPU vs MPS, isolating the device + STFT-config effect.
- `bench_sentences.py` — diagnoses per-sentence-length perf to identify the
  per-call-overhead bottleneck.
- `bench_batchsize.py` — sweeps `batch_sentences` `max_length` across both
  devices to find the sweet spot.

Each is runnable as `.venv/bin/python bench_*.py` from the repo root after
installing dependencies. Note: bench scripts assume a Poe sample EPUB at
`/tmp/audiblez_smoke/poe.epub` (download from gutenberg.org/ebooks/1064).
