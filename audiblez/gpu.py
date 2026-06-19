# -*- coding: utf-8 -*-
"""GPU performance tuning for the torch backends (cuda / rocm).

Every knob here is opt-in and a strict no-op on cpu/mlx, so importing this module
and calling its functions is always safe. It is import-light (stdlib +
:mod:`audiblez.backends`); ``torch`` is imported lazily inside the functions that
need it, so ``import audiblez.gpu`` never pulls in the heavy stack.

The design target is audiblez's actual workload: **repeated batch-size-1 Kokoro
inference**. Kokoro's ``KModel.forward`` hardcodes a batch dimension of 1 and
``KPipeline.__call__`` iterates text segments in a Python loop, so the GPU runs the
same small set of GEMM shapes thousands of times. Two levers fit that shape:

* **TunableOp** (:func:`configure_tunableop`) — auto-benchmarks each recurring GEMM
  (rocBLAS vs hipBLASLt) once and persists the winner to a CSV; the one-time tuning
  cost amortizes across the whole book. Env-only, zero numerical risk.
* **Autocast** (:func:`autocast_context`) — runs eligible matmuls in fp16/bf16 for
  raw throughput. This *changes the waveform*, so it is flag-gated and should be
  auditioned (see :func:`waveform_similarity`).
"""
import os
import contextlib

from audiblez import backends

PRECISIONS = ('fp32', 'fp16', 'bf16')

# Autocast/TunableOp use the CUDA device type, which ROCm also exposes. MPS autocast
# coverage is uneven, so we deliberately keep Apple GPUs on fp32 (nullcontext).
_AUTOCAST_TORCH_DEVICES = ('cuda',)


def _is_torch_gpu(backend: str) -> bool:
    """True only for a GPU-accelerated *torch* backend (cuda/rocm), not mlx/cpu."""
    info = backends.BACKENDS.get(backend)
    return info is not None and info.engine == 'torch' and backends.is_gpu(backend)


def configure_tunableop(enabled: bool, backend: str, results_dir: str | None = None,
                        out=print) -> bool:
    """Enable PyTorch TunableOp GEMM autotuning via environment variables.

    Must run **before the first GEMM** (TunableOp caches each env var on first read);
    callers set this up before building the synthesizer. Returns True iff tuning was
    actually enabled. No-op (returns False) unless ``enabled`` and ``backend`` is a
    GPU torch backend. Existing env values are never overwritten (``setdefault``), so
    a power user can still tune the knobs by hand.
    """
    if not enabled or not _is_torch_gpu(backend):
        return False
    os.environ.setdefault('PYTORCH_TUNABLEOP_ENABLED', '1')
    os.environ.setdefault('PYTORCH_TUNABLEOP_TUNING', '1')
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        # torch inserts the device ordinal before the extension (…_results0.csv).
        os.environ.setdefault('PYTORCH_TUNABLEOP_FILENAME',
                              os.path.join(results_dir, 'tunableop_results.csv'))
    out('GPU GEMM autotuning (TunableOp) enabled — the FIRST run is slower while it '
        'tunes; results are cached to disk and reused on later runs.')
    return True


def autocast_context(backend: str, precision: str = 'fp32'):
    """Return a ``torch.autocast`` context manager for fp16/bf16, else ``nullcontext``.

    fp16/bf16 only engage on a GPU torch backend (cuda/rocm); everything else —
    fp32, cpu, mlx, mps — gets a no-op nullcontext, so wrapping a synth call in this
    is always safe. Autocast keeps numerically-sensitive ops (notably FFT/STFT, which
    Kokoro's vocoder relies on) in fp32 automatically.
    """
    if precision not in PRECISIONS:
        raise ValueError(f'Unknown precision {precision!r}; choose from {PRECISIONS}')
    if precision == 'fp32' or not _is_torch_gpu(backend):
        return contextlib.nullcontext()
    info = backends.BACKENDS[backend]
    if info.torch_device not in _AUTOCAST_TORCH_DEVICES:
        return contextlib.nullcontext()  # e.g. mps: skip, autocast coverage is uneven
    import torch
    dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
    return torch.autocast(device_type='cuda', dtype=dtype)


def waveform_similarity(a, b) -> float:
    """Normalised cross-correlation of two 1-D waveforms in ``[-1, 1]`` (1.0 = identical).

    A lightweight A/B metric for auditioning a lower-precision run against an fp32
    reference: synth the same text both ways and compare. Length-tolerant (compares
    the overlapping prefix, since precision can shift predicted durations slightly).
    Pure numpy — runnable without the TTS stack.
    """
    import numpy as np
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.sum(a * b) / denom)
