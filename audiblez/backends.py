# -*- coding: utf-8 -*-
"""Narration backend registry and capability detection.

Pure data + detection only: this module imports ``torch`` lazily (inside functions)
and never imports ``kokoro``, ``mlx_audio``, or ``audiblez.core``. That keeps it safe
to import from ``cli.py`` and ``ui.py`` without pulling in the heavy TTS stack — the
actual engine construction lives in :func:`audiblez.core.build_synthesizer`.
"""
import os
import shutil
import hashlib
import platform
import importlib.util
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendInfo:
    id: str                    # 'cpu' | 'cuda' | 'rocm' | 'mps' | 'mlx' | 'moss'
    label: str                 # human-readable label for the CLI/GUI
    engine: str                # 'torch' | 'mlx' | 'llamacpp' — selects the generator in core.build_synthesizer
    torch_device: str | None   # 'cpu'|'cuda'|'mps' for torch backends; None for the mlx/llamacpp engines


# Declaration order is also the GUI/menu display order.
BACKENDS = {
    'cpu':  BackendInfo('cpu',  'CPU',                 'torch',    'cpu'),
    'cuda': BackendInfo('cuda', 'CUDA (NVIDIA GPU)',   'torch',    'cuda'),
    'rocm': BackendInfo('rocm', 'ROCm (AMD GPU)',      'torch',    'cuda'),  # ROCm exposes AMD GPUs via the torch.cuda API
    'mps':  BackendInfo('mps',  'MPS (Apple Silicon)', 'torch',    'mps'),
    'mlx':  BackendInfo('mlx',  'MLX (Apple Silicon)', 'mlx',      None),
    'moss': BackendInfo('moss', 'MOSS (llama.cpp)',    'llamacpp', None),  # resident pipe co-process; AMD via Vulkan
}
BACKEND_IDS = tuple(BACKENDS)  # ('cpu', 'cuda', 'rocm', 'mps', 'mlx', 'moss') — argparse choices


# --- MOSS (llamacpp engine) discovery -------------------------------------------------
#
# MOSS runs as a resident `llama-moss-tts --serve` co-process (ADR 0002) over THREE GGUF
# weights. doctor.py (preflight) and core.py (the spawner) MUST agree on where those live,
# so the single source of truth is here in this import-light module. Paths come from env
# vars (so a user can point at their own build) and fall back to the spike's locations
# (tools/moss_serve_spike.py). The encoder is OPTIONAL — needed only for voice cloning —
# so it is NOT a presence requirement.
_MOSS_BIN_NAME = 'llama-moss-tts'  # NOT 'llama-cli' — the patched --serve binary
_MOSS_GGUF_DEFAULT_DIR = Path.home() / 'Projects' / 'moss-work' / 'gguf'
_MOSS_DEFAULTS = {
    'backbone': _MOSS_GGUF_DEFAULT_DIR / 'moss_delay_firstclass_Q5_K_M.gguf',
    'decoder':  _MOSS_GGUF_DEFAULT_DIR / 'moss_tts_audio_decoder_f16.gguf',
    'encoder':  _MOSS_GGUF_DEFAULT_DIR / 'moss_tts_audio_encoder_f16.gguf',
}
MOSS_REQUIRED = ('backbone', 'decoder')  # encoder is clone-only -> warn, never required


def moss_paths() -> dict:
    """Resolve the MOSS binary + the three GGUF paths (env override → spike defaults).

    Returns a dict with keys ``binary``/``backbone``/``decoder``/``encoder``; each value is
    a :class:`pathlib.Path` (the configured/default location) or ``None`` when not found.
    Pure path resolution — never spawns the binary or loads a model (that is a runtime /
    ``--doctor --deep`` concern). Both :func:`audiblez.doctor.check_moss` and
    ``core.build_synthesizer`` resolve MOSS through here so preflight checks exactly what
    the run will spawn.
    """
    bin_env = os.environ.get('AUDIBLEZ_MOSS_BIN')
    if bin_env:
        binary = Path(bin_env) if Path(bin_env).exists() else None
    else:
        found = shutil.which(_MOSS_BIN_NAME)
        binary = Path(found) if found else None
    out = {'binary': binary}
    for kind, env in (('backbone', 'AUDIBLEZ_MOSS_BACKBONE'),
                      ('decoder', 'AUDIBLEZ_MOSS_DECODER'),
                      ('encoder', 'AUDIBLEZ_MOSS_ENCODER')):
        p = Path(os.environ[env]) if os.environ.get(env) else _MOSS_DEFAULTS[kind]
        out[kind] = p if p.exists() else None
    return out


def _gguf_identity(path) -> str:
    """Cheap, stable identity for one GGUF: ``basename:size`` (basename encodes the quant,
    e.g. ``…Q5_K_M``; size guards a same-named re-quantize). Never content-hashes the 8 GB
    weights. Missing files identify as ``basename:absent`` so a swapped-in-then-out file
    still changes the key.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        size = 'absent'
    return f'{p.name}:{size}'


def moss_repo_id() -> str:
    """Stable ``repo_id`` for the MOSS cache key: hashes ALL THREE GGUF identities.

    Each GGUF (backbone/decoder/encoder) independently shapes the waveform, so a change to
    any one must invalidate cached audio. ``core`` passes this string straight into
    :func:`audiblez.cache.make_key` as ``repo_id`` (the MOSS analogue of Kokoro's HF id).
    The encoder is included even when absent (its identity becomes ``…:absent``) so enabling
    cloning later flips the key.
    """
    paths = moss_paths()
    parts = [_gguf_identity(paths[k] or _MOSS_DEFAULTS[k]) for k in ('backbone', 'decoder', 'encoder')]
    digest = hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()[:16]
    return f'moss-gguf:{digest}'


def clone_voice_id(wav_path) -> str:
    """Content-addressed ``voice`` value for a MOSS voice-clone reference WAV.

    For cloning, the reference WAV **is** the voice, so it must enter the cache key — two
    different reference clips synthesizing the same sentence produce different audio and must
    not collide on a constant ``voice='clone'`` (that would serve stale wrong-voice audio on a
    re-run — the silent-wrongness class :doc:`cache-and-resilience` bans). ``core`` passes the
    result straight in as :func:`audiblez.cache.make_key`'s ``voice`` (reusing the existing
    voice axis — no new key field). Hashes the file **contents** (not name/mtime), in chunks so
    a large reference never loads fully into memory; a clip's bytes are tiny vs. the 8 GB GGUFs,
    so a content hash is cheap and exact here (unlike :func:`moss_repo_id`).

    Raises :class:`FileNotFoundError`/:class:`OSError` if the reference is unreadable — a clone
    run with no usable reference must fail loud, never silently fall back to a default voice.
    """
    h = hashlib.sha256()
    with open(wav_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):  # 1 MiB chunks
            h.update(chunk)
    return f'clone:{h.hexdigest()[:8]}'


# The six MOSS sampling parameters, PINNED to fixed defaults (spec "REQUEST" line) so the
# sentence cache actually hits across runs. This is the SINGLE source: core aliases it as
# `core.MOSS_SAMPLING` (used in the synth request AND, hashed, in the cache key), so the two
# can never drift to different values. Changing any value here changes the waveform.
MOSS_SAMPLING_DEFAULTS = {
    'text_temperature': 1.5,
    'text_top_k': 50,
    'audio_temperature': 1.7,
    'audio_top_p': 0.8,
    'audio_top_k': 25,
    'audio_repetition_penalty': 1.0,
}


def moss_sampling_sig(sampling: dict | None = None) -> str:
    """Canonical value-level fingerprint of the MOSS sampling params (human-readable form).

    A deterministic, sorted, value-typed ``k=v;`` rendering of the pinned sampling dict, exposed
    for doctor/diagnostics. The cache KEY uses ``core._moss_sampling_sig`` (a sha256 over this
    same single :data:`MOSS_SAMPLING_DEFAULTS` dict, aliased as ``core.MOSS_SAMPLING``); both read
    the one source, so cached audio and synthesized audio can never disagree on sampling. Defaults
    to :data:`MOSS_SAMPLING_DEFAULTS`; pass a dict to override individual values (merged on top).
    """
    merged = dict(MOSS_SAMPLING_DEFAULTS)
    if sampling:
        merged.update(sampling)
    return ';'.join(f'{k}={merged[k]}' for k in sorted(merged))


def moss_status() -> str:
    """Three-state availability of the MOSS engine, WITHOUT spawning it.

    - ``'present'`` — binary AND every required GGUF (:data:`MOSS_REQUIRED`) found.
    - ``'partial'`` — binary OR some (but not all) required pieces found: MOSS was clearly
      intended but is mis-installed. The auto-selector treats this as NOT available; whether
      a partial install aborts-loud vs. falls back to Kokoro is the caller's (cli/core)
      policy call (PRD stories 3 vs 17 / ADR 0005), not this module's.
    - ``'absent'`` — no binary and no required GGUF: a clean install that never had MOSS.
    """
    paths = moss_paths()
    have = [paths['binary'] is not None] + [paths[k] is not None for k in MOSS_REQUIRED]
    if all(have):
        return 'present'
    if any(have):
        return 'partial'
    return 'absent'


@dataclass(frozen=True)
class EngineChoice:
    """One selectable engine for a UI selector: its backend id, label, and availability.

    ``available`` is False for an engine that is *known about* but not usable right now (today
    only a half-installed MOSS); ``reason`` is then a non-empty human explanation and is '' when
    available. The GUI renders one radio per choice, disabling the unavailable ones with the
    reason rather than silently hiding them (PRD stories 3/4).
    """
    id: str          # backend id ('cpu' | 'rocm' | 'moss' | ...)
    label: str       # display label (BackendInfo.label)
    available: bool  # selectable on this machine right now
    reason: str      # '' when available; why-unavailable text otherwise


def _moss_unavailable_reason() -> str:
    """Non-empty, human reason MOSS isn't selectable, from which piece :func:`moss_paths` lacks.

    Distinguishes a missing binary from a missing required GGUF (the two ``partial`` causes the
    PRD calls out) so the disabled radio tells the user *what to fix*. ``moss_status`` alone
    only says present/partial/absent, so the granular reason is derived here.
    """
    paths = moss_paths()
    missing = []
    if paths['binary'] is None:
        missing.append('binary')
    if any(paths[k] is None for k in MOSS_REQUIRED):
        missing.append('model')
    if not missing:  # defensive: shouldn't happen (caller only asks when not 'present')
        return 'not available'
    return 'missing ' + ' and '.join(missing)


def engine_choices() -> list['EngineChoice']:
    """Ordered engines to surface in a selector, each flagged available / disabled-with-reason.

    Every backend in :func:`available_backends` is returned ``available=True`` in display order;
    MOSS, when not fully installed (``moss_status() != 'present'``, so it is absent from
    ``available_backends``), is appended **present-but-disabled** with a human reason instead of
    being omitted — replacing the GUI's old silent-hide. This is purely an additive presentation
    seam: :func:`available_backends` / :func:`default_backend` semantics are unchanged (Kokoro
    stays the default), so the selector's default selection never lands on the disabled MOSS radio.
    """
    avail = available_backends()
    choices = [EngineChoice(bid, BACKENDS[bid].label, True, '') for bid in avail]
    if 'moss' not in avail:
        choices.append(EngineChoice('moss', BACKENDS['moss'].label, False, _moss_unavailable_reason()))
    return choices


def _mlx_importable() -> bool:
    """True only on Apple Silicon with mlx-audio installed (no import side effects)."""
    return (platform.system() == 'Darwin'
            and importlib.util.find_spec('mlx_audio') is not None)


def available_backends() -> list[str]:
    """Backends actually usable on this machine, in preference/display order."""
    avail = ['cpu']  # torch is a hard dependency, so CPU is always available
    try:
        import torch
        if torch.cuda.is_available():
            # A torch wheel is built against EITHER CUDA or HIP, so at most one applies.
            if getattr(torch.version, 'hip', None) is not None:
                avail.append('rocm')   # HIP build -> AMD GPU
            elif getattr(torch.version, 'cuda', None) is not None:
                avail.append('cuda')   # CUDA build -> NVIDIA GPU
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            avail.append('mps')
    except Exception:
        pass
    if _mlx_importable():
        avail.append('mlx')
    if moss_status() == 'present':
        avail.append('moss')  # listed only when fully installed -> never a partial-install false positive
    return avail


def default_backend() -> str:
    """Best available backend: prefer MOSS (the quality default), then an accelerator, then CPU.

    MOSS is the user's by-ear quality default (ADR 0003), so a bare ``audiblez book.epub``
    auto-selects it when its binary + GGUFs are fully present. Otherwise prefer a torch/MLX
    accelerator, falling back to CPU. A partial MOSS install is deliberately NOT preferred
    here — ``available_backends`` only lists 'moss' when complete, so this never silently
    half-selects MOSS.

    On Apple Silicon, prefer the native 'mlx' engine over torch 'mps' (mlx is the faster
    Apple path per the README).
    """
    avail = available_backends()
    for pref in ('moss', 'cuda', 'rocm', 'mlx', 'mps', 'cpu'):
        if pref in avail:
            return pref
    return 'cpu'


def gpu_works(backend: str) -> bool:
    """Whether a torch GPU backend can actually run a kernel — not just ``is_available()``.

    Catches the gfx-mismatch case (e.g. RDNA3 without the right HSA_OVERRIDE_GFX_VERSION)
    where ``torch.cuda.is_available()`` is True but the first real op faults at the HIP
    level. Returns True for non-GPU / non-torch backends (nothing to probe). Used to make
    the auto-selected backend fall back to CPU instead of crashing/emitting silence.
    """
    info = BACKENDS.get(backend)
    if info is None or info.engine != 'torch' or info.torch_device in (None, 'cpu'):
        return True
    try:
        import torch
        x = torch.ones(8, 8, device=info.torch_device)
        float((x @ x).sum().item())  # force execution + device sync
        return True
    except Exception:
        return False


def is_gpu(backend: str) -> bool:
    """Whether a backend is GPU-accelerated (drives the chars/sec ETA estimate).

    'moss' counts: the resident llama.cpp co-process runs on the GPU via Vulkan (RADV on the
    AMD box), so the ETA should use the GPU rate, not the CPU one.
    """
    return backend in ('cuda', 'rocm', 'mps', 'mlx', 'moss')
