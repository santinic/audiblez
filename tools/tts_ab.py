#!/usr/bin/env python3
"""Standalone A/B harness: compare TTS engines on the RX 7800 XT (ROCm).

Each engine lives in its OWN venv (we never pollute the main .venv's torch
2.12.1+rocm7.2). So run this script with whichever venv has the target model:

    .venv/bin/python            tools/tts_ab.py --make-ref          # mint OPTIONAL synthetic ref (Kokoro)
    .venv/bin/python            tools/tts_ab.py --model kokoro       # baseline
    .venv-styletts2/bin/python  tools/tts_ab.py --model styletts2
    .venv-voxcpm2/bin/python    tools/tts_ab.py --model voxcpm2
    .venv-cosyvoice2/bin/python tools/tts_ab.py --model cosyvoice2

Adapters import their libraries LAZILY, so importing this file never requires a
model that isn't in the current venv. Each run writes:

    ab_out/<model>.wav      the synthesized passage (native sample rate)
    ab_out/<model>.json     metrics: load_s, rtf, vram_gb, sr, audio_s

VOICE: the cloning models (VoxCPM2/CosyVoice2/StyleTTS2) clone the REAL HUMAN clip
tools/ref_human.wav (the default --ref-audio) so the A/B is "same voice, different
model". --make-ref mints a SEPARATE Kokoro-synthesized clip (tools/ref_voice.wav)
that you may optionally pass via --ref-audio; it never touches ref_human.wav. Cloning
from a Kokoro clip would feed the cloners out-of-distribution audio and bias the A/B
toward Kokoro, so ref_human.wav stays the default. Override with --ref-audio / --ref-text.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / 'tools'
OUT = ROOT / 'ab_out'
# Real human English reference (F5-TTS purpose-built clip) — the cloning models clone THIS.
# A Kokoro-synthesized reference would feed the cloners out-of-distribution audio and bias
# the A/B toward Kokoro (which can't clone anyway, and narrates in native af_sky regardless).
# This is the DEFAULT --ref-audio; --make-ref must NEVER write here (it would clobber the
# human clip with Kokoro audio and silently corrupt the A/B).
REF_WAV = TOOLS / 'ref_human.wav'
REF_TXT = TOOLS / 'ref_human.txt'
# SEPARATE, OPTIONAL synthetic reference minted by --make-ref (Kokoro). Kept distinct from
# ref_human.wav so the human clip is never overwritten; pass it explicitly via --ref-audio
# if you specifically want to compare "clone a Kokoro voice" (knowingly Kokoro-biased).
REF_VOICE_WAV = TOOLS / 'ref_voice.wav'
REF_VOICE_TXT = TOOLS / 'ref_voice.txt'

# Fixed book-style passage every engine narrates (≈430 chars, mixed punctuation).
PASSAGE = (
    "The lighthouse keeper had not spoken to another soul in nineteen days. "
    "Each morning he climbed the spiral stair, wound the great brass mechanism, "
    "and watched the grey Atlantic heave against the rocks below. "
    "It was, he often thought, a strange kind of freedom: to be needed by ships "
    "he would never see, and forgotten by everyone else. That evening, the storm came."
)
# Clean, phonetically varied sentence used to mint the shared reference voice.
REF_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while the bright autumn moon "
    "rises gently above the quiet harbor."
)


# --------------------------------------------------------------------------- #
# GPU helpers (torch is present in every model venv; guard anyway)
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    return torch


def _cuda_sync():
    try:
        t = _torch()
        if t.cuda.is_available():
            t.cuda.synchronize()
    except Exception:
        pass


def _vram_gb():
    # CAVEAT: torch-CUDA(/ROCm)-only. max_memory_allocated() tracks ONLY torch's own
    # caching allocator, so it reads ~0 (or None) for non-torch backends — e.g. Orpheus
    # (llama.cpp-Vulkan) whose VRAM lives outside torch. Do NOT compare this field across
    # torch vs non-torch runtimes; the emitted JSON carries a 'vram_gb_note' saying so.
    try:
        t = _torch()
        if t.cuda.is_available():
            return round(t.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return None


def _as_mono_f32(audio):
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    return a


# --------------------------------------------------------------------------- #
# Adapters. Each returns (model_obj, synth_fn) where
#   synth_fn(model_obj, text, ref_wav: Path|None, ref_text: str|None) -> (np.float32 mono, sr:int)
# --------------------------------------------------------------------------- #
def load_kokoro(device):
    """Kokoro via the audiblez engine (only in the main .venv)."""
    import audiblez.core as core
    core.set_espeak_library()
    backend = 'rocm' if device == 'cuda' else 'cpu'
    synth = core.build_synthesizer('af_sky', backend)

    def run(_m, text, ref_wav, ref_text):
        segs = synth(text, 1.0)
        return _as_mono_f32(np.concatenate(segs)), 24000
    return synth, run


def load_kokoro_compiled(device):
    """Kokoro with torch.compile on the tensor-heavy forward (the §14 launch-overhead lever).
    Targets forward_with_tokens (the GPU ops); dynamic=True to tolerate variable sentence lengths
    without per-shape recompiles. mode='default' (reduce-overhead/cudagraphs is fragile on ROCm)."""
    import torch
    import audiblez.core as core
    from kokoro import KPipeline
    core.set_espeak_library()
    dev = 'cuda' if device == 'cuda' else 'cpu'
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=dev)
    pipeline.model.forward_with_tokens = torch.compile(
        pipeline.model.forward_with_tokens, mode='default', dynamic=True)

    def run(m, text, ref_wav, ref_text):
        segs = [core.to_numpy(audio)
                for _gs, _ps, audio in pipeline(text, voice='af_sky', speed=1.0, split_pattern=r'\n\n\n')]
        return np.concatenate(segs), 24000
    return pipeline, run


def load_fish(device):
    """Fish-Speech / OpenAudio S1-mini (#1 open blind-naturalness). Dual-AR LLM (model.pth) +
    DAC codec decoder (codec.pth), via fish_speech.inference_engine. PyTorch-ROCm (eager → slow band).
    Voice-clones the reference wav + transcript. The repo lives outside audiblez at FISH_ROOT."""
    import sys
    import torch
    import torchaudio
    import soundfile as _sf
    # torchaudio 2.11 forces the CUDA-only torchcodec backend for .load on ROCm -> patch to soundfile.
    # fish calls it with both file PATHS and in-memory BytesIO (reference audio bytes), so handle both.
    import io as _io
    def _sf_load(path, *a, **k):
        if hasattr(path, 'read'):
            path.seek(0); src = path
        elif isinstance(path, (bytes, bytearray)):
            src = _io.BytesIO(path)
        else:
            src = str(path)
        audio, srr = _sf.read(src, dtype='float32', always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(audio.T)), srr
    torchaudio.load = _sf_load
    # The s1-era tree (781bf1c) predates torchaudio 2.9's removal of list_audio_backends;
    # ReferenceLoader.__init__ only uses it to pick ffmpeg vs soundfile, and we force soundfile
    # via the patched .load above, so a stub returning [] (-> backend="soundfile") is safe.
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: []
    FISH_ROOT = '/home/caio/Projects/fish-speech'
    if FISH_ROOT not in sys.path:
        sys.path.insert(0, FISH_ROOT)
    from fish_speech.inference_engine import TTSInferenceEngine
    from fish_speech.models.dac.inference import load_model as load_decoder
    from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
    from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio

    ckpt = FISH_ROOT + '/checkpoints/s1-mini'
    dev = 'cuda' if device == 'cuda' else 'cpu'
    precision = torch.half
    llama_queue = launch_thread_safe_queue(checkpoint_path=ckpt, device=dev, precision=precision, compile=False)
    decoder = load_decoder(config_name='modded_dac_vq', checkpoint_path=ckpt + '/codec.pth', device=dev)
    engine = TTSInferenceEngine(llama_queue=llama_queue, decoder_model=decoder, precision=precision, compile=False)
    ref_bytes_cache = {}

    def run(m, text, ref_wav, ref_text):
        refs = []
        if ref_wav:
            if ref_wav not in ref_bytes_cache:
                ref_bytes_cache[ref_wav] = open(str(ref_wav), 'rb').read()
            refs = [ServeReferenceAudio(audio=ref_bytes_cache[ref_wav], text=ref_text or '')]
        req = ServeTTSRequest(text=text, references=refs, format='wav', max_new_tokens=1024,
                              chunk_length=200, top_p=0.7, repetition_penalty=1.2, temperature=0.7)
        sr, audio = None, None
        for r in engine.inference(req):
            if r.code == 'final':
                sr, audio = r.audio
            elif r.code == 'error':
                raise RuntimeError(str(r.error))
        a = _as_mono_f32(np.asarray(audio))
        if np.abs(a).max() > 1.5:          # int16 -> float
            a = a / 32768.0
        return a, sr
    return engine, run


def load_orpheus(device):
    """Orpheus-3B via orpheus-cpp: LLM on llama.cpp-VULKAN (real AMD accel, escapes eager-PyTorch)
    + SNAC-ONNX decoder. n_gpu_layers=-1 offloads all layers to the GPU via Vulkan. 24kHz, voice 'tara'.
    This is the §12 'one viable AMD-first upgrade' path."""
    from orpheus_cpp import OrpheusCpp
    ngl = -1 if device == 'cuda' else 0   # harness 'cuda' == GPU == Vulkan offload here
    model = OrpheusCpp(n_gpu_layers=ngl, lang='en', verbose=False)

    def run(m, text, ref_wav, ref_text):
        sr, audio = m.tts(text, {'voice_id': 'tara'})     # audio int16, shape (1, N)
        a = _as_mono_f32(np.asarray(audio)) / 32768.0      # int16 -> float32 [-1,1]
        return a, sr
    return model, run


def load_styletts2(device):
    """StyleTTS2 (pip wrapper). Auto-downloads LibriTTS ckpt (~1GB) on first construct.
    Clones the reference via target_voice_path (no transcript needed). 24kHz out.
    StyleTTS2 has a ~per-sentence length cap, so we split + concatenate (as audiblez would).
    """
    import re as _re
    import functools
    import torch
    # StyleTTS2 0.1.6 predates torch 2.6's weights_only=True default; its trusted
    # checkpoints (official HF LibriTTS + bundled aux models) pickle non-tensor globals.
    # Restore the old default so torch.load accepts them.
    if not getattr(torch.load, '_wo_patched', False):
        _orig_load = torch.load

        @functools.wraps(_orig_load)
        def _load(*a, **k):
            k.setdefault('weights_only', False)
            return _orig_load(*a, **k)
        _load._wo_patched = True
        torch.load = _load
    from styletts2 import tts
    model = tts.StyleTTS2()

    def run(m, text, ref_wav, ref_text):
        # Compute the clone style vector ONCE (as a real integration would), then reuse it
        # for every sentence — inference(target_voice_path=...) otherwise recomputes it per call.
        if not hasattr(m, '_cached_ref_s'):
            m._cached_ref_s = m.compute_style(str(ref_wav)) if ref_wav else None
        sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        out = []
        for s in sents:
            wav = m.inference(s, ref_s=m._cached_ref_s, output_sample_rate=24000,
                              alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1.0)
            out.append(_as_mono_f32(wav))
        return np.concatenate(out), 24000
    return model, run


def load_styletts2_compiled(device):
    """StyleTTS2 + torch.compile on its heavy, compilable submodules (decoder/vocoder, diffusion,
    text_encoder, bert_encoder). Skips the LSTM predictor (won't fuse) and training-only discriminators.
    Tests whether compile helps a bigger model where it didn't help tiny Kokoro."""
    model, run = load_styletts2(device)  # reuse setup (weights_only patch, compute_style reuse, etc.)
    import torch
    mm = model.model
    # NOTE: 'decoder' (iSTFTNet vocoder) crashes under Inductor on ROCm — upsample_linear1d lowering
    # bug with dynamic shapes. Excluded. Compile only the transformer/diffusion parts.
    for name in ('diffusion', 'text_encoder', 'bert_encoder'):
        try:
            mm[name] = torch.compile(mm[name], mode='default', dynamic=True)
            print(f'[styletts2-compiled] compiled {name}')
        except Exception as e:
            print(f'[styletts2-compiled] SKIP {name}: {type(e).__name__}: {e}')
    return model, run


def load_voxcpm2(device):
    """VoxCPM2 (Apache, 2B diffusion-AR). optimize=False REQUIRED on ROCm (torch.compile is
    CUDA-tuned); load_denoiser=False skips an extra download. Clones prompt_wav + prompt_text.
    Native output is 48kHz. Per-sentence to match the other adapters / audiblez's feeding."""
    import re as _re
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained('openbmb/VoxCPM2', device='cuda' if device == 'cuda' else 'cpu',
                                   optimize=False, load_denoiser=False)
    sr = getattr(getattr(model, 'tts_model', model), 'sample_rate', 48000)

    def run(m, text, ref_wav, ref_text):
        sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        out = []
        for s in sents:
            wav = m.generate(text=s, prompt_wav_path=str(ref_wav) if ref_wav else None,
                             prompt_text=ref_text, reference_wav_path=str(ref_wav) if ref_wav else None,
                             cfg_value=2.0, inference_timesteps=10)
            out.append(_as_mono_f32(wav))
        return np.concatenate(out), sr
    return model, run


def load_cosyvoice2(device):
    """CosyVoice2-0.5B (Apache, LLM-AR). Source repo at COSY_ROOT (not pip-installed), so we
    add it + its Matcha-TTS submodule to sys.path. Auto-uses GPU via torch.cuda (ROCm). 24kHz.
    Zero-shot clone: load_wav(ref,16000) + inference_zero_shot(text, ref_text, prompt_16k)."""
    import sys
    import re as _re
    COSY_ROOT = '/home/caio/Projects/CosyVoice'
    for p in (COSY_ROOT, COSY_ROOT + '/third_party/Matcha-TTS'):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import torchaudio
    import soundfile as sf
    # cosyvoice's load_wav uses torchaudio.load, which on torchaudio 2.11 forces the
    # torchcodec backend whose only wheel is CUDA-only (libnvrtc) and dead on ROCm.
    # Patch torchaudio.load to a soundfile reader returning (channels, frames), sr.
    def _sf_load(path, *a, **k):
        audio, srr = sf.read(str(path), dtype='float32', always_2d=True)  # (frames, channels)
        return torch.from_numpy(np.ascontiguousarray(audio.T)), srr        # (channels, frames)
    torchaudio.load = _sf_load
    from cosyvoice.cli.cosyvoice import CosyVoice2 as CV2
    model = CV2(COSY_ROOT + '/pretrained_models/CosyVoice2-0.5B',
                load_jit=False, load_trt=False, fp16=False)
    sr = model.sample_rate

    def run(m, text, ref_wav, ref_text):
        # inference_zero_shot takes the prompt as a FILE PATH (loaded internally at 24k + 16k).
        ref_path = str(ref_wav)
        sents = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        out = []
        for s in sents:
            for o in m.inference_zero_shot(s, ref_text, ref_path, stream=False):
                out.append(_as_mono_f32(o['tts_speech'].squeeze(0).cpu().numpy()))
        return np.concatenate(out), sr
    return model, run


ADAPTERS = {
    'kokoro': load_kokoro,
    'kokoro-compiled': load_kokoro_compiled,
    'orpheus': load_orpheus,
    'fish': load_fish,
    'styletts2-compiled': load_styletts2_compiled,
    'styletts2': load_styletts2,
    'voxcpm2': load_voxcpm2,
    'cosyvoice2': load_cosyvoice2,
}


# --------------------------------------------------------------------------- #
def make_reference(device):
    """Mint the OPTIONAL synthetic reference voice once, with Kokoro (main .venv).
    Writes ONLY to REF_VOICE_WAV/REF_VOICE_TXT — never REF_WAV/REF_TXT, so the real
    human cloning reference (ref_human.wav) is never clobbered with Kokoro audio."""
    _, run = load_kokoro(device)
    audio, sr = run(None, REF_SENTENCE, None, None)
    TOOLS.mkdir(exist_ok=True)
    soundfile.write(REF_VOICE_WAV, audio, sr)
    REF_VOICE_TXT.write_text(REF_SENTENCE + '\n')
    dur = len(audio) / sr
    print(f'Synthetic reference voice written: {REF_VOICE_WAV} ({dur:.1f}s @ {sr} Hz)')
    print(f'Synthetic reference transcript:    {REF_VOICE_TXT}')
    print(f'NOTE: this is the OPTIONAL Kokoro-minted clip; the cloning default stays {REF_WAV.name}.')
    print(f'      Pass --ref-audio {REF_VOICE_WAV} to clone from it (knowingly Kokoro-biased).')


def benchmark(model, device, ref_wav, ref_text, runs):
    OUT.mkdir(exist_ok=True)
    print(f'[{model}] loading on device={device} ...')
    t0 = time.time()
    model_obj, run = ADAPTERS[model](device)
    load_s = time.time() - t0
    print(f'[{model}] loaded in {load_s:.2f}s')

    # warm-up (compiles kernels / first-call cost excluded from steady-state)
    audio, sr = run(model_obj, PASSAGE, ref_wav, ref_text)
    _cuda_sync()

    # Collect per-run (time, audio_duration) so RTF is self-consistent for stochastic-length
    # models (Fish/VoxCPM2 emit different-length audio per run). Dividing a median time by a
    # NON-median duration (e.g. the last run's) would mix samples; instead we take the median
    # of per-run RTFs. We keep the LAST run's audio array for the .wav write below.
    times = []
    audio_durs = []
    for i in range(runs):
        t = time.time()
        audio, sr = run(model_obj, PASSAGE, ref_wav, ref_text)
        _cuda_sync()
        times.append(time.time() - t)
        audio_durs.append(len(_as_mono_f32(audio)) / sr)
        print(f'[{model}] run {i+1}/{runs}: {times[-1]:.3f}s')

    # NOTE: the .wav holds ONE run (the last); audio_s is the median duration across runs, so for
    # stochastic-length models the wav's own length may differ slightly from audio_s. Harmless —
    # score_wer.py reads the wav, not audio_s — but the two are intentionally decoupled.
    audio = _as_mono_f32(audio)               # last run's audio — written to the .wav
    audio_s = statistics.median(audio_durs)   # representative duration (median across runs)
    median_s = statistics.median(times)
    # Per-run RTF = time_i / audio_s_i, then median — self-consistent (each ratio is same-run).
    per_run_rtf = [t / d for t, d in zip(times, audio_durs) if d]
    rtf = round(statistics.median(per_run_rtf), 4) if per_run_rtf else None
    metrics = {
        'model': model,
        'device': device,
        'sr': sr,
        'audio_s': round(audio_s, 3),
        'load_s': round(load_s, 3),
        'synth_median_s': round(median_s, 3),
        'rtf': rtf,  # median of per-run (time_i/audio_s_i); <1 = faster than real time
        'vram_gb': _vram_gb(),
        # torch-CUDA/ROCm-only metric; ~0/None for llama.cpp-Vulkan & other non-torch backends.
        'vram_gb_note': 'torch CUDA/ROCm allocator only; not comparable to non-torch backends',
        'runs': runs,
        'ref_wav': str(ref_wav) if ref_wav else None,
    }
    wav_path = OUT / f'{model}.wav'
    json_path = OUT / f'{model}.json'
    soundfile.write(wav_path, audio, sr)
    json_path.write_text(json.dumps(metrics, indent=2))
    print(f'[{model}] wrote {wav_path} ({audio_s:.1f}s) and {json_path}')
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', choices=list(ADAPTERS))
    ap.add_argument('--device', default='cuda', help="'cuda' (ROCm) or 'cpu'")
    ap.add_argument('--ref-audio', dest='ref_audio', default=str(REF_WAV))
    ap.add_argument('--ref-text', dest='ref_text', default=None)
    ap.add_argument('--runs', type=int, default=4)
    ap.add_argument('--make-ref', action='store_true',
                    help='mint the OPTIONAL synthetic Kokoro reference (tools/ref_voice.wav) and exit; '
                         'does NOT touch the default human ref (tools/ref_human.wav)')
    args = ap.parse_args()

    if args.make_ref:
        make_reference(args.device)
        return
    if not args.model:
        ap.error('--model is required (or use --make-ref)')

    ref_wav = Path(args.ref_audio) if args.ref_audio and Path(args.ref_audio).exists() else None
    ref_text = args.ref_text
    # ref_text defaults to the HUMAN ref's transcript (REF_TXT). If you override --ref-audio with
    # ref_voice.wav (the Kokoro-minted clip), ALSO pass --ref-text (or the cloners get the wrong
    # transcript for that clip — ref_voice.txt holds the right one).
    if ref_text is None and REF_TXT.exists():
        ref_text = REF_TXT.read_text().strip()
    benchmark(args.model, args.device, ref_wav, ref_text, args.runs)


if __name__ == '__main__':
    main()
