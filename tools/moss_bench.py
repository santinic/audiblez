#!/usr/bin/env python3
"""MOSS-TTS (OpenMOSS MossTTSDelay 8B) benchmark on the RX 7800 XT via llama.cpp-Vulkan.

MOSS's first-class path is a one-shot C++ binary (`llama-moss-tts`), NOT a resident
Python model — so a naive subprocess-per-run RTF would fold the multi-GB model LOAD into
every timed call (unlike Kokoro/Orpheus whose RTF excludes load). To stay comparable we
measure load and generation+decode SEPARATELY:

  * baseline run  (tiny text)  -> wall ≈ load + init + min gen/decode      == fixed overhead
  * passage run   (PASSAGE)    -> wall ≈ overhead + gen(passage) + decode(passage)
  * gen+decode    = median(passage_wall) - median(baseline_wall)           == "resident" cost

In a real audiblez integration the model is loaded ONCE per book, so the production-relevant
number is gen+decode RTF; the total (load-included) RTF is also reported for honesty.

CROSS-TOOL COMPARABILITY (vs tools/tts_ab.py, which shares ab_out/): tts_ab.py's `rtf` is
load-EXCLUDED (model resident across warm runs; load measured separately as load_s). Neither
moss number is method-identical to it, because the llama-moss-tts binary emits NO llama_perf
timings (verified — see findings.md), so we cannot isolate prompt-eval+eval the clean way:
  * rtf_total      = p_med/audio_s        -> load-INCLUDED  -> OVERSTATES (upper bound)
  * rtf_gen_decode = (p_med-t_med)/audio_s-> load-EXCLUDED but baseline-subtracts the TINY
                     run's own gen+decode -> UNDERSTATES (lower bound); this is the CLOSEST
                     to tts_ab.py's method (same load-excluded axis), just biased low.
The true resident RTF is BRACKETED: rtf_gen_decode <= true <= rtf_total. See 'rtf_note' in JSON.

Writes ab_out/moss.wav (the PASSAGE) + ab_out/moss.json. Run with ANY python (no torch needed).
"""
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import soundfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ab_out'
GGUF = Path('/home/caio/Projects/moss-work/gguf')
BIN = Path('/home/caio/Projects/llama.cpp-moss/build-vulkan/bin/llama-moss-tts')
BACKBONE = GGUF / 'moss_delay_firstclass_Q5_K_M.gguf'
DECODER = GGUF / 'moss_tts_audio_decoder_f16.gguf'

# Same passage as tools/tts_ab.py (keep in sync).
PASSAGE = (
    "The lighthouse keeper had not spoken to another soul in nineteen days. "
    "Each morning he climbed the spiral stair, wound the great brass mechanism, "
    "and watched the grey Atlantic heave against the rocks below. "
    "It was, he often thought, a strange kind of freedom: to be needed by ships "
    "he would never see, and forgotten by everyone else. That evening, the storm came."
)
TINY = "Hello."


def synth(text, wav_out, ngl=-1):
    """Run llama-moss-tts once; return (wall_s, log_text)."""
    cmd = [str(BIN), '-m', str(BACKBONE), '--audio-decoder-model', str(DECODER),
           '--text', text, '--wav-out', str(wav_out), '-ngl', str(ngl)]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(f'llama-moss-tts rc={p.returncode}\n{p.stderr[-2000:]}')
    return wall, p.stderr


def parse_vram(log):
    mib = 0.0
    for m in re.finditer(r'model buffer size\s*=\s*([\d.]+)\s*MiB', log):
        mib += float(m.group(1))
    for m in re.finditer(r'(KV buffer size|compute buffer size)\s*=\s*([\d.]+)\s*MiB', log):
        mib += float(m.group(2))
    return round(mib / 1024, 3) if mib else None


def main():
    OUT.mkdir(exist_ok=True)
    assert BIN.exists() and BACKBONE.exists() and DECODER.exists(), 'missing binary or GGUFs'
    runs = 3
    tmp_p = OUT / '_moss_passage.wav'
    tmp_t = OUT / '_moss_tiny.wav'

    print('[moss] warm-up (pays Vulkan pipeline compilation + page cache) ...')
    _, log = synth(PASSAGE, tmp_p)
    vram = parse_vram(log)

    print('[moss] timing PASSAGE runs ...')
    p_walls = []
    for i in range(runs):
        w, _ = synth(PASSAGE, tmp_p)
        p_walls.append(w)
        print(f'  passage {i+1}/{runs}: {w:.2f}s')

    print('[moss] timing baseline (tiny) runs ...')
    t_walls = []
    for i in range(runs):
        w, _ = synth(TINY, tmp_t)
        t_walls.append(w)
        print(f'  tiny {i+1}/{runs}: {w:.2f}s')

    audio, sr = soundfile.read(str(tmp_p), dtype='float32', always_2d=False)
    audio_s = len(audio) / sr
    p_med = statistics.median(p_walls)
    t_med = statistics.median(t_walls)
    # BIAS NOTE: subtracting the TINY ("Hello.") wall removes load+init AND the tiny run's own
    # (small) gen+decode, so this UNDERSTATES MOSS's true gen+decode. The honest fix would parse
    # llama.cpp's prompt-eval/eval timers, but llama-moss-tts emits none (findings.md) — so the
    # baseline subtraction is the sanctioned fallback. Keep the value clamped >=0 for storage,
    # but treat a non-positive raw delta as "unmeasurable" for the RTF (see guard below).
    raw_delta = p_med - t_med
    gen_decode_s = max(0.0, raw_delta)
    # Guard: if the passage barely (or doesn't) exceed the tiny baseline, raw_delta <= ~0 and the
    # old code reported a real-looking rtf_gen_decode of 0.0. Emit null + a note instead, so a
    # degenerate baseline never masquerades as "infinitely faster than real time".
    GEN_DECODE_MIN_S = 0.05  # below this the subtraction is dominated by run-to-run noise
    rtf_gen_decode = (round(gen_decode_s / audio_s, 4)
                      if (audio_s and raw_delta > GEN_DECODE_MIN_S) else None)
    rtf_total = round(p_med / audio_s, 4) if audio_s else None

    metrics = {
        'model': 'moss',
        'runtime': 'llama.cpp-Vulkan (RADV), Q5_K_M backbone + f16 decoder GGUF',
        'device': 'gpu-vulkan',
        'sr': sr,
        'audio_s': round(audio_s, 3),
        'load_overhead_s': round(t_med, 3),          # tiny baseline ≈ load+init
        'synth_total_s': round(p_med, 3),            # passage wall, INCLUDES load
        'synth_gen_decode_s': round(gen_decode_s, 3),  # production-relevant (model resident), biased low
        # rtf_total: load-INCLUDED -> upper bound; NOT method-identical to tts_ab.py's load-excluded rtf.
        'rtf_total': rtf_total,
        # rtf_gen_decode: load-EXCLUDED (closest to tts_ab.py's method) but baseline-subtracted -> lower
        # bound; null when the tiny/passage delta is too small to measure honestly (see guard above).
        'rtf_gen_decode': rtf_gen_decode,
        'rtf_note': ('llama-moss-tts emits no per-stage timers, so no number is method-identical to '
                     'tts_ab.py rtf (which is load-excluded). True resident RTF is bracketed: '
                     'rtf_gen_decode (lower, baseline-subtracted) <= true <= rtf_total (upper, '
                     'load-included). Compare against tts_ab.py rtf with this bracket in mind.'),
        'vram_gb': vram,
        'runs': runs,
    }
    final = OUT / 'moss.wav'
    soundfile.write(final, audio, sr)
    (OUT / 'moss.json').write_text(json.dumps(metrics, indent=2))
    for t in (tmp_p, tmp_t):
        t.unlink(missing_ok=True)
    print(f'[moss] wrote {final} ({audio_s:.1f}s) and moss.json')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
