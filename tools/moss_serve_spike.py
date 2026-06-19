#!/usr/bin/env python3
"""PROTOTYPE — Phase-0 MOSS resident `--serve` spike driver (THROWAWAY).

Drives the patched `llama-moss-tts --serve` (resident pipe co-process) and answers the
5 gate questions from docs/moss-coprocess-spec.md / the handoff. NOT the production
integration — this exists only to convert the design's ESTIMATES into MEASUREMENTS.

Gates:
  1. Resident loop / marginal RTF  — model loads once; steady-state per-request wall/audio
     approaches gen+decode ~0.168 (if ~0.29, the loop failed to amortize the load).
     First request is discarded (pays RADV shader compilation).
  2. Per-request state reset       — A/B/A' replay in ONE session: wav(A) byte-identical to
     wav(A') (same fixed ctx -> same tiling -> bitwise determinism), B differs. Proves
     KV-clear + RNG-reseed correct, no cross-request contamination.
  3. Decoder ctx reuse, variable M — >=2 different frame counts decoded through the SAME
     resident worst-case-sized non-causal decoder ctx; both intelligible (WER) + plausible.
  4. Co-resident VRAM <= 16 GiB    — sample rocm-smi Used DURING a decode (compute buffers
     live), minus idle baseline; text-only and clone (encoder co-resident) peaks.
  5. Cloning resident              — encoder kept loaded, reference encoded once, reused.

Run in the MAIN .venv (soundfile + optional faster-whisper for WER; no torch needed):
    .venv/bin/python tools/moss_serve_spike.py
"""
import hashlib
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import soundfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ab_out' / 'serve_spike'
GGUF = Path('/home/caio/Projects/moss-work/gguf')
BIN = Path('/home/caio/Projects/llama.cpp-moss/build-vulkan/bin/llama-moss-tts')
BACKBONE = GGUF / 'moss_delay_firstclass_Q5_K_M.gguf'
DECODER = GGUF / 'moss_tts_audio_decoder_f16.gguf'
ENCODER = GGUF / 'moss_tts_audio_encoder_f16.gguf'
CLONE_REF = ROOT / 'ab_out' / 'moss_clone_krupp.wav'  # default clone voice (24 kHz mono)

SENTINEL = '@@MOSS@@'

# A handful of real sentences of varying length (gate 1 steady-state + gate 3 frame counts).
SENTENCES = [
    "The lighthouse keeper had not spoken to another soul in nineteen days.",
    "Each morning he climbed the spiral stair and wound the great brass mechanism.",
    "It was, he often thought, a strange kind of freedom.",
    "That evening, the storm came.",
    "He watched the grey Atlantic heave against the rocks far below the gallery rail.",
]
# Gate 2 A/B/A' triple — A and A' identical text+seed, B different.
A_TEXT = "The quick brown fox jumps over the lazy dog near the riverbank at dawn."
B_TEXT = "She sells sea shells by the sea shore on a bright and windy afternoon."
A_SEED = 4242


def rocm_vram_used_bytes():
    """Current VRAM Used (bytes) from rocm-smi; None if unavailable."""
    try:
        p = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                           capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    m = re.search(r'VRAM Total Used Memory \(B\):\s*(\d+)', p.stdout)
    return int(m.group(1)) if m else None


class Serve:
    """Owns one resident llama-moss-tts --serve child over a stdin/stdout pipe."""

    def __init__(self, clone=False, max_prompt_frames=512, max_raw_frames=768, label='text'):
        self.label = label
        self.logf = OUT / f'serve_{label}.stderr.log'
        self._log_fh = open(self.logf, 'wb')
        cmd = [str(BIN), '--serve', '-m', str(BACKBONE),
               '--audio-decoder-model', str(DECODER),
               '--language', 'en', '-ngl', '-1',
               '--max-prompt-frames', str(max_prompt_frames),
               '--max-raw-frames', str(max_raw_frames)]
        if clone:
            cmd += ['--audio-encoder-model', str(ENCODER), '--reference-audio', str(CLONE_REF)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._log_fh, bufsize=0)
        # Drain stdout in a thread into a queue of SENTINEL lines (LOG noise is filtered out).
        self._lines = []
        self._cv = threading.Condition()
        self._ready = False
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def _read_stdout(self):
        for raw in self.proc.stdout:
            line = raw.decode('utf-8', 'replace').rstrip('\n')
            if not line.startswith(SENTINEL):
                continue  # llama / LOG noise — not the control channel
            with self._cv:
                if line == f'{SENTINEL} ready':
                    self._ready = True
                else:
                    self._lines.append(line)
                self._cv.notify_all()

    def wait_ready(self, timeout=180):
        t0 = time.time()
        with self._cv:
            while not self._ready:
                if self.proc.poll() is not None:
                    raise RuntimeError(f'child died before ready (rc={self.proc.returncode}); see {self.logf}')
                if not self._cv.wait_for(lambda: self._ready, timeout=1):
                    if time.time() - t0 > timeout:
                        raise TimeoutError('timeout waiting for ready')
        return time.time() - t0

    def synth(self, req_id, seed, text, wav_out, timeout=120):
        """Send one synth request; block for its response. Returns (status, fields, wall_s)."""
        wav_out = Path(wav_out)
        wav_out.unlink(missing_ok=True)  # stale-WAV guard
        body = text.encode('utf-8')
        header = f'synth {req_id} {seed} {wav_out} {len(body)}\n'.encode('utf-8')
        t0 = time.time()
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()
        # Wait for the response line bearing this id.
        prefix = f'{SENTINEL} id={req_id} '
        with self._cv:
            if not self._cv.wait_for(
                    lambda: any(line.startswith(prefix) for line in self._lines) or self.proc.poll() is not None,
                    timeout=timeout):
                raise TimeoutError(f'timeout on request {req_id}')
            if self.proc.poll() is not None and not any(line.startswith(prefix) for line in self._lines):
                raise RuntimeError(f'child died during request {req_id} (rc={self.proc.returncode}); see {self.logf}')
            line = next(line for line in self._lines if line.startswith(prefix))
            self._lines.remove(line)
        wall = time.time() - t0
        rest = line[len(prefix):]
        parts = rest.split()
        status = parts[0]
        fields = dict(p.split('=', 1) for p in parts[1:] if '=' in p)
        return status, fields, wall

    def shutdown(self):
        try:
            self.proc.stdin.write(b'shutdown\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.kill()
        finally:
            self._log_fh.close()


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def audio_dur(path):
    a, sr = soundfile.read(str(path), dtype='float32')
    return len(a) / sr, sr, len(a)


def try_wer(wav, text):
    """Optional ASR WER round-trip; None if faster-whisper absent."""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    try:
        model = WhisperModel('base.en', device='cpu', compute_type='int8')
        segs, _ = model.transcribe(str(wav))
        hyp = ' '.join(s.text for s in segs)
        norm = lambda s: re.sub(r'[^a-z0-9 ]', '', s.lower()).split()
        ref_w, hyp_w = norm(text), norm(hyp)
        # Levenshtein on words.
        d = list(range(len(hyp_w) + 1))
        for i, rw in enumerate(ref_w, 1):
            prev, d[0] = d[0], i
            for j, hw in enumerate(hyp_w, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
        return round(d[len(hyp_w)] / max(len(ref_w), 1), 3)
    except Exception as e:
        return f'wer_error:{e}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (BIN, BACKBONE, DECODER, ENCODER):
        assert p.exists(), f'missing: {p}'
    assert CLONE_REF.exists(), f'missing clone ref: {CLONE_REF}'

    results = {}
    idle_vram = rocm_vram_used_bytes()
    print(f'[spike] idle VRAM used: {fmt_gib(idle_vram)}')

    # ============================== TEXT-ONLY child ==============================
    print('\n[spike] starting resident text-only child ...')
    s = Serve(clone=False, label='text')
    load_s = s.wait_ready()
    ready_vram = rocm_vram_used_bytes()
    print(f'[spike] ready in {load_s:.2f}s (the ONE load); VRAM used now {fmt_gib(ready_vram)}')

    # ---- Gate 1: marginal RTF over real sentences, discard request #1 ----
    print('\n[gate1] marginal RTF (resident, request #1 discarded) ...')
    rtfs, walls = [], []
    for i, sent in enumerate(SENTENCES):
        wav = OUT / f'g1_{i}.wav'
        st, f, wall = s.synth(1000 + i, 1234, sent, wav)
        assert st == 'ok', f'gate1 req {i} -> {st} {f}'
        dur, sr, _ = audio_dur(wav)
        rtf = wall / dur if dur else None
        tag = ' (discarded: RADV compile)' if i == 0 else ''
        print(f'  [{i}] wall={wall:.2f}s audio={dur:.2f}s rtf={rtf:.3f}{tag}')
        if i > 0:
            rtfs.append(rtf); walls.append(wall)
    rtf_med = round(statistics.median(rtfs), 4)
    results['gate1'] = {
        'load_once_s': round(load_s, 3),
        'steady_rtf_median': rtf_med,
        'steady_rtf_all': [round(r, 4) for r in rtfs],
        'target_gen_decode': 0.168, 'fail_if_near': 0.29,
        'verdict': 'PASS' if rtf_med < 0.235 else 'FAIL',  # midpoint of 0.168/0.29
    }
    print(f'[gate1] steady-state median RTF = {rtf_med}  (target ~0.168, FAIL if ~0.29) -> {results["gate1"]["verdict"]}')

    # ---- Gate 2: A/B/A' byte-identical replay ----
    print('\n[gate2] A/B/A\' replay (determinism + no KV contamination + reseed) ...')
    wa, wb, wa2 = OUT / 'g2_A.wav', OUT / 'g2_B.wav', OUT / 'g2_A2.wav'
    s.synth(2001, A_SEED, A_TEXT, wa)
    s.synth(2002, A_SEED + 7, B_TEXT, wb)   # different text+seed BETWEEN the two A's
    s.synth(2003, A_SEED, A_TEXT, wa2)
    ha, hb, ha2 = md5(wa), md5(wb), md5(wa2)
    a_identical = (ha == ha2)
    b_differs = (ha != hb)
    results['gate2'] = {
        'md5_A': ha, 'md5_A2': ha2, 'md5_B': hb,
        'A_eq_A2': a_identical, 'A_ne_B': b_differs,
        'verdict': 'PASS' if (a_identical and b_differs) else 'FAIL',
    }
    print(f'  md5(A)={ha[:12]} md5(A\')={ha2[:12]} -> identical={a_identical}')
    print(f'  md5(B)={hb[:12]} -> A!=B {b_differs}')
    print(f'[gate2] -> {results["gate2"]["verdict"]}')

    # ---- Gate 3: decoder reuse across >=2 frame counts; WER ----
    print('\n[gate3] decoder ctx reuse across different frame counts ...')
    short_wav, long_wav = OUT / 'g3_short.wav', OUT / 'g3_long.wav'
    short_t = "That evening, the storm came."
    long_t = SENTENCES[0] + ' ' + SENTENCES[1] + ' ' + SENTENCES[4]
    s.synth(3001, 99, short_t, short_wav)
    s.synth(3002, 99, long_t, long_wav)
    sd, _, sn = audio_dur(short_wav)
    ld, _, ln = audio_dur(long_wav)
    wer_s, wer_l = try_wer(short_wav, short_t), try_wer(long_wav, long_t)
    results['gate3'] = {
        'short': {'dur_s': round(sd, 2), 'samples': sn, 'wer': wer_s},
        'long': {'dur_s': round(ld, 2), 'samples': ln, 'wer': wer_l},
        'distinct_frame_counts': sn != ln,
        'verdict': ('PASS' if (sn != ln and sd > 0.3 and ld > sd) else 'CHECK'),
        'note': 'WER None = faster-whisper not installed; verify by ear from the wavs.',
    }
    print(f'  short: {sd:.2f}s ({sn} samp) wer={wer_s}')
    print(f'  long : {ld:.2f}s ({ln} samp) wer={wer_l}')
    print(f'[gate3] distinct frame counts={sn != ln} -> {results["gate3"]["verdict"]}')

    # ---- Gate 4a: text-only VRAM peak during a decode ----
    print('\n[gate4a] sampling VRAM during a text-only decode ...')
    peak = sample_vram_during(s, 4001, 1234, long_t, OUT / 'g4_text.wav')
    results['gate4_text_only'] = vram_record(peak, idle_vram, ready_vram)
    print(f'  peak used={fmt_gib(peak)}  minus idle={fmt_gib(peak - idle_vram) if peak and idle_vram else "?"}  fits16={results["gate4_text_only"]["fits_16gib"]}')

    s.shutdown()

    # ============================== CLONE child ==============================
    print('\n[spike] starting resident CLONE child (encoder co-resident, ref=krupp) ...')
    sc = Serve(clone=True, max_prompt_frames=1024, max_raw_frames=768, label='clone')
    cload = sc.wait_ready()
    cready_vram = rocm_vram_used_bytes()
    print(f'[spike] clone ready in {cload:.2f}s; VRAM used now {fmt_gib(cready_vram)}')

    # ---- Gate 5: cloning works resident ----
    print('\n[gate5] resident cloning ...')
    cl_wav = OUT / 'g5_clone.wav'
    st, f, wall = sc.synth(5001, 7, SENTENCES[0], cl_wav)
    cl_ok = (st == 'ok')
    cd, _, cn = (audio_dur(cl_wav) if cl_ok else (0, 0, 0))
    cl_wer = try_wer(cl_wav, SENTENCES[0]) if cl_ok else None
    results['gate5_clone'] = {
        'status': st, 'fields': f, 'dur_s': round(cd, 2), 'samples': cn,
        'wer': cl_wer, 'ref': str(CLONE_REF),
        'verdict': 'PASS' if (cl_ok and cn > 0) else 'FAIL',
    }
    print(f'  status={st} dur={cd:.2f}s wer={cl_wer} -> {results["gate5_clone"]["verdict"]}')

    # ---- Gate 4b: clone-path VRAM peak (3 models co-resident) ----
    print('\n[gate4b] sampling VRAM during a clone decode (3 models co-resident) ...')
    cpeak = sample_vram_during(sc, 4002, 7, SENTENCES[4], OUT / 'g4_clone.wav')
    results['gate4_clone'] = vram_record(cpeak, idle_vram, cready_vram)
    print(f'  peak used={fmt_gib(cpeak)}  minus idle={fmt_gib(cpeak - idle_vram) if cpeak and idle_vram else "?"}  fits16={results["gate4_clone"]["fits_16gib"]}')

    sc.shutdown()

    # ---- summary ----
    import json
    (OUT / 'spike_results.json').write_text(json.dumps(results, indent=2))
    print('\n================ SPIKE VERDICTS ================')
    for g in ('gate1', 'gate2', 'gate3', 'gate4_text_only', 'gate5_clone', 'gate4_clone'):
        print(f'  {g:18s} {results[g].get("verdict", "—")}')
    print(f'\n[spike] full results -> {OUT / "spike_results.json"}')
    print(f'[spike] wavs in {OUT} — listen to g5_clone.wav / g3_*.wav by ear.')


GIB = 1024 ** 3


def fmt_gib(b):
    return f'{b / GIB:.2f} GiB' if b else '?'


def vram_record(peak, idle, ready):
    moss = (peak - idle) if (peak and idle) else None
    return {
        'peak_used_bytes': peak, 'idle_baseline_bytes': idle, 'ready_loaded_bytes': ready,
        'peak_used_gib': round(peak / GIB, 3) if peak else None,
        'moss_attributable_gib': round(moss / GIB, 3) if moss else None,
        'fits_16gib': (peak < 16 * GIB) if peak else None,
        'verdict': ('PASS' if (peak and peak < 16 * GIB) else 'FAIL' if peak else 'UNMEASURED'),
    }


def sample_vram_during(serve, req_id, seed, text, wav):
    """Poll rocm-smi in a thread while a synth runs; return peak Used bytes."""
    peak = [0]
    stop = threading.Event()

    def poller():
        while not stop.is_set():
            v = rocm_vram_used_bytes()
            if v and v > peak[0]:
                peak[0] = v
            time.sleep(0.05)

    th = threading.Thread(target=poller, daemon=True)
    th.start()
    serve.synth(req_id, seed, text, wav)
    stop.set()
    th.join()
    return peak[0] or None


if __name__ == '__main__':
    sys.exit(main())
