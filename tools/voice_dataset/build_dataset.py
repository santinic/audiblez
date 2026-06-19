#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Turn many hours of podcast audio into a clean single-speaker TTS fine-tuning dataset.

Pipeline (per input file):
  1. Decode to one 24 kHz mono working wav (ffmpeg; optionally Demucs vocal-isolation first).
  2. WhisperX: transcribe -> word-level align -> diarize -> assign a speaker to every word.
  3. Pick the narrator: cosine-match each diarized speaker to a --reference clip
     (SpeechBrain ECAPA), or fall back to the dominant (most-spoken) speaker per file.
  4. Keep only the narrator's words, segment them into 1-15 s clips on sentence boundaries
     (never mid-word; break on time gaps so non-contiguous audio is never glued together).
  5. Loudness-normalize each clip and write it; emit LJSpeech / StyleTTS2 / XTTS manifests.

The output is a SUPERSET dataset that feeds StyleTTS2, Piper, or Coqui XTTS unchanged
(see README.md). Resumable: already-processed source files are skipped; manifests are
regenerated from an append-only records.jsonl on every run.

This is a standalone offline tool — it is NOT imported by audiblez and has its own deps
(see requirements.txt). Heavy ML libs are imported lazily so `--help` works without them.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.mp4', '.flac', '.ogg', '.opus', '.aac', '.wma', '.webm'}
TERMINAL = {'.', '?', '!'}
_CLOSERS = '"\')]}»”’'  # trailing quotes/brackets that can follow terminal punctuation


def ends_sentence(text: str) -> bool:
    """True if text ends a sentence, allowing a trailing quote/bracket (e.g. ``go!"``)."""
    return text.rstrip(_CLOSERS)[-1:] in TERMINAL


# --------------------------------------------------------------------------- #
# Device selection
# --------------------------------------------------------------------------- #
def pick_device(override: str | None) -> str:
    """'cuda' (covers NVIDIA *and* AMD ROCm), else the override, else 'cpu'.

    faster-whisper/CTranslate2 has no MPS kernel, so on a Mac we run on CPU by
    default; ROCm torch reports cuda.is_available()==True, so the AMD box gets 'cuda'.
    """
    if override:
        return override
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


# --------------------------------------------------------------------------- #
# Stage 1 — produce one 24 kHz mono working wav per source
# --------------------------------------------------------------------------- #
def ffmpeg_to_mono_wav(src: Path, dst: Path, sr: int) -> None:
    """Decode any input to mono PCM-16 at `sr` Hz using ffmpeg's soxr resampler."""
    subprocess.run(
        ['ffmpeg', '-y', '-i', str(src), '-ac', '1', '-ar', str(sr),
         '-af', 'aresample=resampler=soxr:precision=28', '-c:a', 'pcm_s16le', str(dst)],
        check=True, capture_output=True, timeout=300)  # hung ffmpeg on a bad file must not block forever


def to_working_wav(src: Path, dst: Path, sr: int, separator) -> None:
    """Write the 24 kHz mono working wav, optionally via Demucs vocal isolation.

    Demucs strips music beds/ads but degrades already-clean speech, so it is opt-in
    (`--demucs`) and only worth it for episodes with music under the dialogue.
    """
    if separator is None:
        ffmpeg_to_mono_wav(src, dst, sr)
        return
    from demucs.api import save_audio  # lazy; only when --demucs
    _origin, stems = separator.separate_audio_file(str(src))
    vocals = stems['vocals'].clamp(-1, 1)  # (channels, samples) @ 44100, can overshoot
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tf:
        raw = Path(tf.name)
    try:
        save_audio(vocals, str(raw), samplerate=separator.samplerate)
        ffmpeg_to_mono_wav(raw, dst, sr)
    finally:
        raw.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Stage 2 — WhisperX transcribe + align + diarize + assign speakers
# --------------------------------------------------------------------------- #
def load_whisperx_models(model_name, device, compute_type, language, hf_token, diarize_model):
    """Load the (reusable) WhisperX models once. Returns a dict of handles."""
    import whisperx
    from whisperx.diarize import DiarizationPipeline  # moved out of top-level in 3.3.4+
    asr = whisperx.load_model(model_name, device, compute_type=compute_type, language=language)
    align_model, align_meta = whisperx.load_align_model(language_code=language, device=device)
    diar = DiarizationPipeline(model_name=diarize_model, token=hf_token, device=device)  # token=, not use_auth_token=
    return {'whisperx': whisperx, 'asr': asr, 'align_model': align_model,
            'align_meta': align_meta, 'diar': diar, 'device': device}


def transcribe_file(work_wav: Path, m, batch_size):
    """Return (WhisperX result with word-level timestamps, the 16 kHz ASR audio array)."""
    whisperx = m['whisperx']
    audio = whisperx.load_audio(str(work_wav))  # float32 mono @ 16 kHz (for ASR only)
    result = m['asr'].transcribe(audio, batch_size=batch_size)
    if result['segments']:
        result = whisperx.align(result['segments'], m['align_model'], m['align_meta'],
                                audio, m['device'], return_char_alignments=False)
    return result, audio


def diarize_and_assign(audio, result, m, min_speakers, max_speakers):
    whisperx = m['whisperx']
    diar_df = m['diar'](audio, min_speakers=min_speakers, max_speakers=max_speakers)
    return whisperx.assign_word_speakers(diar_df, result)


def narrator_words(result):
    """Flatten to [{'word','start','end','speaker'}], using segment speaker as word fallback.

    Words that wav2vec2 could not align (missing start/end) are dropped — they have no
    usable timestamp to cut on.
    """
    out = []
    for seg in result.get('segments', []):
        seg_spk = seg.get('speaker')
        for w in seg.get('words', []):
            if w.get('start') is None or w.get('end') is None:
                continue
            text = (w.get('word') or '').strip()
            if not text:
                continue
            spk = w.get('speaker', seg_spk)
            if spk is None:  # un-diarized word: don't let it pollute the single-speaker set
                continue
            out.append({'word': text, 'start': float(w['start']), 'end': float(w['end']),
                        'speaker': spk})
    out.sort(key=lambda x: x['start'])
    return out


# --------------------------------------------------------------------------- #
# Stage 3 — pick the narrator (reference cosine-match, or dominant fallback)
# --------------------------------------------------------------------------- #
class SpeakerMatcher:
    """SpeechBrain ECAPA embedder for cross-file narrator identification (not HF-gated)."""

    def __init__(self, device):
        from speechbrain.inference.speaker import EncoderClassifier  # 1.0+ path
        import torch
        self._torch = torch
        self.clf = EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb',
            savedir='pretrained_models/spkrec-ecapa-voxceleb',
            run_opts={'device': device})

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """L2-normalized 192-dim embedding from mono float audio (resampled to 16 kHz)."""
        import torchaudio
        t = self._torch.as_tensor(wav, dtype=self._torch.float32).unsqueeze(0)  # [1, T]
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        with self._torch.no_grad():
            emb = self.clf.encode_batch(t).squeeze().cpu().numpy()  # [1,1,192] -> (192,)
        n = np.linalg.norm(emb)
        return (emb / n).astype(np.float32) if n > 0 else emb


def select_narrator(words, full_audio, full_sr, matcher, ref_emb, threshold):
    """Return (narrator_speaker_label, info_str) or (None, reason) to skip the file."""
    if not words:
        return None, 'no aligned words'
    spans = {}
    for w in words:
        if w['speaker'] is None:  # never treat un-diarized audio as a valid narrator
            continue
        spans.setdefault(w['speaker'], []).append((w['start'], w['end']))
    if not spans:
        return None, 'no diarized words'
    if matcher is None or ref_emb is None:
        # Fallback: the dominant speaker (most total speech) is the host.
        best = max(spans, key=lambda s: sum(e - st for st, e in spans[s]))
        secs = sum(e - st for st, e in spans[best])
        return best, f'dominant speaker {best} ({secs:.0f}s)'
    # Reference match: embed up to ~30s of each speaker, cosine-compare to the reference.
    best_label, best_sim = None, -1.0
    for spk, segs in spans.items():
        chunks, total = [], 0.0
        for st, e in sorted(segs, key=lambda se: se[1] - se[0], reverse=True):
            chunks.append(full_audio[int(st * full_sr):int(e * full_sr)])
            total += e - st
            if total >= 30.0:
                break
        if not chunks:
            continue
        sim = float(np.dot(ref_emb, matcher.embed(np.concatenate(chunks), full_sr)))
        if sim > best_sim:
            best_label, best_sim = spk, sim
    if best_sim < threshold:
        return None, f'no speaker matched reference (best sim {best_sim:.2f} < {threshold})'
    return best_label, f'matched {best_label} (cosine {best_sim:.2f})'


# --------------------------------------------------------------------------- #
# Stage 4 — segment narrator words into clips
# --------------------------------------------------------------------------- #
def segment_words(words, min_dur, max_dur, min_chars, max_gap):
    """Pack contiguous narrator words into [min_dur, max_dur] clips ending on . ? !.

    Breaks a clip when: a time gap > max_gap appears (non-contiguous audio — a guest
    spoke, or a long pause), it overflows max_dur (cut at the last sentence end inside),
    or a word ends a sentence and the clip is already >= min_dur.
    """
    segs, cur = [], []

    def _emit_one(chunk):
        s, e = chunk[0]['start'], chunk[-1]['end']
        text = ' '.join(w['word'] for w in chunk).strip()
        if (e - s) >= min_dur and len(text) >= min_chars:
            segs.append((s, e, text))

    def emit(chunk):
        # Only keep clips that actually END a sentence — a non-terminal tail is almost
        # always a mid-sentence ASR cut, which the research says to drop, not force-punctuate.
        if not chunk or not ends_sentence(chunk[-1]['word']):
            return
        # Cap clip length at max_dur: split on interior sentence ends, dropping any
        # final over-long run-on with no usable break (main() would silently drop it anyway).
        start = 0
        while start < len(chunk):
            if chunk[-1]['end'] - chunk[start]['start'] <= max_dur:
                _emit_one(chunk[start:])
                break
            cut = next((i for i in range(len(chunk) - 1, start - 1, -1)
                        if ends_sentence(chunk[i]['word'])
                        and chunk[i]['end'] - chunk[start]['start'] <= max_dur), None)
            if cut is None:  # no sentence end fits within max_dur from here: drop the rest
                break
            _emit_one(chunk[start:cut + 1])
            start = cut + 1

    for w in words:
        if cur and (w['start'] - cur[-1]['end'] > max_gap):
            emit(cur)
            cur = []
        cur.append(w)
        dur = cur[-1]['end'] - cur[0]['start']
        if dur >= max_dur:
            cut = next((i for i in range(len(cur) - 1, -1, -1)
                        if ends_sentence(cur[i]['word'])), None)
            if cut is not None:
                emit(cur[:cut + 1])
                cur = cur[cut + 1:]
            else:
                cur = []  # one run-on longer than max_dur with no sentence end: drop it
        elif ends_sentence(w['word']) and dur >= min_dur:
            emit(cur)
            cur = []
    emit(cur)
    return segs


# --------------------------------------------------------------------------- #
# Clip writing — loudness-normalize and export
# --------------------------------------------------------------------------- #
def normalize_text(raw: str) -> str:
    """Collapse whitespace and force a terminal mark so the model learns to stop."""
    t = ' '.join(raw.split())
    if t and not ends_sentence(t):
        t += '.'
    return t


def write_clip(audio: np.ndarray, sr: int, dst: Path, target_lufs: float, peak_db: float) -> bool:
    """Loudness-normalize (pure gain) then one-directional peak-limit, write PCM-16."""
    import pyloudnorm as pyln
    if len(audio) < int(0.4 * sr):  # shorter than the LUFS gating block
        return False
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(audio)
    if np.isfinite(loud):
        audio = pyln.normalize.loudness(audio, loud, target_lufs)  # gain only -> can clip
    ceiling = 10.0 ** (peak_db / 20.0)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > ceiling:
        audio = audio * (ceiling / peak)  # attenuate only; never boost (would undo LUFS)
    np.clip(audio, -1.0, 1.0, out=audio)
    sf.write(str(dst), audio.astype(np.float32), sr, subtype='PCM_16')
    return True


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #
def write_manifests(out: Path, records: list[dict], speaker: str, val_fraction: float) -> None:
    """Emit LJSpeech (Piper/Coqui), StyleTTS2, and XTTS manifests from the SAME clips."""
    import csv
    # 1) LJSpeech metadata.csv: id|raw|normalized  (no header, no .wav extension)
    with open(out / 'metadata.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        for r in records:
            w.writerow([r['id'], r['raw'], r['norm']])
    # 2) StyleTTS2: filename.wav|text|speaker_id  (integer speaker, .wav included)
    lines = [f"{r['id']}.wav|{r['norm']}|0" for r in records]
    split = max(1, int(len(lines) * val_fraction)) if len(lines) > 10 else 0
    train = lines[split:]
    (out / 'val_list.txt').write_text('\n'.join(lines[:split]) + ('\n' if split else ''), encoding='utf-8')
    (out / 'train_list.txt').write_text('\n'.join(train) + ('\n' if train else ''), encoding='utf-8')
    # 3) Coqui XTTS: header + audio_file|text|speaker_name  (path with .wav)
    with open(out / 'metadata_xtts.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        w.writerow(['audio_file', 'text', 'speaker_name'])
        for r in records:
            w.writerow([f"wavs/{r['id']}.wav", r['norm'], speaker])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob('*') if p.suffix.lower() in AUDIO_EXTS)


def total_seconds(records: list[dict]) -> float:
    return sum(r.get('dur', 0.0) for r in records)


def load_records(records_path: Path) -> list[dict]:
    """Read records.jsonl, tolerating a truncated final line from a kill mid-write."""
    if not records_path.exists():
        return []
    records = []
    for line in records_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f'    ! skipping malformed records.jsonl line (truncated write?): {line[:80]!r}')
    return records


def write_state_atomic(state_path: Path, state: dict) -> None:
    """Write state.json via temp-file + os.replace so a crash can't leave it half-written."""
    tmp = state_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state, indent=0))
    os.replace(tmp, state_path)


def main():
    ap = argparse.ArgumentParser(
        description='Build a clean single-speaker LJSpeech/StyleTTS2/XTTS dataset from podcast audio.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('input', type=Path, help='Audio file or a directory of episodes (recursed).')
    ap.add_argument('-o', '--out', type=Path, default=Path('dataset'), help='Output dataset directory.')
    ap.add_argument('-r', '--reference', type=Path, default=None,
                    help='Clean clip of the target narrator. Strongly recommended for podcasts with '
                         'guests; without it the dominant speaker per file is assumed to be the narrator.')
    ap.add_argument('--speaker', default='host', help='Speaker name used in clip ids and XTTS manifest.')
    ap.add_argument('--sr', type=int, default=24000, help='Output sample rate (StyleTTS2/Kokoro want 24000).')
    ap.add_argument('--min-dur', type=float, default=1.0, help='Min clip seconds.')
    ap.add_argument('--max-dur', type=float, default=15.0, help='Max clip seconds (ideal band 3-11).')
    ap.add_argument('--min-chars', type=int, default=6, help='Drop clips with shorter transcripts.')
    ap.add_argument('--max-gap', type=float, default=1.0,
                    help='Break a clip when narrator words are more than this many seconds apart.')
    ap.add_argument('--sim-threshold', type=float, default=0.50,
                    help='ECAPA cosine threshold to accept a speaker as the narrator (calibrate; 0.5-0.75).')
    ap.add_argument('--lufs', type=float, default=-23.0, help='Target integrated loudness (EBU R128).')
    ap.add_argument('--peak-db', type=float, default=-1.0, help='Post-normalization peak ceiling (dBFS).')
    ap.add_argument('--whisper-model', default='large-v3', help='WhisperX model name.')
    ap.add_argument('--language', default='en', help='Language code (skips detection).')
    ap.add_argument('--diarize-model', default='pyannote/speaker-diarization-3.1',
                    help='Diarization model (needs accepting its HF conditions + a token).')
    ap.add_argument('--batch-size', type=int, default=16, help='WhisperX transcription batch size.')
    ap.add_argument('--min-speakers', type=int, default=1, help='Diarization min speakers.')
    ap.add_argument('--max-speakers', type=int, default=4, help='Diarization max speakers.')
    ap.add_argument('--val-fraction', type=float, default=0.1, help='Fraction of clips for StyleTTS2 val list.')
    ap.add_argument('--demucs', action='store_true',
                    help='Isolate vocals with Demucs before ASR (for music-heavy episodes; slower, '
                         'can degrade clean speech — use selectively).')
    ap.add_argument('--device', default=None, help="Force a torch device ('cuda'|'cpu'|'mps'). Auto by default.")
    ap.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'),
                    help='HuggingFace token for diarization (or set $HF_TOKEN).')
    ap.add_argument('--limit', type=int, default=None, help='Process only the first N files (smoke test).')
    args = ap.parse_args()

    if not shutil.which('ffmpeg'):
        sys.exit('error: ffmpeg not found on PATH (required to decode audio).')
    if not args.hf_token:
        sys.exit('error: a HuggingFace token is required for diarization. Pass --hf-token or set '
                 '$HF_TOKEN, and accept the conditions for ' + args.diarize_model + ' on huggingface.co.')

    inputs = discover_inputs(args.input)
    if args.limit:
        inputs = inputs[:args.limit]
    if not inputs:
        sys.exit(f'error: no audio files found under {args.input}')

    out = args.out
    wavs = out / 'wavs'
    wavs.mkdir(parents=True, exist_ok=True)
    # Advisory lock: a second run over the same out dir would collide on clip ids.
    lock_f = open(out / '.lock', 'w')
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit(f'error: another build_dataset run is already using {out}/ (lock held). '
                 'Wait for it to finish or pick a different --out.')
    state_path = out / 'state.json'
    records_path = out / 'records.jsonl'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'done': []}
    done = set(state['done'])
    records = load_records(records_path)
    next_idx = max((int(r['id'].rsplit('-', 1)[1]) for r in records), default=-1) + 1

    device = pick_device(args.device)
    compute_type = 'float16' if device == 'cuda' else 'int8'
    print(f'Device: {device} (compute_type={compute_type}) | {len(inputs)} input file(s) | '
          f'{len(records)} clips already built ({total_seconds(records)/3600:.2f} h)')

    print('Loading WhisperX models...')
    m = load_whisperx_models(args.whisper_model, device, compute_type, args.language,
                             args.hf_token, args.diarize_model)

    matcher, ref_emb = None, None
    if args.reference:
        print('Loading speaker-embedding model (ECAPA) and reference clip...')
        matcher = SpeakerMatcher(device)
        ref_wav, ref_sr = sf.read(str(args.reference), dtype='float32', always_2d=False)
        if ref_wav.ndim == 2:
            ref_wav = ref_wav.mean(axis=1)
        ref_emb = matcher.embed(ref_wav, ref_sr)

    separator = None
    if args.demucs:
        print('Loading Demucs (htdemucs)...')
        from demucs.api import Separator
        separator = Separator(model='htdemucs', device=device, segment=7, progress=False)

    rec_f = open(records_path, 'a', encoding='utf-8')
    try:
        for n, src in enumerate(inputs, 1):
            key = str(src.resolve())
            if key in done:
                print(f'[{n}/{len(inputs)}] skip (done): {src.name}')
                continue
            print(f'[{n}/{len(inputs)}] {src.name}')
            with tempfile.TemporaryDirectory() as td:
                work = Path(td) / 'work.wav'
                # Buffer this file's records and commit them + mark done together, so a
                # mid-file crash leaves NO partial records (resume reprocesses cleanly,
                # reusing the same next_idx and overwriting any orphan wavs).
                file_records = []
                try:
                    to_working_wav(src, work, args.sr, separator)
                    result, audio16k = transcribe_file(work, m, args.batch_size)
                    result = diarize_and_assign(audio16k, result, m, args.min_speakers, args.max_speakers)

                    words = narrator_words(result)
                    full, full_sr = sf.read(str(work), dtype='float32', always_2d=False)
                    narrator, info = select_narrator(words, full, full_sr, matcher, ref_emb, args.sim_threshold)
                    if narrator is None:
                        print(f'    - {info}; no clips from this file')
                        done.add(key)
                        write_state_atomic(state_path, {'done': sorted(done)})
                        state['done'] = sorted(done)
                        continue
                    print(f'    narrator: {info}')

                    nwords = [w for w in words if w['speaker'] == narrator]
                    clips = segment_words(nwords, args.min_dur, args.max_dur, args.min_chars, args.max_gap)
                    for (s, e, raw) in clips:
                        seg = full[int(s * full_sr):int(e * full_sr)]
                        if (len(seg) / full_sr) < args.min_dur or (len(seg) / full_sr) > args.max_dur:
                            continue
                        norm = normalize_text(raw)
                        if len(norm) < args.min_chars or not ends_sentence(norm):
                            continue
                        cid = f"{args.speaker}-{next_idx:06d}"
                        if not write_clip(seg.copy(), full_sr, wavs / f'{cid}.wav', args.lufs, args.peak_db):
                            continue
                        file_records.append({'id': cid, 'wav': f'wavs/{cid}.wav', 'raw': raw.strip(),
                                             'norm': norm, 'speaker': args.speaker,
                                             'dur': round(len(seg) / full_sr, 3), 'source': src.name})
                        next_idx += 1
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    # ffmpeg failed or hung — mark done so a permanently-bad file isn't retried every resume.
                    stderr = getattr(e, 'stderr', None)
                    detail = stderr.decode()[-200:] if isinstance(stderr, bytes) else e
                    print(f'    ! ffmpeg failed, skipping: {detail}')
                    done.add(key)
                    write_state_atomic(state_path, {'done': sorted(done)})
                    state['done'] = sorted(done)
                    continue
                except Exception as e:  # noqa: BLE001 - one bad file shouldn't kill an hours-long run
                    print(f'    ! processing failed, skipping: {type(e).__name__}: {e}')
                    done.add(key)
                    write_state_atomic(state_path, {'done': sorted(done)})
                    state['done'] = sorted(done)
                    continue

            # File fully processed: flush its records and mark it done in one atomic step.
            for rec in file_records:
                records.append(rec)
                rec_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            rec_f.flush()
            print(f'    + {len(file_records)} clips')
            done.add(key)
            write_state_atomic(state_path, {'done': sorted(done)})
            state['done'] = sorted(done)
    finally:
        rec_f.close()

    write_manifests(out, records, args.speaker, args.val_fraction)
    print(f'\nDone. {len(records)} clips, {total_seconds(records)/3600:.2f} h total in {out}/')
    print('  wavs/  metadata.csv (LJSpeech)  train_list.txt+val_list.txt (StyleTTS2)  metadata_xtts.csv (XTTS)')


if __name__ == '__main__':
    main()
