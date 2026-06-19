#!/usr/bin/env python3
"""Objective intelligibility proxy for the A/B: ASR round-trip WER.

Runs in the MAIN .venv only (uses faster-whisper / CTranslate2 — no torch, so it
can't disturb the ROCm wheel). Transcribes each ab_out/<model>.wav with Whisper and
computes Word Error Rate against the exact passage that was synthesized. LOWER = more
intelligible. This measures intelligibility, NOT naturalness — naturalness is for the
user to judge by listening to the wavs.

    .venv/bin/python tools/score_wer.py            # score every ab_out/*.wav

Note: WER is a sanity floor (did the model say the right words?), not a quality ranking.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'ab_out'

# Must match tools/tts_ab.py PASSAGE exactly.
PASSAGE = (
    "The lighthouse keeper had not spoken to another soul in nineteen days. "
    "Each morning he climbed the spiral stair, wound the great brass mechanism, "
    "and watched the grey Atlantic heave against the rocks below. "
    "It was, he often thought, a strange kind of freedom: to be needed by ships "
    "he would never see, and forgotten by everyone else. That evening, the storm came."
)


# Spelled-out <-> digit map for 0-99 so "nineteen" (in PASSAGE) vs ASR's "19" isn't a phantom
# error. Applied PER TOKEN to BOTH ref and hyp (symmetric), folding every word/number form to a
# single canonical digit string. stdlib-only, no pip dependency.
# LIMITATION: compound tens (21-99, e.g. "twenty-one") are split into ["twenty","one"] by the
# tokenizer and do NOT recombine, so "twenty-one" vs "21" still counts as an error. PASSAGE only
# contains "nineteen", which is covered; building a full number parser isn't worth a dependency.
_ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
         'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
         'eighteen', 'nineteen']
_TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
         'eighty': 80, 'ninety': 90}
_WORD_TO_NUM = {w: str(i) for i, w in enumerate(_ONES)}
_WORD_TO_NUM.update({w: str(n) for w, n in _TENS.items()})


def _canon_num(tok):
    # Fold a spelled-out number token to its digit string; pass other tokens through unchanged.
    return _WORD_TO_NUM.get(tok, tok)


def normalize(s):
    # Replace EVERY run of non-alphanumerics with a SPACE (not nothing) so word boundaries
    # survive: "well-known" -> "well known" (2 words), not "wellknown" (1 phantom word).
    words = re.sub(r'[^a-z0-9]+', ' ', s.lower()).split()
    return [_canon_num(w) for w in words]


def wer(ref_words, hyp_words):
    # Levenshtein over word lists / len(ref).
    n, m = len(ref_words), len(hyp_words)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


def main():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit('faster-whisper not installed. Run: .venv/bin/pip install faster-whisper')

    wavs = sorted(OUT.glob('*.wav'))
    if not wavs:
        sys.exit(f'no wavs in {OUT}')
    print('Loading Whisper (base.en, CPU/int8)...')
    model = WhisperModel('base.en', device='cpu', compute_type='int8')
    ref = normalize(PASSAGE)
    rows = []
    for w in wavs:
        segments, _ = model.transcribe(str(w), language='en')
        hyp_text = ' '.join(s.text for s in segments)
        score = wer(ref, normalize(hyp_text))
        rows.append((w.stem, round(score, 4)))
        print(f'{w.stem:14} WER={score:.3f}  | {hyp_text.strip()[:90]}')
        # fold WER into the model's metrics json if present
        mj = OUT / f'{w.stem}.json'
        if mj.exists():
            d = json.loads(mj.read_text())
            d['wer'] = round(score, 4)
            mj.write_text(json.dumps(d, indent=2))
    print('\nWER summary (lower = more intelligible):')
    for name, s in sorted(rows, key=lambda r: r[1]):
        print(f'  {name:14} {s:.3f}')


if __name__ == '__main__':
    main()
