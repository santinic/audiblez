# -*- coding: utf-8 -*-
"""Voice tables, quality metadata, and blend resolution.

Pure-Python and stdlib-only (imports just ``platform`` + a few stdlib helpers) so
``cli.py`` and ``ui.py`` can import it without pulling in torch/kokoro. The actual
synthesis lives in :func:`audiblez.core.build_synthesizer`, which consumes the
``kokoro_voice_string`` / ``voice_lang_code`` resolved here.

Voice blending: Kokoro averages comma-separated voicepacks (e.g. ``"af_heart,af_bella"``),
and BOTH the torch and the Apple-Silicon MLX Kokoro engines implement the *same*
comma-mean semantics. So a weighted blend is encoded as **repetition** in the comma
string (``af_heart`` ×3 + ``am_michael`` ×1 ⇒ 75%/25%), which works identically on
every backend — no tensors, no per-engine special-casing.
"""
import platform
from fractions import Fraction
from functools import reduce
from math import gcd, isfinite

flags = {'a': '🇺🇸', 'b': '🇬🇧', 'e': '🇪🇸', 'f': '🇫🇷', 'h': '🇮🇳', 'i': '🇮🇹', 'j': '🇯🇵', 'p': '🇧🇷', 'z': '🇨🇳'}

flags_win = {'a': 'american', 'b': 'british', 'e': 'spanish', 'f': 'french', 'h': 'hindi', 'i': 'italian',
             'j': 'japanese', 'p': 'portuguese', 'z': 'chinese'}

voices = {
    'a': ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova',
          'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
          'am_michael', 'am_onyx', 'am_puck', 'am_santa'],
    'b': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
    'e': ['ef_dora', 'em_alex', 'em_santa'],
    'f': ['ff_siwis'],
    'h': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'i': ['if_sara', 'im_nicola'],
    'j': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia',
          'zm_yunyang']
}

# Flat set of every known single-voice id (used to validate specs).
ALL_VOICES = {v for lang in voices for v in voices[lang]}

# Authoritative overall grade per voice, from Kokoro's official VOICES.md
# (https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md). Surfaced in the
# CLI/GUI and used to choose sensible defaults. Voices absent here are ungraded
# upstream (the es/pt sets) — shown without a grade.
VOICE_QUALITY = {
    'af_heart': 'A', 'af_bella': 'A-', 'af_nicole': 'B-', 'af_aoede': 'C+', 'af_kore': 'C+',
    'af_sarah': 'C+', 'af_alloy': 'C', 'af_nova': 'C', 'af_sky': 'C-', 'af_jessica': 'D', 'af_river': 'D',
    'am_fenrir': 'C+', 'am_michael': 'C+', 'am_puck': 'C+', 'am_echo': 'D', 'am_eric': 'D',
    'am_liam': 'D', 'am_onyx': 'D', 'am_santa': 'D-', 'am_adam': 'F+',
    'bf_emma': 'B-', 'bf_isabella': 'C', 'bf_alice': 'D', 'bf_lily': 'D',
    'bm_fable': 'C', 'bm_george': 'C', 'bm_lewis': 'D+', 'bm_daniel': 'D',
    'ff_siwis': 'B-',
    'jf_alpha': 'C+', 'jf_gongitsune': 'C', 'jf_tebukuro': 'C', 'jf_nezumi': 'C-', 'jm_kumo': 'C-',
    'hf_alpha': 'C', 'hf_beta': 'C', 'hm_omega': 'C', 'hm_psi': 'C',
    'if_sara': 'C', 'im_nicola': 'C',
    'zf_xiaobei': 'D', 'zf_xiaoni': 'D', 'zf_xiaoxiao': 'D', 'zf_xiaoyi': 'D',
    'zm_yunjian': 'D', 'zm_yunxi': 'D', 'zm_yunxia': 'D', 'zm_yunyang': 'D',
}

_GRADE_ORDER = {'A': 0, 'A-': 1, 'B+': 2, 'B': 3, 'B-': 4, 'C+': 5, 'C': 6, 'C-': 7,
                'D+': 8, 'D': 9, 'D-': 10, 'F+': 11, 'F': 12}

# Hard cap on the total number of comma-repetition parts a blend expands to. The
# engine string repeats each voice ``count`` times and Kokoro torch.stacks one
# voicepack per part, so an unbounded count (e.g. 'af_bella:1000000,af_heart:1')
# would try to materialise ~1M voicepacks (~500GB) and OOM. We rescale the ratio
# down to fit this bound while preserving it as closely as integer rounding allows.
MAX_BLEND_PARTS = 64

# The single best-graded English voice. The old default (af_sky) is only C-.
DEFAULT_VOICE = 'af_heart'

# Best-graded English single voices, surfaced first in the pickers.
RECOMMENDED_VOICES = ['af_heart', 'af_bella', 'bf_emma', 'af_nicole']

# Curated "house narrator" blends. Each is a list of (voice_id, integer_weight);
# weights are encoded as repetition in the engine string (see kokoro_voice_string),
# which both Kokoro engines average identically — so blends run on every backend.
PRESET_BLENDS = {
    'af_warm':        [('af_heart', 1), ('af_bella', 1)],
    'af_expressive':  [('af_heart', 2), ('af_nicole', 1)],
    'ab_storyteller': [('af_heart', 1), ('bf_emma', 1)],
    'am_warm':        [('am_michael', 1), ('am_fenrir', 1)],
    'am_deep':        [('am_puck', 2), ('am_onyx', 1)],
}

# Friendly one-liners for the curated blends (GUI dropdown + CLI help).
PRESET_BLEND_INFO = {
    'af_warm':        'Warm — smooth natural female (Heart + Bella)',
    'af_expressive':  'Expressive — intimate female (Heart + Nicole)',
    'ab_storyteller': 'Storyteller — neutral US/UK female (Heart + Emma)',
    'am_warm':        'Warm male (Michael + Fenrir)',
    'am_deep':        'Deep — rich low male baritone (Puck + Onyx)',
}


def grade_rank(grade):
    """Sort key for a quality grade (best first); ungraded sorts last."""
    return _GRADE_ORDER.get(grade, 99)


def _normalize_weights(weights):
    """Scale positive numeric weights to small integers with the same ratio.

    e.g. [0.6, 0.4] -> [3, 2]; [60, 40] -> [3, 2]; [1, 1] -> [1, 1].

    The weights are first divided by the smallest positive weight before
    ``limit_denominator`` so near-equal ratios (e.g. 33,33,34) stay small instead
    of expanding to ~100 parts. The total part count ``sum(counts)`` is then capped
    at ``MAX_BLEND_PARTS``: anything larger (e.g. 1000000,1) is rescaled
    proportionally down to fit — preserving the ratio as closely as integer
    rounding allows and never dropping a positive weight to zero — so the engine's
    comma-repetition string can never explode into an OOM-sized voicepack stack.
    """
    if any(not isfinite(w) or w <= 0 for w in weights):
        raise ValueError('Voice blend weights must be positive')
    smallest = min(weights)
    fracs = [Fraction(w / smallest).limit_denominator(100) for w in weights]
    if any(f <= 0 for f in fracs):
        # A weight below the precision floor (< smallest/100) rounds to 0 here.
        raise ValueError('Voice blend weights are too small/uneven to represent; '
                         'use ratios within ~100x of each other')
    denom = reduce(lambda a, b: a * b // gcd(a, b), [f.denominator for f in fracs], 1)
    counts = [int(f * denom) for f in fracs]
    g = reduce(gcd, counts)
    counts = [c // g for c in counts]
    total = sum(counts)
    if total > MAX_BLEND_PARTS:
        scale = MAX_BLEND_PARTS / total
        counts = [max(1, round(c * scale)) for c in counts]
        g = reduce(gcd, counts)
        counts = [c // g for c in counts]
    return counts


def parse_voice_spec(spec):
    """Resolve a voice string into a list of ``(voice_id, integer_weight)``.

    Accepts:
      - a single voice id:         ``'af_heart'``              -> ``[('af_heart', 1)]``
      - a named preset blend:      ``'af_warm'``               -> the preset recipe
      - an inline equal blend:     ``'af_bella,af_heart'``     -> ``[('af_bella', 1), ('af_heart', 1)]``
      - an inline weighted blend:  ``'af_bella:60,af_heart:40'`` -> ``[('af_bella', 3), ('af_heart', 2)]``

    Raises ``ValueError`` for unknown voice ids or malformed specs.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError('Voice must be a non-empty string')
    spec = spec.strip()
    if spec in PRESET_BLENDS:
        return [(v, w) for v, w in PRESET_BLENDS[spec]]
    ids, weights = [], []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            vid, _, raw = part.partition(':')
            vid = vid.strip()
            try:
                weight = float(raw)
            except ValueError:
                raise ValueError(f'Invalid blend weight in {part!r} (expected e.g. af_bella:60)') from None
            if not isfinite(weight):
                # 'inf'/'-inf'/'1e400' parse as floats but later blow up in Fraction(...) with an
                # *uncaught* OverflowError; reject here so callers' ValueError handling catches it cleanly.
                raise ValueError(f'Invalid blend weight in {part!r} (expected a finite number)')
        else:
            vid, weight = part, 1.0
        if vid not in ALL_VOICES:
            raise ValueError(
                f'Unknown voice {vid!r}. Use a voice id (e.g. af_heart), a preset blend '
                f'({", ".join(PRESET_BLENDS)}), or a custom blend like "af_bella:60,af_heart:40".')
        # A voice repeated in a blend (e.g. 'af_heart:1,af_heart:1') sums its weights
        # rather than silently double-counting as two separate parts; first-seen order kept.
        if vid in ids:
            weights[ids.index(vid)] += weight
        else:
            ids.append(vid)
            weights.append(weight)
    if not ids:
        raise ValueError(f'No voices found in {spec!r}')
    if len(ids) == 1:
        return [(ids[0], 1)]
    return list(zip(ids, _normalize_weights(weights)))


def is_blend(spec):
    """True if the spec resolves to more than one underlying voice."""
    return len(parse_voice_spec(spec)) > 1


def voice_lang_code(spec):
    """Kokoro language code for a spec: its first component's first character."""
    return parse_voice_spec(spec)[0][0][0]


def kokoro_voice_string(spec):
    """Engine-ready voice argument.

    Returns the bare id for a single voice, or a comma string whose repetition
    encodes the blend weights (e.g. ``'af_heart,af_heart,af_heart,am_michael'``).
    Both the torch and MLX Kokoro engines average comma entries equally, so this one
    string is correct on every backend.
    """
    comps = parse_voice_spec(spec)
    if len(comps) == 1 and comps[0][1] == 1:
        return comps[0][0]
    parts = []
    for vid, weight in comps:
        parts.extend([vid] * weight)
    return ','.join(parts)


def voice_label(spec):
    """Filesystem/display-safe label for a spec (used in chapter wav filenames)."""
    if spec in PRESET_BLENDS:
        return spec
    comps = parse_voice_spec(spec)
    if len(comps) == 1 and comps[0][1] == 1:
        return comps[0][0]
    return 'blend-' + '-'.join(f'{vid}x{w}' if w != 1 else vid for vid, w in comps)


def voice_with_grade(v):
    """Voice id annotated with its grade, e.g. ``'af_heart (A)'``; bare id if ungraded.

    Public so ui.py / core.py can reuse the exact CLI grade-formatting.
    """
    g = VOICE_QUALITY.get(v)
    return f'{v} ({g})' if g else v


def _build_available_voices_str():
    """Human-readable voice catalogue for the CLI epilog, annotated with grades."""
    label_flags = flags_win if platform.system() == 'Windows' else flags
    lines = []
    rec = ', '.join(voice_with_grade(v) for v in RECOMMENDED_VOICES)
    lines.append(f'  recommended (English):\t{rec}')
    lines.append('  preset blends:\t\t' + ', '.join(f'{n} [{PRESET_BLEND_INFO[n]}]' for n in PRESET_BLENDS))
    lines.append("  custom blend example:\t\taf_bella:60,af_heart:40")
    lines.append('')
    lines.append('  all voices (with quality grade):')
    for lang in voices:
        ranked = sorted(voices[lang], key=lambda v: (grade_rank(VOICE_QUALITY.get(v, '')), v))
        lines.append(f'  {label_flags[lang]}:\t{", ".join(voice_with_grade(v) for v in ranked)}')
    return '\n'.join(lines)


available_voices_str = _build_available_voices_str()
