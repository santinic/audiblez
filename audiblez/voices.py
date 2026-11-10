# -*- coding: utf-8 -*-
"""Registry of the Kokoro-82M voices supported by Audiblez.

This module is the single source of truth for voice metadata.  A voice name
encodes its language and gender::

    af_sky
    ||  |
    ||  +-- voice name ("Sky")
    |+----- gender: f = female, m = male
    +------ language code: a = American English, b = British English, ...

Public API
----------
Metadata: ``LANGUAGES``, ``VOICES``, ``VOICES_BY_LANG``, ``VOICE_NAMES``,
``DEFAULT_VOICE``.
Helpers: ``is_voice``, ``get_voice``, ``parse_voice``, ``format_voice``,
``format_voices``, ``language_code``.
Legacy: ``voices``, ``flags``, ``flags_win`` and ``available_voices_str`` keep
their original shape, so existing imports keep working.

Flag emoji are used only where the output stream can encode them (and never on
Windows, whose terminals do not render flags).  Set ``AUDIBLEZ_VOICE_FLAGS`` to
``emoji`` or ``text`` to override that detection.
"""

from __future__ import annotations

import difflib
import os
import platform
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

__all__ = [
    # Metadata
    "Language",
    "Voice",
    "LANGUAGES",
    "VOICES",
    "VOICES_BY_LANG",
    "VOICE_NAMES",
    "DEFAULT_VOICE",
    # Helpers
    "is_voice",
    "get_voice",
    "parse_voice",
    "format_voice",
    "format_voices",
    "language_code",
    # Legacy / backwards-compatible views
    "voices",
    "flags",
    "flags_win",
    "available_voices_str",
]

DEFAULT_VOICE = "af_sky"


class Language(NamedTuple):
    """A language supported by the Kokoro model."""

    code: str
    name: str
    flag: str
    plain_name: str  # value historically exposed by ``flags_win``


class Voice(NamedTuple):
    """A Kokoro voice and the language it speaks."""

    name: str
    language: Language

    @property
    def lang_code(self) -> str:
        """Audiblez language code, i.e. the first character of the name."""
        return self.language.code

    @property
    def gender(self) -> str:
        """``"female"`` or ``"male"``, from the second character of the name."""
        return "female" if self.name[1] == "f" else "male"

    @property
    def label(self) -> str:
        """GUI label, e.g. ``"🇺🇸 af_sky"``."""
        return f"{self.language.flag} {self.name}"

    def __str__(self) -> str:
        return self.name


_LANGUAGES: tuple[Language, ...] = (
    Language("a", "American English", "🇺🇸", "american"),
    Language("b", "British English", "🇬🇧", "british"),
    Language("e", "Spanish", "🇪🇸", "spanish"),
    Language("f", "French", "🇫🇷", "french"),
    Language("h", "Hindi", "🇮🇳", "hindi"),
    Language("i", "Italian", "🇮🇹", "italian"),
    Language("j", "Japanese", "🇯🇵", "japanese"),
    Language("p", "Brazilian Portuguese", "🇧🇷", "portuguese"),
    Language("z", "Mandarin Chinese", "🇨🇳", "chinese"),
)

LANGUAGES: Mapping[str, Language] = MappingProxyType(
    {language.code: language for language in _LANGUAGES}
)

# The order of languages and voices is user-visible (CLI help, GUI dropdown),
# so keep it stable.  Voice names come from Kokoro-82M (kokoro==0.9.4).
_VOICE_NAMES: dict[str, tuple[str, ...]] = {
    "a": (
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
    ),
    "b": (
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ),
    "e": ("ef_dora", "em_alex", "em_santa"),
    "f": ("ff_siwis",),
    "h": ("hf_alpha", "hf_beta", "hm_omega", "hm_psi"),
    "i": ("if_sara", "im_nicola"),
    "j": ("jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"),
    "p": ("pf_dora", "pm_alex", "pm_santa"),
    "z": (
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    ),
}

VOICES_BY_LANG: Mapping[str, tuple[Voice, ...]] = MappingProxyType({
    code: tuple(Voice(name, LANGUAGES[code]) for name in names)
    for code, names in _VOICE_NAMES.items()
})

VOICES: tuple[Voice, ...] = tuple(
    voice for group in VOICES_BY_LANG.values() for voice in group
)

VOICE_NAMES: tuple[str, ...] = tuple(voice.name for voice in VOICES)

_VOICES_BY_NAME: Mapping[str, Voice] = MappingProxyType(
    {voice.name: voice for voice in VOICES}
)


def _validate_registry() -> None:
    """Fail fast if a future edit breaks the voice naming contract."""
    seen: set[str] = set()
    for code, names in _VOICE_NAMES.items():
        if code not in LANGUAGES:
            raise RuntimeError(f"Voice group {code!r} has no matching language")
        if not names:
            raise RuntimeError(f"Language {code!r} has no voices")
        for name in names:
            if name in seen:
                raise RuntimeError(f"Duplicate voice {name!r}")
            seen.add(name)
            if len(name) < 4 or name[0] != code:
                raise RuntimeError(f"Voice {name!r} does not start with language code {code!r}")
            if name[1] not in "fm":
                raise RuntimeError(f"Voice {name!r} has an invalid gender marker: {name[1]!r}")
    missing = set(LANGUAGES) - set(_VOICE_NAMES)
    if missing:
        raise RuntimeError(f"Languages without voices: {sorted(missing)}")
    if DEFAULT_VOICE not in seen:
        raise RuntimeError(f"DEFAULT_VOICE {DEFAULT_VOICE!r} is not a known voice")


_validate_registry()


def _detect_use_flags() -> bool:
    """Return True when flag emoji can be displayed in the current terminal."""
    override = os.environ.get("AUDIBLEZ_VOICE_FLAGS", "").strip().lower()
    if override in {"1", "true", "yes", "emoji"}:
        return True
    if override in {"0", "false", "no", "text", "plain"}:
        return False
    if platform.system() == "Windows":
        # Windows terminals render regional-indicator pairs as letters, not flags.
        return False
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None)
        if not encoding:
            continue
        try:
            "🇺🇸".encode(encoding)
        except (LookupError, UnicodeEncodeError):
            return False
    return True


def is_voice(name: object) -> bool:
    """Return True if ``name`` is a known voice (case and surrounding spaces ignored)."""
    return isinstance(name, str) and name.strip().lower() in _VOICES_BY_NAME


def get_voice(name: str) -> Voice:
    """Look up a voice by name.

    Raises:
        TypeError: if ``name`` is not a string.
        ValueError: if the voice is unknown; the message suggests a close match
            and points at ``audiblez --help``.
    """
    if not isinstance(name, str):
        raise TypeError(f"Voice name must be a string, not {type(name).__name__}")
    key = name.strip().lower()
    try:
        return _VOICES_BY_NAME[key]
    except KeyError:
        pass
    suggestion = difflib.get_close_matches(key, VOICE_NAMES, n=1)
    hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
    raise ValueError(
        f"Unknown voice {name!r}.{hint} "
        f"Run 'audiblez --help' to see all {len(VOICES)} voices."
    )


def parse_voice(value: str) -> str:
    """Return the canonical voice name contained in ``value``.

    Accepts plain names, odd casing, and the labels produced by the GUI or by
    ``format_voice``::

        parse_voice("af_sky")               -> "af_sky"
        parse_voice("  AF_SKY  ")           -> "af_sky"
        parse_voice("🇺🇸 af_sky")            -> "af_sky"
        parse_voice("American English af_sky") -> "af_sky"

    Raises:
        TypeError: if ``value`` is not a string.
        ValueError: if no known voice can be found in ``value``.
    """
    if not isinstance(value, str):
        raise TypeError(f"Voice must be a string, not {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError("Voice cannot be empty.")
    candidates = [text]
    if len(text.split()) > 1:
        candidates.append(text.split()[-1])
    for candidate in candidates:
        voice = _VOICES_BY_NAME.get(candidate.strip().lower())
        if voice is not None:
            return voice.name
    # Raise the standard, helpful error (with a "did you mean" hint).
    return get_voice(text).name


def _coerce_voice(voice: str | Voice) -> Voice:
    """Accept a Voice, a plain name, or a GUI label such as ``"🇺🇸 af_sky"``."""
    if isinstance(voice, Voice):
        return voice
    return get_voice(parse_voice(voice))


def language_code(voice: str | Voice) -> str:
    """Return the language code of a voice: ``language_code("af_sky") == "a"``."""
    return _coerce_voice(voice).lang_code


def format_voice(voice: str | Voice, *, use_flags: bool | None = None) -> str:
    """Format one voice, e.g. ``"🇺🇸 af_sky"`` or ``"af_sky"``.

    Accepts plain names and labels previously produced by this function.
    ``use_flags=None`` (the default) auto-detects whether flag emoji are safe
    to print.
    """
    info = _coerce_voice(voice)
    if use_flags is None:
        use_flags = _detect_use_flags()
    return f"{info.language.flag} {info.name}" if use_flags else info.name


def format_voices(*, use_flags: bool | None = None) -> str:
    """Format the full, multi-line voice table shown by ``audiblez --help``."""
    if use_flags is None:
        use_flags = _detect_use_flags()
    lines = []
    for language in _LANGUAGES:
        names = _VOICE_NAMES[language.code]
        label = f"{language.flag} {language.name}" if use_flags else language.name
        noun = "voice" if len(names) == 1 else "voices"
        lines.append(f"  {label} ({len(names)} {noun}):\t{', '.join(names)}")
    return "\n".join(lines)


# --- Backwards-compatible views -------------------------------------------
# These used to be the module's only API; keep their names and values stable.
flags: Mapping[str, str] = MappingProxyType(
    {code: language.flag for code, language in LANGUAGES.items()}
)
flags_win: Mapping[str, str] = MappingProxyType(
    {code: language.plain_name for code, language in LANGUAGES.items()}
)
voices: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {code: tuple(voice.name for voice in group) for code, group in VOICES_BY_LANG.items()}
)
available_voices_str: str = format_voices()
