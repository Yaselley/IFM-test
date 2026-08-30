"""Scoring contract. Keep ة. Keep Latin. Strip tashkeel / unify alef+yeh."""

from __future__ import annotations

import re
import unicodedata

_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ALEF = str.maketrans({
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
})
_YEH = str.maketrans({"ى": "ي", "ي": "ي"})
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
# Keep letters (Arabic + Latin), digits, and %. Everything else → space.
_KEEP = re.compile(r"[^\w%]+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize_text(text: str | None, *, aggressive: bool = False) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _TASHKEEL.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ALEF).translate(_YEH).translate(_ARABIC_INDIC_DIGITS)
    if aggressive:
        # Optional, not the default. See module docstring.
        text = text.replace("ة", "ه")
    text = _KEEP.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def has_latin_codeswitch(text: str | None) -> bool:
    if not text:
        return False
    return any("LATIN" in unicodedata.name(ch, "") for ch in text if ch.isalpha())


def script_mix(text: str | None) -> dict[str, float]:
    """Fraction of alphabetic characters in Arabic vs Latin."""
    arabic = latin = other = 0
    for ch in text or "":
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if "ARABIC" in name:
            arabic += 1
        elif "LATIN" in name:
            latin += 1
        else:
            other += 1
    total = arabic + latin + other
    if total == 0:
        return {"arabic": 0.0, "latin": 0.0, "other": 0.0, "n": 0}
    return {
        "arabic": arabic / total,
        "latin": latin / total,
        "other": other / total,
        "n": total,
    }


def is_mostly_arabic(text: str | None, threshold: float = 0.6) -> bool:
    mix = script_mix(text)
    return mix["n"] > 0 and mix["arabic"] >= threshold
