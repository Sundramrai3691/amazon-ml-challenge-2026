"""Deterministic string normalization. Raw values are never overwritten."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

LEGAL_SUFFIXES = (
    "incorporated",
    "corporation",
    "company",
    "limited",
    "llc",
    "llp",
    "l.l.c",
    "l.l.p",
    "inc",
    "corp",
    "ltd",
    "plc",
    "gmbh",
    "sarl",
    "sas",
    "pvt",
    "private",
    "co",
    "pc",
    "pllc",
)

# Internal spelling variants only — not an external address database.
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
    "ste": "suite",
    "apt": "apartment",
    "pkwy": "parkway",
    "ct": "court",
    "cir": "circle",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s).replace(r"\.", r"\.?") for s in LEGAL_SUFFIXES) + r")\b",
    flags=re.IGNORECASE,
)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def unicode_normalize(text: object) -> str:
    return unicodedata.normalize("NFKC", _as_text(text))


def lowercase(text: object) -> str:
    return unicode_normalize(text).casefold()


def normalize_whitespace(text: object) -> str:
    return _WS_RE.sub(" ", _as_text(text)).strip()


def normalize_punctuation(text: object) -> str:
    cleaned = _PUNCT_RE.sub(" ", unicode_normalize(text))
    return normalize_whitespace(cleaned)


def normalize_ampersand(text: object) -> str:
    return normalize_whitespace(_as_text(text).replace("&", " and "))


def compact_alnum(text: object) -> str:
    return _NON_ALNUM_RE.sub("", lowercase(text))


def name_lowercase(text: object) -> str:
    return lowercase(text)


def name_punctuation_normalized(text: object) -> str:
    return lowercase(normalize_punctuation(text))


def name_whitespace_normalized(text: object) -> str:
    return lowercase(normalize_whitespace(unicode_normalize(text)))


def name_ampersand_normalized(text: object) -> str:
    return lowercase(normalize_ampersand(text))


def normalize_name(text: object) -> str:
    """Canonical name key: unicode, lower, ampersand, punctuation, whitespace."""
    return lowercase(normalize_punctuation(normalize_ampersand(unicode_normalize(text))))


def strip_legal_suffixes(text: object) -> str:
    lowered = normalize_name(text)
    prev = None
    while prev != lowered:
        prev = lowered
        lowered = normalize_whitespace(_SUFFIX_RE.sub(" ", lowered))
    return lowered


def name_tokens(text: object) -> tuple[str, ...]:
    core = strip_legal_suffixes(text)
    return tuple(core.split()) if core else ()


def expand_address_abbreviations(text: object) -> str:
    tokens = lowercase(normalize_punctuation(normalize_ampersand(unicode_normalize(text)))).split()
    expanded = [ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    return " ".join(expanded)


def normalize_address(text: object) -> str:
    return expand_address_abbreviations(text)


def address_tokens(text: object) -> tuple[str, ...]:
    norm = normalize_address(text)
    if not norm:
        return ()
    return tuple(norm.split())


def numeric_address_tokens(text: object) -> tuple[str, ...]:
    return tuple(_NUM_RE.findall(unicode_normalize(text)))


def numeric_signature(text: object) -> str:
    """Blocking key from longer digit runs (avoids matching every '1')."""
    nums = [n for n in numeric_address_tokens(text) if len(n) >= 3]
    if not nums:
        return ""
    return "|".join(nums)


@dataclass(frozen=True)
class NameViews:
    raw: str
    unicode_normalized: str
    lowercase: str
    punctuation_normalized: str
    whitespace_normalized: str
    ampersand_normalized: str
    compact: str
    without_legal_suffixes: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class AddressViews:
    raw: str
    unicode_normalized: str
    lowercase: str
    punctuation_normalized: str
    whitespace_normalized: str
    compact: str
    normalized: str
    token_form: tuple[str, ...]
    numeric_tokens: tuple[str, ...]
    alphanumeric: str


def name_views(raw: object) -> NameViews:
    text = _as_text(raw)
    return NameViews(
        raw=text,
        unicode_normalized=unicode_normalize(text),
        lowercase=name_lowercase(text),
        punctuation_normalized=name_punctuation_normalized(text),
        whitespace_normalized=name_whitespace_normalized(text),
        ampersand_normalized=name_ampersand_normalized(text),
        compact=compact_alnum(text),
        without_legal_suffixes=strip_legal_suffixes(text),
        tokens=name_tokens(text),
    )


def address_views(raw: object) -> AddressViews:
    text = _as_text(raw)
    tokens = address_tokens(text)
    punct = lowercase(normalize_punctuation(text))
    return AddressViews(
        raw=text,
        unicode_normalized=unicode_normalize(text),
        lowercase=lowercase(text),
        punctuation_normalized=punct,
        whitespace_normalized=lowercase(normalize_whitespace(unicode_normalize(text))),
        compact=compact_alnum(text),
        normalized=normalize_address(text),
        token_form=tokens,
        numeric_tokens=numeric_address_tokens(text),
        alphanumeric=compact_alnum(normalize_address(text)),
    )
