"""Deterministic string normalization. Raw values are never overwritten."""

from __future__ import annotations

import re
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
_SUFFIX_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(s).replace(r"\.", r"\.?") for s in LEGAL_SUFFIXES) + r")\b",
    flags=re.IGNORECASE,
)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def lowercase(text: object) -> str:
    return _as_text(text).casefold()


def normalize_whitespace(text: object) -> str:
    return _WS_RE.sub(" ", _as_text(text)).strip()


def normalize_punctuation(text: object) -> str:
    cleaned = _PUNCT_RE.sub(" ", _as_text(text))
    return normalize_whitespace(cleaned)


def normalize_ampersand(text: object) -> str:
    return normalize_whitespace(_as_text(text).replace("&", " and "))


def name_lowercase(text: object) -> str:
    return lowercase(text)


def name_punctuation_normalized(text: object) -> str:
    return lowercase(normalize_punctuation(text))


def name_whitespace_normalized(text: object) -> str:
    return lowercase(normalize_whitespace(text))


def name_ampersand_normalized(text: object) -> str:
    return lowercase(normalize_ampersand(text))


def normalize_name(text: object) -> str:
    """Canonical name key: lower, ampersand, punctuation, whitespace. Keeps legal suffixes."""
    return lowercase(normalize_punctuation(normalize_ampersand(text)))


def strip_legal_suffixes(text: object) -> str:
    lowered = normalize_name(text)
    prev = None
    while prev != lowered:
        prev = lowered
        lowered = normalize_whitespace(_SUFFIX_RE.sub(" ", lowered))
    return lowered


def expand_address_abbreviations(text: object) -> str:
    tokens = lowercase(normalize_punctuation(normalize_ampersand(text))).split()
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
    return tuple(_NUM_RE.findall(_as_text(text)))


@dataclass(frozen=True)
class NameViews:
    raw: str
    lowercase: str
    punctuation_normalized: str
    whitespace_normalized: str
    ampersand_normalized: str
    without_legal_suffixes: str


@dataclass(frozen=True)
class AddressViews:
    raw: str
    normalized: str
    token_form: tuple[str, ...]
    numeric_tokens: tuple[str, ...]


def name_views(raw: object) -> NameViews:
    text = _as_text(raw)
    return NameViews(
        raw=text,
        lowercase=name_lowercase(text),
        punctuation_normalized=name_punctuation_normalized(text),
        whitespace_normalized=name_whitespace_normalized(text),
        ampersand_normalized=name_ampersand_normalized(text),
        without_legal_suffixes=strip_legal_suffixes(text),
    )


def address_views(raw: object) -> AddressViews:
    text = _as_text(raw)
    tokens = address_tokens(text)
    return AddressViews(
        raw=text,
        normalized=normalize_address(text),
        token_form=tokens,
        numeric_tokens=numeric_address_tokens(text),
    )
