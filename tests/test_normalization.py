from src.normalize import (
    address_views,
    name_views,
    normalize_address,
    normalize_ampersand,
    normalize_punctuation,
    normalize_whitespace,
    strip_legal_suffixes,
)


def test_deterministic_normalization() -> None:
    raw = "  Acme  &  Sons, Inc.  "
    first = name_views(raw)
    second = name_views(raw)
    assert first == second
    assert first.raw == raw
    assert first.lowercase == raw.casefold()


def test_punctuation_and_whitespace() -> None:
    assert normalize_whitespace("  Foo   Bar\t ") == "Foo Bar"
    assert "llc" not in strip_legal_suffixes("Foo-Bar LLC")
    assert normalize_punctuation("A.B,C!") == "A B C"


def test_ampersand() -> None:
    assert "and" in normalize_ampersand("Foo & Bar").casefold()


def test_legal_suffix_representation() -> None:
    views = name_views("Globex Corporation")
    assert views.without_legal_suffixes == "globex"
    assert strip_legal_suffixes("Globex Inc") == "globex"
    assert strip_legal_suffixes("Globex LLC") == "globex"
    # raw is untouched
    assert views.raw == "Globex Corporation"


def test_address_views_numeric_tokens() -> None:
    views = address_views("123 Main St, Suite 4")
    assert views.raw == "123 Main St, Suite 4"
    assert "street" in views.normalized
    assert "123" in views.numeric_tokens
    assert "4" in views.numeric_tokens
    assert normalize_address("1 Rd") == "1 road"
