import pytest

from trajlens.claims import extract_claims, ungrounded_claims


def texts(s):
    return [c.text for c in extract_claims(s)]


@pytest.mark.parametrize("text,expected", [
    ("Revenue was INR 41,105 crore, up 7.2%.", ["41,105", "7.2"]),
    ("Indian grouping: 1,00,000 rupees", ["1,00,000"]),
    ("Meet on 2026-08-18 at 14:00", ["2026-08-18", "14:00"]),
    ("a 30-minute call at 3pm", ["30", "3"]),
])
def test_extracts(text, expected):
    assert texts(text) == expected


@pytest.mark.parametrize("text", [
    "refund id RF-88213", "order B2047", "see test_auth.py", "pytest 8.2.1",
    "mail rohit.menon94@gmail.com", "saved to reports/july_2026.png",
    "https://example.com/item/42", "5-7 business days",
])
def test_identifiers_versions_and_ranges_are_not_claims(text):
    assert [c for c in extract_claims(text) if c.kind == "number"] == []


def test_formatting_differences_still_match():
    assert ungrounded_claims("total 41,105", "total 41105") == []


def test_ungrounded_is_reported():
    assert [c.text for c in ungrounded_claims("fare INR 7500", "fare INR 8600")] == ["7500"]


def test_offsets_point_at_the_token():
    s = "a 30-minute call at 3pm on 2026-01-02"
    for c in extract_claims(s):
        assert s[c.start:c.start + len(c.text)] == c.text
