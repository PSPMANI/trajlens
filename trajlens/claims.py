"""Automatic grounding: pull the checkable facts out of an agent's final answer.

C3 used to rely only on hand-written ``final_answer_claims``. That works for a curated
corpus but not for a real agent log, where nobody has annotated the claims. This module
extracts the facts that can be checked mechanically (numbers, dates and times) from any
answer, so C3 can flag an ungrounded figure in a trace it has never seen before.

Deliberately conservative: an identifier such as ``RF-88213``, ``B2047`` or
``test_auth.py`` is not treated as a number, and neither is a version string such as
``8.2.1``. Missing a claim is cheaper than a false alarm on a correct run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\S+@\S+")
_PATH = re.compile(r"\S*/\S*")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
# A standalone number: not glued to a letter, digit, underscore or hyphen on the left,
# and not the start of a range, version, date or time on the right. A hyphenated
# quantity such as "30-minute" still counts. Grouping commas in
# both Western (41,105) and Indian (1,00,000) styles are accepted.
_NUMBER = re.compile(
    r"(?<![\w.\-/:@#])"
    r"(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?![\d/@_]|-\d|[.,:]\d)"
)


@dataclass(frozen=True)
class Claim:
    text: str   # the token exactly as written in the answer
    kind: str   # "number" | "date" | "time"
    key: str    # normalised form used for matching
    start: int = -1  # offset in the original text, so a mutation can edit this exact token


def _blank(pattern: re.Pattern, text: str) -> str:
    """Blank out matches with spaces of equal length, so offsets still line up."""
    return pattern.sub(lambda m: " " * len(m.group(0)), text)


def _scrub(text: str) -> str:
    for pattern in (_URL, _EMAIL, _PATH):
        text = _blank(pattern, text)
    return text


def _norm_number(token: str) -> str:
    value = float(token.replace(",", ""))
    return str(int(value)) if value.is_integer() else repr(value)


def extract_claims(text: str) -> list[Claim]:
    """Return the mechanically checkable claims in ``text``, in order of appearance."""
    text = _scrub(text or "")
    claims: list[Claim] = []
    for m in _DATE.finditer(text):
        claims.append(Claim(m.group(0), "date", m.group(0), m.start()))
    for m in _TIME.finditer(text):
        claims.append(Claim(m.group(0), "time", m.group(0), m.start()))
    masked = _blank(_TIME, _blank(_DATE, text))
    for m in _NUMBER.finditer(masked):
        claims.append(Claim(m.group(1), "number", _norm_number(m.group(1)), m.start(1)))
    return sorted(claims, key=lambda c: c.start)


def grounded_keys(*sources: str) -> set[str]:
    """Every claim key that appears anywhere in the given source texts."""
    keys: set[str] = set()
    for src in sources:
        keys.update(c.key for c in extract_claims(src))
    return keys


def ungrounded_claims(answer: str, *sources: str, allow: tuple[str, ...] = ()) -> list[Claim]:
    """Claims in ``answer`` that no source supports.

    ``sources`` are the texts an answer may legitimately draw on: the tool observations
    and the task itself (a number the user supplied is not a hallucination). ``allow``
    lists values the task explicitly permits the agent to derive, e.g. a computed sum.
    """
    keys = grounded_keys(*sources)
    allowed = {c.key for a in allow for c in extract_claims(str(a))}
    return [c for c in extract_claims(answer) if c.key not in keys and c.key not in allowed]
