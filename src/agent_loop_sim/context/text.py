"""Text units for the context module: the pre-tokenizer's pieces, paragraphs and sentences as
ranges of pieces, the BM25 analyser, and number formatting shared with the TS port.

Every chunker cuts only between pieces of the Qwen2.5 pre-tokenizer (``Tokenizer.pieces``), so a
chunk's token count is the sum of its pieces' counts and a boundary is a piece index, the same
in both languages. Sentence and paragraph boundaries are rules over pieces, not a model:
- a paragraph ends after a piece that contains a newline (the corpus joins paragraphs with a
  blank line);
- a sentence ends after a piece ending in ``.``, ``!`` or ``?`` (optionally followed by closing
  quotes or brackets) when the next piece starts with white space and the next visible
  character is an upper-case letter, a digit or an opening quote or bracket.
The rule splits after abbreviations such as "U.S. Army"; it is deterministic, which is what the
parity tests need, not a claim about the best sentence splitter.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import regex

TERM = regex.compile(r"[\p{L}\p{N}]+")
# Lucene's English stop set (EnglishAnalyzer.ENGLISH_STOP_WORDS_SET), 33 words.
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then there these they "
    "this to was will with".split()
)
_SENT_END = regex.compile(r"[.!?][\"'”’)\]]*[\r\n]*$")
_WS_START = regex.compile(r"^[\t\n\x0b\x0c\r \x85\xa0  -     　]")
_OPENER = regex.compile(r"^[\p{Lu}\p{N}\"'“‘(\[]")
_LEAD_WS = regex.compile(r"^[\t\n\x0b\x0c\r \x85\xa0  -     　]+")


def analyse(text: str, stop: bool = True) -> list[str]:
    """BM25's analyser: runs of letters and digits, lower-cased; optionally without stop words.
    No stemming."""
    out = []
    for m in TERM.finditer(text):
        t = m.group(0).lower()
        if stop and t in STOPWORDS:
            continue
        out.append(t)
    return out


def paragraph_starts(text: str, pieces: list[list[int]]) -> list[int]:
    """Piece indices that start a paragraph (always including 0)."""
    out = [0]
    for i in range(1, len(pieces)):
        if "\n" in text[pieces[i - 1][0]:pieces[i - 1][1]]:
            out.append(i)
    return out


def _visible_start(text: str, pieces: list[list[int]], i: int) -> str:
    s = _LEAD_WS.sub("", text[pieces[i][0]:pieces[i][1]])
    if s == "" and i + 1 < len(pieces):
        s = text[pieces[i + 1][0]:pieces[i + 1][1]]
    return s


def sentence_starts(text: str, pieces: list[list[int]]) -> list[int]:
    """Piece indices that start a sentence (always including 0; every paragraph start too)."""
    out = [0]
    for i in range(1, len(pieces)):
        prev = text[pieces[i - 1][0]:pieces[i - 1][1]]
        if "\n" in prev:
            out.append(i)
            continue
        cur = text[pieces[i][0]:pieces[i][1]]
        if _SENT_END.search(prev) and _WS_START.match(cur) and _OPENER.match(_visible_start(text, pieces, i)):
            out.append(i)
    return out


def ranges(starts: list[int], n: int) -> list[list[int]]:
    """Consecutive [start, end) piece ranges from sorted start indices."""
    return [[s, starts[j + 1] if j + 1 < len(starts) else n] for j, s in enumerate(starts)]


def fixed(x: float, d: int) -> str:
    """``x`` with ``d`` decimals, rounding the exact binary value half up, as JavaScript's
    ``toFixed`` does (Python's own formatting rounds exact ties to even)."""
    return str(Decimal(x).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))
