"""Three chunkers over an article's pre-tokenizer pieces: fixed, recursive and semantic.

Each returns ``[start, end)`` ranges of piece indices; ``spans`` turns them into character
offsets and token counts. Sizes are Qwen2.5 tokens.

- **fixed** fills a window with whole pieces up to ``size`` tokens; with ``overlap`` the next
  window starts as far back as the last ``overlap`` tokens of the previous one allow.
- **recursive** (after LangChain's RecursiveCharacterTextSplitter): paragraphs; a paragraph over
  ``size`` is split into sentences, a sentence over ``size`` into fixed windows; then the pieces
  are merged greedily, in order, while the merged chunk stays within ``size``.
- **semantic** (after the "semantic chunking" recipe): sentences; the cosine similarity of each
  pair of neighbouring sentences' embeddings (the shipped int8 vectors); a break wherever it is at
  or below the ``pct``-th percentile of the article's similarities (nearest rank, integer
  arithmetic); sentences merged greedily between breaks while within ``size``.
"""
from __future__ import annotations

from .text import paragraph_starts, ranges, sentence_starts
from .vectors import cosine


def _tok(pieces: list[list[int]], a: int, b: int) -> int:
    t = 0
    for i in range(a, b):
        t += pieces[i][2]
    return t


def fixed_ranges(pieces: list[list[int]], a0: int, n: int, size: int, overlap: int = 0) -> list[list[int]]:
    """Fixed windows over pieces[a0:n]."""
    out: list[list[int]] = []
    a = a0
    while a < n:
        t = 0
        b = a
        while b < n and (t + pieces[b][2] <= size or b == a):
            t += pieces[b][2]
            b += 1
        out.append([a, b])
        if b >= n:
            break
        if overlap > 0:
            j = b
            t2 = 0
            while j - 1 > a and t2 + pieces[j - 1][2] <= overlap:
                j -= 1
                t2 += pieces[j][2]
            a = j
        else:
            a = b
    return out


def _merge(pieces: list[list[int]], leaves: list[list[int]], size: int, breaks: set[int] | None = None) -> list[list[int]]:
    out: list[list[int]] = []
    cur: list[int] | None = None
    cur_t = 0
    for leaf in leaves:
        t = _tok(pieces, leaf[0], leaf[1])
        if cur is not None and cur_t + t <= size and not (breaks is not None and leaf[0] in breaks):
            cur = [cur[0], leaf[1]]
            cur_t += t
        else:
            if cur is not None:
                out.append(cur)
            cur = [leaf[0], leaf[1]]
            cur_t = t
    if cur is not None:
        out.append(cur)
    return out


def recursive_ranges(text: str, pieces: list[list[int]], size: int) -> list[list[int]]:
    n = len(pieces)
    sents = sentence_starts(text, pieces)
    leaves: list[list[int]] = []
    for pa, pb in ranges(paragraph_starts(text, pieces), n):
        if _tok(pieces, pa, pb) <= size:
            leaves.append([pa, pb])
            continue
        inner = [s for s in sents if pa <= s < pb]
        for sa, sb in ranges(inner, pb):
            if _tok(pieces, sa, sb) <= size:
                leaves.append([sa, sb])
            else:
                leaves.extend(fixed_ranges(pieces, sa, sb, size))
    return _merge(pieces, leaves, size)


def semantic_breaks(sims: list[float], pct: int) -> tuple[float, list[bool]]:
    """The threshold (nearest-rank ``pct``-th percentile of ``sims``) and, for each neighbouring
    pair, whether the text breaks between them."""
    if not sims:
        return 0.0, []
    srt = sorted(sims)
    thr = srt[(len(srt) - 1) * pct // 100]
    return thr, [s <= thr for s in sims]


def semantic_ranges(text: str, pieces: list[list[int]], size: int, pct: int, sent_vecs: list[list[int]]) -> list[list[int]]:
    n = len(pieces)
    sents = ranges(sentence_starts(text, pieces), n)
    if len(sent_vecs) != len(sents):
        raise ValueError(f"{len(sent_vecs)} sentence vectors for {len(sents)} sentences")
    sims = [cosine(sent_vecs[i], sent_vecs[i + 1]) for i in range(len(sents) - 1)]
    _, brk = semantic_breaks(sims, pct)
    leaves: list[list[int]] = []
    breaks: set[int] = set()
    for i, (sa, sb) in enumerate(sents):
        if i > 0 and brk[i - 1]:
            breaks.add(sa)
        if _tok(pieces, sa, sb) <= size:
            leaves.append([sa, sb])
        else:
            parts = fixed_ranges(pieces, sa, sb, size)
            for p in parts:
                breaks.add(p[0])
            leaves.extend(parts)
            if sb < n:
                breaks.add(sb)
    return _merge(pieces, leaves, size, breaks)


def spans(pieces: list[list[int]], rs: list[list[int]]) -> list[list[int]]:
    """Piece ranges as ``[char_start, char_end, tokens]``."""
    return [[pieces[a][0], pieces[b - 1][1], _tok(pieces, a, b)] for a, b in rs]
