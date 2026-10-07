"""Qwen2.5's tokenizer (byte-level BPE), vendored, so every token count is a real one.

The merges file is Qwen2.5-1.5B-Instruct's ``merges.txt`` (Apache-2.0, see
``data/QWEN2.5-LICENSE``), byte for byte. Its 151,387 merges are all the tokenizer needs:
in Qwen2.5's vocabulary the 256 byte tokens come first (ids 0-255, in GPT-2's
bytes-to-unicode order) and the token made by merge ``r`` has id ``256 + r``, so the ids
follow from the merge ranks (``tests/test_tokenizer.py`` checks this against ``vocab.json``
and the Hugging Face tokenizer). The 22 added tokens (``<|im_start|>``, ``<tool_call>`` …)
have ids 151643 onwards.

The pre-tokenizer is Qwen2.5's split regex with two rewrites that keep Python's ``regex``
module and JavaScript's ``u``-flag RegExp in step: the case-insensitive contractions are
spelled out, and ``\\s`` is the explicit Unicode White_Space class (the two engines' own
``\\s`` differ on U+0085 and U+FEFF). Text is NFC-normalised first, as Qwen2.5 does.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache
from importlib import resources

import regex

WS = "\\t\\n\\x0b\\x0c\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000"
PATTERN = (
    "'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])"
    "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"
    "|\\p{N}"
    f"| ?[^{WS}\\p{{L}}\\p{{N}}]+[\\r\\n]*"
    f"|[{WS}]*[\\r\\n]+"
    f"|[{WS}]+(?![^{WS}])"
    f"|[{WS}]+"
)

ADDED_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>",
    "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>",
]
ADDED_BASE = 151643


def bytes_to_unicode() -> tuple[list[str], list[int]]:
    """GPT-2's byte -> printable character map, as a list indexed by byte value, and the
    order in which Qwen2.5 numbers its byte tokens."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    table = [""] * 256
    for b, c in zip(bs, cs):
        table[b] = chr(c)
    return table, bs


class Tokenizer:
    """Encode text to Qwen2.5 token ids; ``count`` is the number of tokens."""

    def __init__(self, merges_text: str) -> None:
        self.byte_char, order = bytes_to_unicode()
        self.ids: dict[str, int] = {}
        for i, b in enumerate(order):
            self.ids[self.byte_char[b]] = i
        self.ranks: dict[tuple[str, str], int] = {}
        r = 0
        for line in merges_text.split("\n"):
            if not line or line.startswith("#version"):
                continue
            a, b = line.split(" ")
            self.ranks[(a, b)] = r
            self.ids[a + b] = 256 + r
            r += 1
        self.n_merges = r
        self.pre = regex.compile(PATTERN)
        self.special = regex.compile("|".join(regex.escape(t) for t in ADDED_TOKENS))
        self.special_id = {t: ADDED_BASE + i for i, t in enumerate(ADDED_TOKENS)}
        self._cache: dict[str, list[int]] = {}

    def _bpe(self, word: str) -> list[int]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        parts = list(word)
        while len(parts) > 1:
            best = -1
            best_rank = self.n_merges
            for i in range(len(parts) - 1):
                rank = self.ranks.get((parts[i], parts[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best = i
            if best < 0:
                break
            a, b = parts[best], parts[best + 1]
            merged: list[str] = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == a and parts[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = merged
        out = [self.ids[p] for p in parts]
        self._cache[word] = out
        return out

    def _encode_plain(self, text: str, out: list[int]) -> None:
        for piece in self.pre.findall(text):
            word = "".join(self.byte_char[b] for b in piece.encode("utf-8"))
            out.extend(self._bpe(word))

    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        out: list[int] = []
        pos = 0
        for m in self.special.finditer(text):
            if m.start() > pos:
                self._encode_plain(text[pos:m.start()], out)
            out.append(self.special_id[m.group(0)])
            pos = m.end()
        if pos < len(text):
            self._encode_plain(text[pos:], out)
        return out

    def count(self, text: str) -> int:
        return len(self.encode(text))


def merges_text() -> str:
    return resources.files("agent_loop_sim").joinpath("data/qwen2.5-merges.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def default_tokenizer() -> Tokenizer:
    return Tokenizer(merges_text())
