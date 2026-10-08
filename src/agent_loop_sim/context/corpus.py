"""The fixed corpus, its labelled questions, the chunking configurations and the shipped vectors.

Files in ``agent_loop_sim/data/context/`` (written once by ``scripts/build_context_data.py``,
offline; see ``manifest.json`` for the model, the source, the licences and every SHA-256):
- ``corpus.json``: six articles of the SQuAD v1.1 development set (CC BY-SA 4.0), paragraphs
  joined by a blank line, and 200 of their questions with the answer's character span;
- ``emb-questions.json``, ``emb-sentences.json``, ``emb-<config>.json``: int8 embeddings
  (all-MiniLM-L6-v2) of the questions, of every sentence, and of every chunk of each config;
- ``rerank.json``: a cross-encoder's scores for each question's candidate pool (the default config's
  RRF top 30 with the BM25 and dense top 10s);
- ``pca.json``: a 2-D PCA projection of the default config's chunks and the questions.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources
from typing import Any

from ..tokenizer import Tokenizer, default_tokenizer
from .chunking import fixed_ranges, recursive_ranges, semantic_ranges, spans
from .text import ranges, sentence_starts
from .vectors import decode

CHUNKERS: dict[str, dict[str, Any]] = {
    "fixed-128": {"kind": "fixed", "size": 128, "overlap": 0},
    "fixed-256": {"kind": "fixed", "size": 256, "overlap": 0},
    "fixed-512": {"kind": "fixed", "size": 512, "overlap": 0},
    "fixed-256-o64": {"kind": "fixed", "size": 256, "overlap": 64},
    "recursive-256": {"kind": "recursive", "size": 256},
    "semantic-256": {"kind": "semantic", "size": 256, "pct": 25},
}
DEFAULT = "recursive-256"


def read_data(name: str) -> str:
    return resources.files("agent_loop_sim").joinpath(f"data/context/{name}").read_text(encoding="utf-8")


def spans_sha256(sp: list[list[int]]) -> str:
    """SHA-256 of a chunk list, ``article:start:end`` lines, so an embedding file can name the
    chunks it embeds and a reader can check they are the chunks it computes."""
    return hashlib.sha256("\n".join(f"{c[0]}:{c[1]}:{c[2]}" for c in sp).encode("utf-8")).hexdigest()


class Corpus:
    def __init__(self, corpus: dict[str, Any], files: dict[str, dict[str, Any]], tok: Tokenizer) -> None:
        self.articles: list[dict[str, Any]] = corpus["articles"]
        self.questions: list[dict[str, Any]] = corpus["questions"]
        self.files = files
        self.tok = tok
        self._pieces: list[list[list[int]]] | None = None
        self._chunks: dict[str, list[dict[str, Any]]] = {}
        self._vecs: dict[str, list[list[int]]] = {}
        self._rerank: list[dict[int, int]] | None = None

    @property
    def pieces(self) -> list[list[list[int]]]:
        if self._pieces is None:
            self._pieces = [self.tok.pieces(a["text"]) for a in self.articles]
        return self._pieces

    def sentences(self) -> list[list[int]]:
        """Every sentence as ``[article, start, end]`` (characters), in corpus order."""
        out = []
        for ai, a in enumerate(self.articles):
            p = self.pieces[ai]
            for sa, sb in ranges(sentence_starts(a["text"], p), len(p)):
                out.append([ai, p[sa][0], p[sb - 1][1]])
        return out

    def vectors(self, name: str) -> list[list[int]]:
        """Decoded int8 vectors of ``questions``, ``sentences`` or a chunking config."""
        if name not in self._vecs:
            f = self.files[f"emb-{name}.json"]
            self._vecs[name] = decode(f["b64"], f["dim"])
        return self._vecs[name]

    def chunks(self, config: str) -> list[dict[str, Any]]:
        """The config's chunks, in corpus order: ``{article, start, end, tokens}``."""
        if config in self._chunks:
            return self._chunks[config]
        c = CHUNKERS[config]
        out: list[dict[str, Any]] = []
        sent_vecs = self.vectors("sentences") if c["kind"] == "semantic" else []
        s0 = 0
        for ai, a in enumerate(self.articles):
            p = self.pieces[ai]
            if c["kind"] == "fixed":
                rs = fixed_ranges(p, 0, len(p), c["size"], c["overlap"])
            elif c["kind"] == "recursive":
                rs = recursive_ranges(a["text"], p, c["size"])
            else:
                ns = len(sentence_starts(a["text"], p))
                rs = semantic_ranges(a["text"], p, c["size"], c["pct"], sent_vecs[s0:s0 + ns])
                s0 += ns
            for st, en, tk in spans(p, rs):
                out.append({"article": ai, "start": st, "end": en, "tokens": tk})
        self._chunks[config] = out
        return out

    def chunk_text(self, ch: dict[str, Any]) -> str:
        return self.articles[ch["article"]]["text"][ch["start"]:ch["end"]]

    def spans_sha256(self, config: str) -> str:
        return spans_sha256([[c["article"], c["start"], c["end"]] for c in self.chunks(config)])

    def rerank_scores(self) -> list[dict[int, int]]:
        """The cross-encoder's scores (logit x 1000, rounded) per question, for the chunks of its
        pool (see ``rerank.json["pool"]``), default config only."""
        if self._rerank is None:
            self._rerank = [{ch: sc for ch, sc in row} for row in self.files["rerank.json"]["scores"]]
        return self._rerank


FILES = ["corpus.json", "emb-questions.json", "emb-sentences.json", "rerank.json", "pca.json"] + [
    f"emb-{c}.json" for c in CHUNKERS
]


@lru_cache(maxsize=1)
def default_corpus() -> Corpus:
    files = {n: json.loads(read_data(n)) for n in FILES}
    return Corpus(files["corpus.json"], files, default_tokenizer())
