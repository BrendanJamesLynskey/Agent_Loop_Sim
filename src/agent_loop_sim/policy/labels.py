"""Labels: what the engine knows about a value instead of its content (since 1.10).

A label has two parts, each a small chain:

- **source** (integrity): ``user`` ⊑ ``trusted`` (a trusted tool, such as the user's own notes) ⊑ ``untrusted``
  (anything a third party could have written: web pages, messages from others, shared tickets);
- **sensitivity** (confidentiality): ``public`` ⊑ ``private``.

Labels form the product lattice of the two chains: a value derived from several values carries their
**join** (the larger source and the larger sensitivity). Data can only become more tainted and more
secret as it flows, never less, unless a trusted step declassifies it (nothing in this model does).
"""
from __future__ import annotations

from typing import Any

SOURCES = ["user", "trusted", "untrusted"]
SENSITIVITY = ["public", "private"]
SOURCE_TEXT = {"user": "user", "trusted": "trusted tool", "untrusted": "untrusted"}

Label = dict[str, str]
BOTTOM: Label = {"src": "user", "sens": "public"}
TOP: Label = {"src": "untrusted", "sens": "private"}


def label(src: str, sens: str) -> Label:
    if src not in SOURCES or sens not in SENSITIVITY:
        raise ValueError(f"no label ({src}, {sens})")
    return {"src": src, "sens": sens}


def leq(a: Label, b: Label) -> bool:
    """a ⊑ b: b is at least as tainted and at least as secret as a."""
    return SOURCES.index(a["src"]) <= SOURCES.index(b["src"]) and \
        SENSITIVITY.index(a["sens"]) <= SENSITIVITY.index(b["sens"])


def join(a: Label, b: Label) -> Label:
    """The least upper bound."""
    s = a["src"] if SOURCES.index(a["src"]) >= SOURCES.index(b["src"]) else b["src"]
    c = a["sens"] if SENSITIVITY.index(a["sens"]) >= SENSITIVITY.index(b["sens"]) else b["sens"]
    return {"src": s, "sens": c}


def meet(a: Label, b: Label) -> Label:
    """The greatest lower bound."""
    s = a["src"] if SOURCES.index(a["src"]) <= SOURCES.index(b["src"]) else b["src"]
    c = a["sens"] if SENSITIVITY.index(a["sens"]) <= SENSITIVITY.index(b["sens"]) else b["sens"]
    return {"src": s, "sens": c}


def join_all(labels: list[Label]) -> Label:
    out = dict(BOTTOM)
    for x in labels:
        out = join(out, x)
    return out


def all_labels() -> list[Label]:
    return [{"src": s, "sens": c} for s in SOURCES for c in SENSITIVITY]


def text(x: Label) -> str:
    return f"{SOURCE_TEXT[x['src']]}, {x['sens']}"


def key(x: Any) -> str:
    return f"{x['src']}/{x['sens']}"
