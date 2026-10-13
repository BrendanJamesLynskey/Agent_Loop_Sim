"""Sandbox layers (since 1.10): which of a tool process's actions each layer stops, and why. The actions are
abstract (read a file in a zone, start a process, connect to a destination), never commands.

- a **container** gives the process its own file system and process space with only the workspace mounted: a
  process it starts stays inside, nothing outside the mounts is visible, and system writes land in the
  container's own throw-away layer; it does not limit the network;
- a **file-system scope** limits file access to the allowed zones;
- a **network allow-list** limits outbound connections to the listed destinations. It cannot tell a download
  from an upload: data sent to an allowed destination still leaves (the residual the label policy covers).
"""
from __future__ import annotations

from typing import Any

from .catalogue import ACTIONS, LAYERS


def stops(layer: dict[str, Any], action: dict[str, Any]) -> str | None:
    """The reason ``layer`` stops ``action``, or None."""
    lid = layer["id"]
    if lid == "container":
        if action["kind"] == "exec":
            return "the new process starts inside the container"
        if action["kind"] == "file" and action["zone"] not in layer["mounts"]:
            if action["op"] == "write":
                return f"the {action['zone']} files are the container's own copy, discarded"
            return f"the {action['zone']} directory is not mounted"
        return None
    if lid == "fs_scope":
        if action["kind"] == "file" and action["zone"] not in layer["allow"]:
            return f"{action['zone']} is outside the allowed file scope"
        return None
    if lid == "egress":
        if action["kind"] == "net" and action["dest"] not in layer["allow"]:
            return f"{action['dest']} is not on the allow-list"
        return None
    raise ValueError(lid)


def matrix(active: list[str]) -> list[dict[str, Any]]:
    """For each action: the active layers that stop it (with reasons), and the outcome."""
    rows = []
    for a in ACTIONS:
        by = []
        for layer in LAYERS:
            if layer["id"] in active:
                r = stops(layer, a)
                if r is not None:
                    by.append({"layer": layer["id"], "reason": r})
        stopped = len(by) > 0
        rows.append({"id": a["id"], "need": a["need"], "stopped": stopped, "by": by,
                     "outcome": ("broken" if a["need"] else "stopped") if stopped else
                                ("works" if a["need"] else "open")})
    return rows


def tally(rows: list[dict[str, Any]]) -> dict[str, int]:
    t = {"works": 0, "broken": 0, "stopped": 0, "open": 0}
    for r in rows:
        t[r["outcome"]] += 1
    return t
