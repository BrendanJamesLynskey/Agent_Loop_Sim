"""The abstract world the policy module reasons about (since 1.10): tools described only by what they
read, where their output comes from and whether they can carry data out; destinations; small task plans;
sandbox actions and layers; and the published measurements the maths takes its parameter ranges from.
Every name is a placeholder for a kind of tool, not a product; nothing here holds content."""
from __future__ import annotations

from typing import Any

# kind: read (returns a value), transform (the model rewrites values), write (changes local state),
# send (carries data to a destination). out: the label of what a read returns. egress: an argument
# leaves the system the task controls (dests: where it can go). aud / scope: the resource server and the OAuth scope it needs.
TOOLS: dict[str, dict[str, Any]] = {
    "read_notes": {"kind": "read", "out": {"src": "trusted", "sens": "private"}, "egress": False,
                   "aud": "notes-api", "scope": "notes:read", "text": "the user's own notes"},
    "read_inbox": {"kind": "read", "out": {"src": "untrusted", "sens": "private"}, "egress": False,
                   "aud": "mail-api", "scope": "mail:read", "text": "messages other people sent the user"},
    "read_ticket": {"kind": "read", "out": {"src": "untrusted", "sens": "public"}, "egress": False,
                    "aud": "tickets-api", "scope": "tickets:read", "text": "a public issue page anyone can comment on"},
    "search_docs": {"kind": "read", "out": {"src": "trusted", "sens": "public"}, "egress": False,
                    "aud": "docs-api", "scope": "docs:read", "text": "the organisation's published documentation"},
    "fetch_page": {"kind": "read", "out": {"src": "untrusted", "sens": "public"}, "egress": True,
                   "aud": "web", "scope": "web:fetch", "dests": ["web"], "text": "any web page (the address it requests leaves the system)"},
    "summarise": {"kind": "transform", "out": None, "egress": False, "aud": None, "scope": None,
                  "text": "the model rewrites its inputs"},
    "write_file": {"kind": "write", "out": None, "egress": False, "aud": "files", "scope": "files:write",
                   "text": "a file in the task's workspace"},
    "send_message": {"kind": "send", "out": None, "egress": True, "aud": "mail-api", "scope": "mail:send",
                     "dests": ["self", "org", "web"],
                     "text": "a message to a destination"},
}
TOOL_ORDER = ["read_notes", "read_inbox", "read_ticket", "search_docs", "fetch_page", "summarise", "write_file",
              "send_message"]

# Destinations an egress-capable call can name. "self" may read the user's private data.
DESTS = {"self": "the user", "org": "a colleague", "web": "any outside address"}
DEST_ORDER = ["self", "org", "web"]

# Plans: the steps of a task, fixed in advance. An input is the user's request; a call names a tool, the
# values it takes and (for egress) a destination. Labels flow along `args`.
PLANS: dict[str, dict[str, Any]] = {
    "notes_to_file": {"title": "Summarise my notes into a file", "steps": [
        {"op": "input", "out": "q", "sens": "public"},
        {"op": "call", "tool": "read_notes", "args": ["q"], "out": "n"},
        {"op": "call", "tool": "summarise", "args": ["n"], "out": "s"},
        {"op": "call", "tool": "write_file", "args": ["s"], "out": None}]},
    "reply_from_notes": {"title": "Read my inbox and notes, then reply to a colleague", "steps": [
        {"op": "input", "out": "q", "sens": "public"},
        {"op": "call", "tool": "read_notes", "args": ["q"], "out": "n"},
        {"op": "call", "tool": "read_inbox", "args": ["q"], "out": "m"},
        {"op": "call", "tool": "summarise", "args": ["n", "m"], "out": "s"},
        {"op": "call", "tool": "send_message", "args": ["s"], "dest": "org", "out": None}]},
    "notes_then_page": {"title": "Send a colleague a summary of my notes, after looking up a page", "steps": [
        {"op": "input", "out": "q", "sens": "public"},
        {"op": "call", "tool": "read_notes", "args": ["q"], "out": "n"},
        {"op": "call", "tool": "fetch_page", "args": ["q"], "dest": "web", "out": "p"},
        {"op": "call", "tool": "summarise", "args": ["n"], "out": "s"},
        {"op": "call", "tool": "send_message", "args": ["s"], "dest": "org", "out": None}]},
    "inbox_digest": {"title": "Send me a digest of my inbox", "steps": [
        {"op": "input", "out": "q", "sens": "public"},
        {"op": "call", "tool": "read_inbox", "args": ["q"], "out": "m"},
        {"op": "call", "tool": "summarise", "args": ["m"], "out": "s"},
        {"op": "call", "tool": "send_message", "args": ["s"], "dest": "self", "out": None}]},
    "research_page": {"title": "Read a web page and save a summary", "steps": [
        {"op": "input", "out": "q", "sens": "public"},
        {"op": "call", "tool": "fetch_page", "args": ["q"], "dest": "web", "out": "p"},
        {"op": "call", "tool": "summarise", "args": ["p"], "out": "s"},
        {"op": "call", "tool": "write_file", "args": ["s"], "out": None}]},
    "share_ticket": {"title": "Share a public ticket's status with a colleague", "steps": [
        {"op": "input", "out": "q", "sens": "public"},
        {"op": "call", "tool": "read_ticket", "args": ["q"], "out": "t"},
        {"op": "call", "tool": "summarise", "args": ["t"], "out": "s"},
        {"op": "call", "tool": "send_message", "args": ["s"], "dest": "org", "out": None}]},
}
PLAN_ORDER = ["notes_to_file", "reply_from_notes", "notes_then_page", "inbox_digest", "research_page",
              "share_ticket"]

# A grant: the tools and scopes a task holds, the destinations its egress may name, and the destinations
# allowed to read private data.
BROAD: dict[str, Any] = {"tools": list(TOOL_ORDER),
                         "scopes": sorted(t["scope"] for t in TOOLS.values() if t["scope"]),
                         "egress": list(DEST_ORDER), "private_dests": ["self"]}

# Chapter 3's circles: each holds tools that put the agent in it. A tool can sit in two circles.
CIRCLES = {"private": "Private data", "untrusted": "Untrusted input", "outbound": "Outbound channel"}
CIRCLE_TOOLS = {"private": ["read_notes"], "untrusted": ["read_ticket"], "outbound": ["send_message"]}

# Chapter 4: one task, and a call outside it (a step the task does not need).
PRIVILEGE = {"plan": "notes_to_file", "extra": {"op": "call", "tool": "send_message", "args": ["n"], "dest": "web",
                                                "out": None}}

# Sandbox actions: what a tool's process tries to do, abstractly. need: the task (run the tests, install a
# package) needs it.
ACTIONS: list[dict[str, Any]] = [
    {"id": "ws_read", "kind": "file", "op": "read", "zone": "workspace", "need": True,
     "text": "read a file in the workspace"},
    {"id": "ws_write", "kind": "file", "op": "write", "zone": "workspace", "need": True,
     "text": "write a file in the workspace"},
    {"id": "home_read", "kind": "file", "op": "read", "zone": "home", "need": False,
     "text": "read a file in the user's home directory (keys, tokens)"},
    {"id": "sys_write", "kind": "file", "op": "write", "zone": "system", "need": False,
     "text": "write a system file"},
    {"id": "host_exec", "kind": "exec", "op": "run", "zone": "host", "need": False,
     "text": "start a process on the host, outside the sandbox"},
    {"id": "net_registry", "kind": "net", "op": "get", "dest": "registry", "need": True,
     "text": "download a package from the registry"},
    {"id": "net_other", "kind": "net", "op": "send", "dest": "other-host", "need": False,
     "text": "connect to any other host"},
    {"id": "net_registry_put", "kind": "net", "op": "send", "dest": "registry", "need": False,
     "text": "upload data to the allow-listed registry"},
]
LAYERS: list[dict[str, Any]] = [
    {"id": "fs_scope", "name": "File-system scope", "allow": ["workspace"],
     "text": "file access is limited to the workspace"},
    {"id": "container", "name": "Container", "mounts": ["workspace"],
     "text": "the process runs in its own file system and process space; only the workspace is mounted"},
    {"id": "egress", "name": "Network allow-list", "allow": ["registry"],
     "text": "outbound connections only to the package registry"},
]

# Published measurements (each checked at export.arxiv.org). The maths takes parameter ranges only from these.
PUBLISHED: list[dict[str, Any]] = [
    {"id": "injecagent_react_gpt4", "value": 0.24, "text": "24%", "arxiv": "2403.02691",
     "what": "InjecAgent: share of its test cases in which a ReAct-prompted GPT-4 agent was steered by text in a "
             "tool result (base setting; the enhanced setting nearly doubled it)"},
    {"id": "injecagent_cases", "value": 1054, "text": "1,054", "arxiv": "2403.02691",
     "what": "InjecAgent: test cases (17 user tools, 62 attacker tools)"},
    {"id": "spotlighting_base", "value": 0.5, "text": "> 50%", "arxiv": "2403.14720",
     "what": "Spotlighting: attack success rate without the defence (GPT-family models; a lower bound)"},
    {"id": "spotlighting_with", "value": 0.02, "text": "< 2%", "arxiv": "2403.14720",
     "what": "Spotlighting: attack success rate with it (an upper bound)"},
    {"id": "agentdojo_tasks", "value": 97, "text": "97", "arxiv": "2406.13352",
     "what": "AgentDojo: realistic user tasks"},
    {"id": "agentdojo_cases", "value": 629, "text": "629", "arxiv": "2406.13352",
     "what": "AgentDojo: security test cases"},
    {"id": "camel_secure", "value": 0.77, "text": "77%", "arxiv": "2503.18813",
     "what": "CaMeL: AgentDojo tasks solved with provable security"},
    {"id": "camel_undefended", "value": 0.84, "text": "84%", "arxiv": "2503.18813",
     "what": "CaMeL: the same tasks solved by the undefended system"},
]

# Defence in depth: an attempt that would steer the model passes each layer with probability `miss`.
# base: InjecAgent's 24% (range up to Spotlighting's > 50% baseline). Provenance marking's pass-through,
# 0.04, is the ratio of Spotlighting's published bounds (< 2% / > 50%), so at most 0.04 as measured there.
# The other two layers' rates are illustrative. c: the share of attempts every layer misses together.
DEPTH: dict[str, Any] = {
    "base": 0.24, "base_range": [0.24, 0.5],
    "layers": [
        {"id": "marking", "name": "Provenance marking", "miss": 0.04, "illustrative": False},
        {"id": "classifier", "name": "Input classifier", "miss": 0.2, "illustrative": True},
        {"id": "output_check", "name": "Output check", "miss": 0.3, "illustrative": True},
    ],
    "c": [0.0, 0.005, 0.01, 0.02, 0.04],
    "attempts": 1000, "seed": 2601, "trials": 20000,
}

# ── the utility-cost model and the frontier (since 1.11) ────────────────────

# A fixed suite of benign tasks, generated with the shared seeded generator: each reads one to three sources
# (from READ_TOOLS, without repeats), summarises them, and finishes at one sink (or just answers the user).
# `adaptive`: the share of tasks whose last step is decided by what the last read returned (the plan branches
# on it, as in "file it wherever the ticket's category says"). Every task is one the user asked for, so a
# defence that blocks it costs utility. The generator's shares are illustrative.
SUITE: dict[str, Any] = {
    "n": 120, "seed": 2611, "min_reads": 1, "max_reads": 3, "adaptive": 0.25,
    "read_tools": ["read_notes", "read_inbox", "read_ticket", "search_docs", "fetch_page"],
    "sinks": [{"tool": None, "dest": None, "text": "answer the user"},
              {"tool": "write_file", "dest": None, "text": "save a file"},
              {"tool": "send_message", "dest": "self", "text": "send it to the user"},
              {"tool": "send_message", "dest": "org", "text": "send it to a colleague"},
              {"tool": "send_message", "dest": "web", "text": "send it to an outside address"}],
}

# The defences the frontier compares. policy: the label policy is enforced; grant: "broad" or "task" (least
# privilege, only with the policy); mode: "context" (a classic agent) or "dataflow" (plan-then-execute);
# layers: detector layers from DEPTH, with shared blind spot c; approval: which calls a human must approve
# besides the policy's asks ("none", "consequential": every write or send, "every": every tool call);
# approve: whether the policy's asks can be approved (False: they are refused).
DEFENCES: list[dict[str, Any]] = [
    {"id": "none", "name": "No defence", "policy": False, "grant": "broad", "mode": "context", "layers": [],
     "c": 0.0, "approval": "none", "approve": True},
    {"id": "marking", "name": "Provenance marking", "policy": False, "grant": "broad", "mode": "context",
     "layers": ["marking"], "c": 0.0, "approval": "none", "approve": True},
    {"id": "detectors", "name": "Three detector layers", "policy": False, "grant": "broad", "mode": "context",
     "layers": ["marking", "classifier", "output_check"], "c": 0.01, "approval": "none", "approve": True},
    {"id": "approve_all", "name": "Approve every write and send", "policy": False, "grant": "broad",
     "mode": "context", "layers": [], "c": 0.0, "approval": "consequential", "approve": True},
    {"id": "policy", "name": "Label policy", "policy": True, "grant": "broad", "mode": "context", "layers": [],
     "c": 0.0, "approval": "none", "approve": True},
    {"id": "policy_refuse", "name": "Label policy, no approvals", "policy": True, "grant": "broad",
     "mode": "context", "layers": [], "c": 0.0, "approval": "none", "approve": False},
    {"id": "least", "name": "Label policy + least privilege", "policy": True, "grant": "task", "mode": "context",
     "layers": [], "c": 0.0, "approval": "none", "approve": True},
    {"id": "planned", "name": "Plan-then-execute + policy + least privilege", "policy": True, "grant": "task",
     "mode": "dataflow", "layers": [], "c": 0.0, "approval": "none", "approve": True},
]
DEFENCE_ORDER = [d["id"] for d in DEFENCES]

# The cost side's parameters. fp: the share of benign results (or, for the output check, calls) a detector
# layer wrongly flags, which stops the task. Provenance marking's is 0: Spotlighting reports "minimal impact on
# task efficacy" (arXiv 2403.14720); the other two are illustrative. A human's chance of approving a call they
# should have refused is `approval_first_miss` at a task's first prompt, and each further prompt in the same
# task keeps only `approval_decay` of the remaining attention (approval fatigue; illustrative).
# planner_calls: the extra model call a plan-then-execute agent makes to write its plan. Every steered
# attempt is one per task that holds untrusted input, at DEPTH's base rate (InjecAgent). risk_budget: chapter
# 10's illustrative ceiling on a design's risk per task against an adaptive attacker.
UTILITY: dict[str, Any] = {
    "fp": {"marking": 0.0, "classifier": 0.02, "output_check": 0.03},
    "fp_illustrative": {"marking": False, "classifier": True, "output_check": True},
    "approval_first_miss": 0.1, "approval_decay": 0.85,
    "planner_calls": 1, "reps": 400, "seed": 2612,
    "thresholds": ["refuse", "policy", "consequential", "every"],
    "frames": 12, "risk_budget": 0.05,
}

# Chapter 7: a task whose last step is decided by what a read returned (the plan branches on the ticket's
# category), and the plan-then-execute pieces each step runs in.
ADAPTIVE: dict[str, Any] = {"title": "File a ticket's summary wherever its category says", "steps": [
    {"op": "input", "out": "q", "sens": "public"},
    {"op": "call", "tool": "read_ticket", "args": ["q"], "out": "t"},
    {"op": "call", "tool": "summarise", "args": ["t"], "out": "s"},
    {"op": "call", "tool": "write_file", "args": ["s"], "ctrl": ["t"], "out": None}]}
DUAL_PLANS = ["notes_then_page", "research_page", "reply_from_notes", "adaptive"]

# Chapter 6's animation: attempts drawn one by one through the layers.
DEPTH_DOTS: dict[str, Any] = {"n": 400, "seed": 2613}
