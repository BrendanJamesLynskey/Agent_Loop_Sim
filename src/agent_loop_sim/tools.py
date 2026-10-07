"""The tool registry and deterministic fake tools.

Every tool has a JSON-Schema ``parameters`` object (what the model is shown), a
``read_only`` flag (what the permission modes key on), a ``subject`` argument (what
permission rules and hooks match against: a path or a command), a latency in
milliseconds, and an optional transient-failure rate drawn from the run's seeded RNG.

The tools act on a small in-memory world, so a run is reproducible in Python and in the
TypeScript port alike:

- a file system (a dict of path -> text);
- a shell simulator that understands ``ls``, ``cat``, ``pytest`` (it really evaluates the
  tiny repository's tests, see ``run_pytest``), ``rm``, ``echo`` and ``git status``;
- search over a fixed corpus of documents;
- a calculator (+ - * / and parentheses).
"""
from __future__ import annotations

import re
from typing import Any

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "list_files": {
        "name": "list_files",
        "description": "List the files in the repository, optionally under a directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory to list; omit for all files."}},
            "required": [],
        },
        "read_only": True,
        "subject": "path",
        "latency_ms": 40,
    },
    "read_file": {
        "name": "read_file",
        "description": "Read a text file and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path of the file to read."}},
            "required": ["path"],
        },
        "read_only": True,
        "subject": "path",
        "latency_ms": 60,
    },
    "write_file": {
        "name": "write_file",
        "description": "Create or overwrite a text file with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to write."},
                "content": {"type": "string", "description": "The complete new contents of the file."},
            },
            "required": ["path", "content"],
        },
        "read_only": False,
        "subject": "path",
        "latency_ms": 80,
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Replace the first occurrence of old_text with new_text in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to edit."},
                "old_text": {"type": "string", "description": "Exact text to find."},
                "new_text": {"type": "string", "description": "Text to put in its place."},
            },
            "required": ["path", "old_text", "new_text"],
        },
        "read_only": False,
        "subject": "path",
        "latency_ms": 80,
    },
    "run_shell": {
        "name": "run_shell",
        "description": "Run a shell command in the repository (a sandbox) and return its exit code and output.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command line to run."}},
            "required": ["command"],
        },
        "read_only": False,
        "subject": "command",
        "latency_ms": 900,
    },
    "search": {
        "name": "search",
        "description": "Search the documentation for a query and return the best matching passages.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for."}},
            "required": ["query"],
        },
        "read_only": True,
        "subject": "query",
        "latency_ms": 350,
    },
    "calculator": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression with + - * / and parentheses.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "For example (14.4 / 900) * 1000."}},
            "required": ["expression"],
        },
        "read_only": True,
        "subject": "expression",
        "latency_ms": 5,
    },
    "task": {
        "name": "task",
        "description": "Hand a self-contained sub-task to a sub-agent with a fresh context; it returns a short report.",
        "parameters": {
            "type": "object",
            "properties": {"prompt": {"type": "string", "description": "The sub-task, with everything the sub-agent needs to know."}},
            "required": ["prompt"],
        },
        "read_only": True,
        "subject": "prompt",
        "latency_ms": 0,
    },
}


WS = " \t\n\r"


def trim(s: str) -> str:
    """Strip spaces, tabs and newlines (the same set in both languages)."""
    return s.strip(WS)


def split_ws(s: str) -> list[str]:
    t = trim(s)
    return re.split("[ \t\n\r]+", t) if t else []


def norm_path(p: str) -> str:
    p = trim(p)
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def is_digit(c: str) -> bool:
    return "0" <= c <= "9"


def glob_match(pattern: str, s: str) -> bool:
    """``*`` matches any run of characters (including ``/``), ``?`` exactly one; the rest
    literally. The classic two-pointer wildcard match, over code points."""
    p, t = list(pattern), list(s)
    i = j = 0
    star, mark = -1, 0
    while j < len(t):
        if i < len(p) and (p[i] == "?" or p[i] == t[j]):
            i += 1
            j += 1
        elif i < len(p) and p[i] == "*":
            star, mark = i, j
            i += 1
        elif star >= 0:
            i = star + 1
            mark += 1
            j = mark
        else:
            return False
    while i < len(p) and p[i] == "*":
        i += 1
    return i == len(p)


# ── validation ────────────────────────────────────────────────────────────


def validate_args(tool: dict[str, Any], args: Any) -> str | None:
    """None if ``args`` fits the tool's schema, else a one-line reason."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    props = tool["parameters"]["properties"]
    for k in tool["parameters"]["required"]:
        if k not in args:
            return f"missing required argument '{k}'"
    for k in sorted(args):
        v = args[k]
        if k not in props:
            return f"unknown argument '{k}'"
        want = props[k]["type"]
        if want == "string" and not isinstance(v, str):
            return f"argument '{k}' must be a string"
    return None


# ── the calculator ────────────────────────────────────────────────────────


class CalcError(Exception):
    pass


def _calc_tokens(src: str) -> list[str]:
    toks = []
    i = 0
    while i < len(src):
        c = src[i]
        if c in " \t":
            i += 1
        elif c in "+-*/()":
            toks.append(c)
            i += 1
        elif is_digit(c) or c == ".":
            j = i
            while j < len(src) and (is_digit(src[j]) or src[j] == "."):
                j += 1
            num = src[i:j]
            if re.fullmatch(r"[0-9]+(\.[0-9]+)?", num) is None:
                raise CalcError(f"bad number '{num}'")
            toks.append(num)
            i = j
        else:
            raise CalcError(f"unexpected character '{c}'")
    return toks


def evaluate(src: str, env: dict[str, float] | None = None) -> float:
    """Evaluate + - * / ( ) over decimal numbers (and named variables, for the test
    runner). Floats throughout, so Python and JavaScript agree bit for bit."""
    env = env or {}
    toks: list[str] = []
    for part in re.split(r"([A-Za-z_][A-Za-z0-9_]*)", src):
        if part == "":
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            if part not in env:
                raise CalcError(f"unknown name '{part}'")
            toks.append("$" + part)
        else:
            toks.extend(_calc_tokens(part))
    pos = [0]

    def peek() -> str | None:
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take() -> str:
        t = peek()
        if t is None:
            raise CalcError("unexpected end of expression")
        pos[0] += 1
        return t

    def factor() -> float:
        t = take()
        if t == "-":
            return -factor()
        if t == "(":
            v = expr()
            if take() != ")":
                raise CalcError("missing ')'")
            return v
        if t.startswith("$"):
            return env[t[1:]]
        if t in "+*/)":
            raise CalcError(f"unexpected '{t}'")
        return float(t)

    def term() -> float:
        v = factor()
        while peek() in ("*", "/"):
            op = take()
            r = factor()
            if op == "*":
                v = v * r
            else:
                if r == 0:
                    raise CalcError("division by zero")
                v = v / r
        return v

    def expr() -> float:
        v = term()
        while peek() in ("+", "-"):
            op = take()
            r = term()
            v = v + r if op == "+" else v - r
        return v

    if not toks:
        raise CalcError("empty expression")
    v = expr()
    if pos[0] != len(toks):
        raise CalcError(f"unexpected '{toks[pos[0]]}'")
    return v


def fmt_number(x: float) -> str:
    """A result as text, the same in both languages: integers plainly, other values in
    shortest round-trip form within [1e-4, 1e15), else refused."""
    if x != x or x in (float("inf"), float("-inf")):
        raise CalcError("result is not a finite number")
    ax = abs(x)
    if x == int(x) and ax < 1e15:
        return str(int(x))
    if ax < 1e-4 or ax >= 1e15:
        raise CalcError("result out of display range")
    return repr(x)


# ── the tiny repository's test runner ─────────────────────────────────────

_DEF = re.compile(r"def ([A-Za-z_][A-Za-z0-9_]*)\(([A-Za-z0-9_, ]*)\):[ \t\r]*")
_RET = re.compile(r"[ \t]+return (.*[^ \t\r])[ \t\r]*")
_TEST = re.compile(r"def (test_[A-Za-z0-9_]*)\(\):[ \t\r]*")
_ASSERT = re.compile(r"[ \t]+assert ([A-Za-z_][A-Za-z0-9_]*)\((-?[0-9]+(?:, *-?[0-9]+)*)\) == (-?[0-9]+)[ \t\r]*")


def _functions(src: str) -> dict[str, tuple[list[str], str]]:
    """``def f(a, b):`` followed by ``return <expression>``: name -> (params, expression)."""
    out: dict[str, tuple[list[str], str]] = {}
    lines = src.split("\n")
    for i, line in enumerate(lines):
        m = _DEF.fullmatch(line)
        if m is None:
            continue
        params = [trim(p) for p in m.group(2).split(",") if trim(p) != ""]
        for body in lines[i + 1:]:
            if trim(body) == "" or trim(body).startswith("#") or trim(body).startswith('"""'):
                continue
            r = _RET.fullmatch(body)
            if r is not None:
                out[m.group(1)] = (params, r.group(1))
            break
    return out


def run_pytest(files: dict[str, str]) -> tuple[int, str]:
    """Run ``test_*.py`` against the module they import, pytest-style output.

    Understands what the tiny repository uses: tests made of ``assert f(x, y) == z`` and
    functions whose body is one ``return`` of an arithmetic expression in their parameters.
    """
    impl: dict[str, tuple[list[str], str]] = {}
    for path in sorted(files):
        if path.endswith(".py") and not path.split("/")[-1].startswith("test_"):
            impl.update(_functions(files[path]))
    results: list[tuple[str, str, str, str | None]] = []  # (file, test, status, failure)
    for path in sorted(files):
        if not (path.split("/")[-1].startswith("test_") and path.endswith(".py")):
            continue
        current = None
        for line in files[path].split("\n"):
            t = _TEST.fullmatch(line)
            if t is not None:
                current = t.group(1)
                results.append((path, current, "passed", None))
                continue
            a = _ASSERT.fullmatch(line)
            if a is None or current is None or results[-1][2] == "failed":
                continue
            fn, raw_args, want = a.group(1), a.group(2), int(a.group(3))
            if fn not in impl:
                results[-1] = (path, current, "failed", f"NameError: name '{fn}' is not defined")
                continue
            params, body = impl[fn]
            vals = [float(trim(v)) for v in raw_args.split(",")]
            try:
                got = evaluate(body, dict(zip(params, vals)))
            except CalcError as e:
                results[-1] = (path, current, "failed", f"SyntaxError: {e}")
                continue
            if got != want:
                try:
                    shown = fmt_number(got)
                except CalcError:
                    shown = "nan"
                results[-1] = (
                    path,
                    current,
                    "failed",
                    f"assert {shown} == {want}\n +  where {shown} = {fn}({raw_args})",
                )
    if not results:
        return 5, "collected 0 items\n\nno tests ran in 0.01s"
    lines = [f"collected {len(results)} items", ""]
    by_file: dict[str, str] = {}
    for f, _, st, _ in results:
        by_file[f] = by_file.get(f, "") + ("." if st == "passed" else "F")
    for f, marks in by_file.items():
        lines.append(f"{f} {marks}")
    failed = [r for r in results if r[2] == "failed"]
    passed = len(results) - len(failed)
    if failed:
        lines += ["", "FAILURES"]
        for f, name, _, why in failed:
            lines.append(f"___ {name} ___")
            lines.append(f"{f}: {why}")
        lines += ["", "short test summary info"]
        for f, name, _, why in failed:
            lines.append(f"FAILED {f}::{name} - {str(why).split(chr(10))[0]}")
        summary = f"{len(failed)} failed, {passed} passed" if passed else f"{len(failed)} failed"
        lines.append(f"{summary} in 0.02s")
        return 1, "\n".join(lines)
    lines.append(f"{passed} passed in 0.01s")
    return 0, "\n".join(lines)


# ── the world the tools act on ────────────────────────────────────────────


class World:
    """The file system, the corpus and the original files (for ``git status``)."""

    def __init__(self, files: dict[str, str], corpus: list[dict[str, str]]) -> None:
        self.files = dict(files)
        self.original = dict(files)
        self.corpus = corpus

    def list_files(self, path: str) -> str:
        prefix = norm_path(path).rstrip("/")
        names = sorted(p for p in self.files if prefix in ("", ".") or p == prefix or p.startswith(prefix + "/"))
        if not names:
            raise ToolError(f"No such directory: {path}")
        return "\n".join(names)

    def read_file(self, path: str) -> str:
        p = norm_path(path)
        if p not in self.files:
            raise ToolError(f"No such file: {path}")
        return self.files[p]

    def write_file(self, path: str, content: str) -> str:
        p = norm_path(path)
        if p == "":
            raise ToolError("empty path")
        self.files[p] = content
        n = len(content.split("\n"))
        return f"Wrote {n} lines to {p}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = norm_path(path)
        if p not in self.files:
            raise ToolError(f"No such file: {path}")
        if old == "" or old not in self.files[p]:
            raise ToolError(f"old_text not found in {p}")
        self.files[p] = self.files[p].replace(old, new, 1)
        return f"Edited {p}: replaced 1 occurrence"

    def shell(self, command: str) -> str:
        argv = split_ws(command)
        if not argv:
            raise ToolError("empty command")
        cmd = argv[0]
        if cmd == "python" and argv[1:3] == ["-m", "pytest"]:
            cmd, argv = "pytest", ["pytest"] + argv[3:]
        if cmd == "pytest":
            code, out = run_pytest(self.files)
        elif cmd == "ls":
            code, out = 0, self.list_files(argv[1] if len(argv) > 1 else "")
        elif cmd == "cat":
            if len(argv) < 2:
                code, out = 1, "cat: missing operand"
            else:
                code, out = 0, self.read_file(argv[1])
        elif cmd == "echo":
            code, out = 0, " ".join(argv[1:])
        elif cmd == "rm":
            targets = [a for a in argv[1:] if not a.startswith("-")]
            gone = []
            for t in targets:
                pat = t.rstrip("/")
                for p in sorted(self.files):
                    if p == pat or p.startswith(pat + "/") or glob_match(pat, p):
                        del self.files[p]
                        gone.append(p)
            code, out = 0, ("removed " + ", ".join(gone)) if gone else ""
        elif cmd == "git" and argv[1:2] == ["status"]:
            changed = sorted(p for p in self.files if self.original.get(p) != self.files[p])
            deleted = sorted(p for p in self.original if p not in self.files)
            rows = [f"modified: {p}" if p in self.original else f"new file: {p}" for p in changed]
            rows += [f"deleted: {p}" for p in deleted]
            code, out = 0, "\n".join(rows) if rows else "nothing to commit, working tree clean"
        else:
            code, out = 127, f"{cmd}: command not found"
        return f"exit code: {code}\n{out}"

    def search(self, query: str) -> str:
        words = [w for w in re.split(r"[^a-z0-9.]+", query.lower()) if len(w) > 1]
        scored = []
        for i, d in enumerate(self.corpus):
            text = (d["title"] + " " + d["text"]).lower()
            score = sum(1 for w in words if w in text)
            if score > 0:
                scored.append((-score, i))
        scored.sort()
        if not scored:
            return "No results."
        out = []
        for _, i in scored[:3]:
            d = self.corpus[i]
            out.append(f"[{d['id']}] {d['title']}\n{d['text']}")
        return "\n\n".join(out)

    def calculator(self, expression: str) -> str:
        try:
            return fmt_number(evaluate(expression))
        except CalcError as e:
            raise ToolError(f"calculator error: {e}") from None


class ToolError(Exception):
    """A tool's own (permanent) error: shown to the model as the result."""


def execute(world: World, name: str, args: dict[str, Any]) -> str:
    """Run one fake tool. Raises ToolError for errors the model should see."""
    if name == "list_files":
        return world.list_files(args.get("path", ""))
    if name == "read_file":
        return world.read_file(args["path"])
    if name == "write_file":
        return world.write_file(args["path"], args["content"])
    if name == "edit_file":
        return world.edit_file(args["path"], args["old_text"], args["new_text"])
    if name == "run_shell":
        return world.shell(args["command"])
    if name == "search":
        return world.search(args["query"])
    if name == "calculator":
        return world.calculator(args["expression"])
    raise ToolError(f"unknown tool '{name}'")
