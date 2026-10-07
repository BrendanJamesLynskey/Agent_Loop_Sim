/** The fake tools and their world: the TS twin of `agent_loop_sim/tools.py`. */
import type { Obj } from "./data";

export class ToolError extends Error {}
export class CalcError extends Error {}

const WS = " \t\n\r";

export function trim(s: string): string {
  let a = 0;
  let b = s.length;
  while (a < b && WS.includes(s[a]!)) a++;
  while (b > a && WS.includes(s[b - 1]!)) b--;
  return s.slice(a, b);
}

export function splitWs(s: string): string[] {
  const t = trim(s);
  return t ? t.split(/[ \t\n\r]+/) : [];
}

export function normPath(p: string): string {
  p = trim(p);
  while (p.startsWith("./")) p = p.slice(2);
  let i = 0;
  while (i < p.length && p[i] === "/") i++;
  return p.slice(i);
}

const isDigit = (c: string) => c >= "0" && c <= "9";

/** `*` any run (including `/`), `?` one character: two-pointer match over code points. */
export function globMatch(pattern: string, s: string): boolean {
  const p = Array.from(pattern);
  const t = Array.from(s);
  let i = 0;
  let j = 0;
  let star = -1;
  let mark = 0;
  while (j < t.length) {
    if (i < p.length && (p[i] === "?" || p[i] === t[j])) {
      i++;
      j++;
    } else if (i < p.length && p[i] === "*") {
      star = i;
      mark = j;
      i++;
    } else if (star >= 0) {
      i = star + 1;
      mark++;
      j = mark;
    } else return false;
  }
  while (i < p.length && p[i] === "*") i++;
  return i === p.length;
}

const isPlainObject = (x: unknown): x is Obj =>
  typeof x === "object" && x !== null && !Array.isArray(x);

export function validateArgs(tool: Obj, args: unknown): string | null {
  if (!isPlainObject(args)) return "arguments must be a JSON object";
  const props = tool.parameters.properties as Obj;
  for (const k of tool.parameters.required as string[])
    if (!Object.prototype.hasOwnProperty.call(args, k)) return `missing required argument '${k}'`;
  for (const k of Object.keys(args).sort()) {
    if (!Object.prototype.hasOwnProperty.call(props, k)) return `unknown argument '${k}'`;
    if (props[k].type === "string" && typeof args[k] !== "string")
      return `argument '${k}' must be a string`;
  }
  return null;
}

// ── the calculator ────────────────────────────────────────────────────────

function calcTokens(src: string): string[] {
  const chars = Array.from(src);
  const toks: string[] = [];
  let i = 0;
  while (i < chars.length) {
    const c = chars[i]!;
    if (c === " " || c === "\t") i++;
    else if ("+-*/()".includes(c)) {
      toks.push(c);
      i++;
    } else if (isDigit(c) || c === ".") {
      let j = i;
      while (j < chars.length && (isDigit(chars[j]!) || chars[j] === ".")) j++;
      const num = chars.slice(i, j).join("");
      if (!/^[0-9]+(\.[0-9]+)?$/.test(num)) throw new CalcError(`bad number '${num}'`);
      toks.push(num);
      i = j;
    } else throw new CalcError(`unexpected character '${c}'`);
  }
  return toks;
}

export function evaluate(src: string, env: Record<string, number> = {}): number {
  const toks: string[] = [];
  for (const part of src.split(/([A-Za-z_][A-Za-z0-9_]*)/)) {
    if (part === "") continue;
    if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(part)) {
      if (!Object.prototype.hasOwnProperty.call(env, part))
        throw new CalcError(`unknown name '${part}'`);
      toks.push("$" + part);
    } else toks.push(...calcTokens(part));
  }
  let pos = 0;
  const peek = () => (pos < toks.length ? toks[pos]! : null);
  const take = () => {
    const t = peek();
    if (t === null) throw new CalcError("unexpected end of expression");
    pos++;
    return t;
  };
  const factor = (): number => {
    const t = take();
    if (t === "-") return -factor();
    if (t === "(") {
      const v = expr();
      if (take() !== ")") throw new CalcError("missing ')'");
      return v;
    }
    if (t.startsWith("$")) return env[t.slice(1)]!;
    if ("+*/)".includes(t)) throw new CalcError(`unexpected '${t}'`);
    return Number(t);
  };
  const term = (): number => {
    let v = factor();
    while (peek() === "*" || peek() === "/") {
      const op = take();
      const r = factor();
      if (op === "*") v = v * r;
      else {
        if (r === 0) throw new CalcError("division by zero");
        v = v / r;
      }
    }
    return v;
  };
  const expr = (): number => {
    let v = term();
    while (peek() === "+" || peek() === "-") {
      const op = take();
      const r = term();
      v = op === "+" ? v + r : v - r;
    }
    return v;
  };
  if (toks.length === 0) throw new CalcError("empty expression");
  const v = expr();
  if (pos !== toks.length) throw new CalcError(`unexpected '${toks[pos]}'`);
  return v;
}

export function fmtNumber(x: number): string {
  if (!Number.isFinite(x)) throw new CalcError("result is not a finite number");
  const ax = Math.abs(x);
  if (Number.isInteger(x) && ax < 1e15) return String(x === 0 ? 0 : x);
  if (ax < 1e-4 || ax >= 1e15) throw new CalcError("result out of display range");
  return String(x);
}

// ── the test runner ──────────────────────────────────────────────────────

const DEF = /^def ([A-Za-z_][A-Za-z0-9_]*)\(([A-Za-z0-9_, ]*)\):[ \t\r]*$/;
const RET = /^[ \t]+return (.*[^ \t\r])[ \t\r]*$/;
const TEST = /^def (test_[A-Za-z0-9_]*)\(\):[ \t\r]*$/;
const ASSERT =
  /^[ \t]+assert ([A-Za-z_][A-Za-z0-9_]*)\((-?[0-9]+(?:, *-?[0-9]+)*)\) == (-?[0-9]+)[ \t\r]*$/;

function functions(src: string): Map<string, [string[], string]> {
  const out = new Map<string, [string[], string]>();
  const lines = src.split("\n");
  lines.forEach((line, i) => {
    const m = DEF.exec(line);
    if (!m) return;
    const params = m[2]!.split(",").map(trim).filter((p) => p !== "");
    for (const body of lines.slice(i + 1)) {
      const tb = trim(body);
      if (tb === "" || tb.startsWith("#") || tb.startsWith('"""')) continue;
      const r = RET.exec(body);
      if (r) out.set(m[1]!, [params, r[1]!]);
      break;
    }
  });
  return out;
}

const basename = (p: string) => p.split("/").pop()!;

export function runPytest(files: Record<string, string>): [number, string] {
  const impl = new Map<string, [string[], string]>();
  const paths = Object.keys(files).sort();
  for (const path of paths)
    if (path.endsWith(".py") && !basename(path).startsWith("test_"))
      for (const [k, v] of functions(files[path]!)) impl.set(k, v);
  const results: [string, string, string, string | null][] = [];
  for (const path of paths) {
    if (!(basename(path).startsWith("test_") && path.endsWith(".py"))) continue;
    let current: string | null = null;
    for (const line of files[path]!.split("\n")) {
      const t = TEST.exec(line);
      if (t) {
        current = t[1]!;
        results.push([path, current, "passed", null]);
        continue;
      }
      const a = ASSERT.exec(line);
      if (!a || current === null || results[results.length - 1]![2] === "failed") continue;
      const fn = a[1]!;
      const rawArgs = a[2]!;
      const want = parseInt(a[3]!, 10);
      if (!impl.has(fn)) {
        results[results.length - 1] = [path, current, "failed", `NameError: name '${fn}' is not defined`];
        continue;
      }
      const [params, body] = impl.get(fn)!;
      const vals = rawArgs.split(",").map((v) => Number(trim(v)));
      const env: Record<string, number> = {};
      params.forEach((p, k) => {
        if (k < vals.length) env[p] = vals[k]!;
      });
      let got: number;
      try {
        got = evaluate(body, env);
      } catch (e) {
        if (!(e instanceof CalcError)) throw e;
        results[results.length - 1] = [path, current, "failed", `SyntaxError: ${e.message}`];
        continue;
      }
      if (got !== want) {
        let shown: string;
        try {
          shown = fmtNumber(got);
        } catch {
          shown = "nan";
        }
        results[results.length - 1] = [
          path,
          current,
          "failed",
          `assert ${shown} == ${want}\n +  where ${shown} = ${fn}(${rawArgs})`,
        ];
      }
    }
  }
  if (results.length === 0) return [5, "collected 0 items\n\nno tests ran in 0.01s"];
  const lines = [`collected ${results.length} items`, ""];
  const byFile = new Map<string, string>();
  for (const [f, , st] of results) byFile.set(f, (byFile.get(f) ?? "") + (st === "passed" ? "." : "F"));
  for (const [f, marks] of byFile) lines.push(`${f} ${marks}`);
  const failed = results.filter((r) => r[2] === "failed");
  const passed = results.length - failed.length;
  if (failed.length > 0) {
    lines.push("", "FAILURES");
    for (const [f, name, , why] of failed) lines.push(`___ ${name} ___`, `${f}: ${why}`);
    lines.push("", "short test summary info");
    for (const [f, name, , why] of failed)
      lines.push(`FAILED ${f}::${name} - ${String(why).split("\n")[0]}`);
    const summary = passed ? `${failed.length} failed, ${passed} passed` : `${failed.length} failed`;
    lines.push(`${summary} in 0.02s`);
    return [1, lines.join("\n")];
  }
  lines.push(`${passed} passed in 0.01s`);
  return [0, lines.join("\n")];
}

// ── the world ────────────────────────────────────────────────────────────

export class World {
  files: Record<string, string>;
  original: Record<string, string>;
  corpus: Obj[];

  constructor(files: Record<string, string>, corpus: Obj[]) {
    this.files = { ...files };
    this.original = { ...files };
    this.corpus = corpus;
  }

  listFiles(path: string): string {
    let prefix = normPath(path);
    while (prefix.endsWith("/")) prefix = prefix.slice(0, -1);
    const names = Object.keys(this.files)
      .filter((p) => prefix === "" || prefix === "." || p === prefix || p.startsWith(prefix + "/"))
      .sort();
    if (names.length === 0) throw new ToolError(`No such directory: ${path}`);
    return names.join("\n");
  }

  readFile(path: string): string {
    const p = normPath(path);
    if (!Object.prototype.hasOwnProperty.call(this.files, p)) throw new ToolError(`No such file: ${path}`);
    return this.files[p]!;
  }

  writeFile(path: string, content: string): string {
    const p = normPath(path);
    if (p === "") throw new ToolError("empty path");
    this.files[p] = content;
    return `Wrote ${content.split("\n").length} lines to ${p}`;
  }

  editFile(path: string, old: string, neu: string): string {
    const p = normPath(path);
    if (!Object.prototype.hasOwnProperty.call(this.files, p)) throw new ToolError(`No such file: ${path}`);
    const src = this.files[p]!;
    const i = old === "" ? -1 : src.indexOf(old);
    if (i < 0) throw new ToolError(`old_text not found in ${p}`);
    this.files[p] = src.slice(0, i) + neu + src.slice(i + old.length);
    return `Edited ${p}: replaced 1 occurrence`;
  }

  shell(command: string): string {
    let argv = splitWs(command);
    if (argv.length === 0) throw new ToolError("empty command");
    let cmd = argv[0]!;
    if (cmd === "python" && argv[1] === "-m" && argv[2] === "pytest") {
      cmd = "pytest";
      argv = ["pytest", ...argv.slice(3)];
    }
    let code: number;
    let out: string;
    if (cmd === "pytest") [code, out] = runPytest(this.files);
    else if (cmd === "ls") [code, out] = [0, this.listFiles(argv.length > 1 ? argv[1]! : "")];
    else if (cmd === "cat") {
      if (argv.length < 2) [code, out] = [1, "cat: missing operand"];
      else [code, out] = [0, this.readFile(argv[1]!)];
    } else if (cmd === "echo") [code, out] = [0, argv.slice(1).join(" ")];
    else if (cmd === "rm") {
      const targets = argv.slice(1).filter((a) => !a.startsWith("-"));
      const gone: string[] = [];
      for (const t of targets) {
        let pat = t;
        while (pat.endsWith("/")) pat = pat.slice(0, -1);
        for (const p of Object.keys(this.files).sort()) {
          if (p === pat || p.startsWith(pat + "/") || globMatch(pat, p)) {
            delete this.files[p];
            gone.push(p);
          }
        }
      }
      [code, out] = [0, gone.length ? "removed " + gone.join(", ") : ""];
    } else if (cmd === "git" && argv[1] === "status") {
      const changed = Object.keys(this.files)
        .filter((p) => this.original[p] !== this.files[p])
        .sort();
      const deleted = Object.keys(this.original)
        .filter((p) => !Object.prototype.hasOwnProperty.call(this.files, p))
        .sort();
      const rows = changed.map((p) =>
        Object.prototype.hasOwnProperty.call(this.original, p) ? `modified: ${p}` : `new file: ${p}`,
      );
      for (const p of deleted) rows.push(`deleted: ${p}`);
      [code, out] = [0, rows.length ? rows.join("\n") : "nothing to commit, working tree clean"];
    } else [code, out] = [127, `${cmd}: command not found`];
    return `exit code: ${code}\n${out}`;
  }

  search(query: string): string {
    const words = query
      .toLowerCase()
      .split(/[^a-z0-9.]+/)
      .filter((w) => w.length > 1);
    const scored: [number, number][] = [];
    this.corpus.forEach((d, i) => {
      const text = (d.title + " " + d.text).toLowerCase();
      const score = words.filter((w) => text.includes(w)).length;
      if (score > 0) scored.push([-score, i]);
    });
    scored.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (scored.length === 0) return "No results.";
    return scored
      .slice(0, 3)
      .map(([, i]) => {
        const d = this.corpus[i]!;
        return `[${d.id}] ${d.title}\n${d.text}`;
      })
      .join("\n\n");
  }

  calculator(expression: string): string {
    try {
      return fmtNumber(evaluate(expression));
    } catch (e) {
      if (e instanceof CalcError) throw new ToolError(`calculator error: ${e.message}`);
      throw e;
    }
  }
}

export function execute(world: World, name: string, args: Obj): string {
  switch (name) {
    case "list_files":
      return world.listFiles(Object.prototype.hasOwnProperty.call(args, "path") ? args.path : "");
    case "read_file":
      return world.readFile(args.path);
    case "write_file":
      return world.writeFile(args.path, args.content);
    case "edit_file":
      return world.editFile(args.path, args.old_text, args.new_text);
    case "run_shell":
      return world.shell(args.command);
    case "search":
      return world.search(args.query);
    case "calculator":
      return world.calculator(args.expression);
    default:
      throw new ToolError(`unknown tool '${name}'`);
  }
}
