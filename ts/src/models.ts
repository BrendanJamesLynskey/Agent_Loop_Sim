/** The model back ends (scripted and replay; recording is Python-only): twin of `models.py`. */
import type { Obj } from "./data";
import { dumps } from "./jsonfmt";

const utf8 = new TextEncoder();

/** FNV-1a, 32-bit, over the UTF-8 bytes. */
export function fnv1a32(text: string): number {
  let h = 0x811c9dc5;
  for (const b of utf8.encode(text)) {
    h ^= b;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function renderStep(step: Obj, style: string): string {
  if ("raw" in step) return typeof step.raw === "string" ? step.raw : step.raw[style];
  if ("final" in step) {
    if (style === "native") return step.final;
    return "Thought: I now know the final answer\nFinal Answer: " + step.final;
  }
  const thought: string = step.thought ?? "";
  const calls = step.calls as Obj[];
  if (style === "native") {
    const parts = thought ? [thought] : [];
    for (const c of calls)
      parts.push("<tool_call>\n" + dumps({ name: c.name, arguments: c.args }) + "\n</tool_call>");
    return parts.join("\n");
  }
  const c = calls[0]!;
  return "Thought: " + thought + "\nAction: " + c.name + "\nAction Input: " + dumps(c.args);
}

export interface Model {
  name: string;
  complete(req: Obj): string;
}

export class ScriptedModel implements Model {
  name = "scripted";
  last: number | null = null;

  constructor(public script: Obj[]) {}

  complete(req: Obj): string {
    let nxt = 0;
    if (this.last !== null) {
      const step = this.script[this.last]!;
      nxt = this.last + 1;
      const prev = req.last as Obj;
      if (prev.kind === "results") {
        const failed = (prev.results as Obj[]).filter((r) => !r.ok);
        if (failed.length > 0) {
          const refused = failed.some((r) => r.kind === "denied" || r.kind === "blocked" || r.kind === "sandboxed");
          const action = refused ? (step.on_denied ?? "skip") : (step.on_error ?? "repeat");
          if (action === "repeat") nxt = this.last;
        }
      }
    }
    this.last = nxt;
    if (nxt >= this.script.length) return renderStep({ final: "I could not finish the task." }, req.style);
    return renderStep(this.script[nxt]!, req.style);
  }
}

export class ReplayMismatch extends Error {}

export class ReplayModel implements Model {
  name = "replay";
  i = 0;

  constructor(public calls: Obj[]) {}

  complete(req: Obj): string {
    if (this.i >= this.calls.length) throw new ReplayMismatch(`the trace has only ${this.calls.length} calls`);
    const rec = this.calls[this.i]!;
    const got = fnv1a32(req.prompt);
    if (got !== rec.prompt_fnv1a32 || req.input_tokens !== rec.prompt_tokens)
      throw new ReplayMismatch(
        `call ${this.i + 1}: prompt fingerprint ${got} / ${req.input_tokens} tokens, ` +
          `recorded ${rec.prompt_fnv1a32} / ${rec.prompt_tokens}`,
      );
    this.i++;
    return rec.text;
  }
}

/** The scripted summariser of summarising compaction (keeps the important facts). */
export function summarise(dropped: Obj[], facts: Obj[]): string {
  const lines = ["Summary of the earlier work (the harness replaced older messages with this):"];
  for (const f of facts)
    if (f.important && dropped.some((m) => (m.content as string).includes(f.text))) lines.push("- " + f.text);
  if (lines.length === 1) lines.push("- nothing important was found yet");
  return lines.join("\n");
}
