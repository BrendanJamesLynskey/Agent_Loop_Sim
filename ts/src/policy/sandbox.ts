/**
 * Sandbox layers (engine 1.10.0): which abstract actions each layer stops, and why. A port of
 * agent_loop_sim/policy/sandbox.py.
 */
import type { Obj } from "../data";
import { ACTIONS, LAYERS } from "./catalogue";

export function stops(layer: Obj, action: Obj): string | null {
  const lid = layer.id;
  if (lid === "container") {
    if (action.kind === "exec") return "the new process starts inside the container";
    if (action.kind === "file" && !layer.mounts.includes(action.zone)) {
      if (action.op === "write") return `the ${action.zone} files are the container's own copy, discarded`;
      return `the ${action.zone} directory is not mounted`;
    }
    return null;
  }
  if (lid === "fs_scope") {
    if (action.kind === "file" && !layer.allow.includes(action.zone)) return `${action.zone} files are outside the allowed file scope`;
    return null;
  }
  if (lid === "egress") {
    if (action.kind === "net" && !layer.allow.includes(action.dest)) return `${action.dest} is not on the allow-list`;
    return null;
  }
  throw new Error(lid);
}

export function matrix(active: string[]): Obj[] {
  return ACTIONS.map((a) => {
    const by: Obj[] = [];
    for (const layer of LAYERS) {
      if (!active.includes(layer.id)) continue;
      const r = stops(layer, a);
      if (r !== null) by.push({ layer: layer.id, reason: r });
    }
    const stopped = by.length > 0;
    return {
      id: a.id,
      need: a.need,
      stopped,
      by,
      outcome: stopped ? (a.need ? "broken" : "stopped") : a.need ? "works" : "open",
    };
  });
}

export function tally(rows: Obj[]): Record<string, number> {
  const t: Record<string, number> = { works: 0, broken: 0, stopped: 0, open: 0 };
  for (const r of rows) t[r.outcome] = t[r.outcome]! + 1;
  return t;
}
