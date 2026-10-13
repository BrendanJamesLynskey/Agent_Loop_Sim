/**
 * Labels (engine 1.10.0): source (user ⊑ trusted ⊑ untrusted) × sensitivity (public ⊑ private), the
 * product lattice. A port of agent_loop_sim/policy/labels.py; see it for the model.
 */
import data from "../policy_data.json";

export const SOURCES: string[] = data.sources;
export const SENSITIVITY: string[] = data.sensitivity;
export const SOURCE_TEXT: Record<string, string> = data.source_text as Record<string, string>;

export type Label = { src: string; sens: string };
export const BOTTOM: Label = { src: "user", sens: "public" };
export const TOP: Label = { src: "untrusted", sens: "private" };

export function label(src: string, sens: string): Label {
  if (!SOURCES.includes(src) || !SENSITIVITY.includes(sens)) throw new Error(`no label (${src}, ${sens})`);
  return { src, sens };
}

const si = (s: string) => SOURCES.indexOf(s);
const ci = (s: string) => SENSITIVITY.indexOf(s);

export function leq(a: Label, b: Label): boolean {
  return si(a.src) <= si(b.src) && ci(a.sens) <= ci(b.sens);
}

export function join(a: Label, b: Label): Label {
  return { src: si(a.src) >= si(b.src) ? a.src : b.src, sens: ci(a.sens) >= ci(b.sens) ? a.sens : b.sens };
}

export function meet(a: Label, b: Label): Label {
  return { src: si(a.src) <= si(b.src) ? a.src : b.src, sens: ci(a.sens) <= ci(b.sens) ? a.sens : b.sens };
}

export function joinAll(labels: Label[]): Label {
  let out: Label = { ...BOTTOM };
  for (const x of labels) out = join(out, x);
  return out;
}

export function allLabels(): Label[] {
  const out: Label[] = [];
  for (const s of SOURCES) for (const c of SENSITIVITY) out.push({ src: s, sens: c });
  return out;
}

export function text(x: Label): string {
  return `${SOURCE_TEXT[x.src]}, ${x.sens}`;
}

export function key(x: Label): string {
  return `${x.src}/${x.sens}`;
}
