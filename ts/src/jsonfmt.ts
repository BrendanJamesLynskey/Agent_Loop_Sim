/**
 * JSON text exactly as the Python reference writes it: `json.dumps(x, ensure_ascii=False)`
 * (separators ", " and ": "), the format of the chat templates' `tojson` filter. Strings are
 * escaped as JSON.stringify does, which matches Python with ensure_ascii=False.
 */
export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };

function enc(x: unknown, sortKeys: boolean): string {
  if (x === null) return "null";
  if (typeof x === "boolean") return x ? "true" : "false";
  if (typeof x === "number") {
    if (!Number.isInteger(x)) throw new Error(`float ${x} in prompt JSON (not portable)`);
    return String(x);
  }
  if (typeof x === "string") return JSON.stringify(x);
  if (Array.isArray(x)) return "[" + x.map((v) => enc(v, sortKeys)).join(", ") + "]";
  if (typeof x === "object") {
    let keys = Object.keys(x as object);
    if (sortKeys) keys = keys.slice().sort();
    return (
      "{" +
      keys
        .map((k) => JSON.stringify(k) + ": " + enc((x as Record<string, unknown>)[k], sortKeys))
        .join(", ") +
      "}"
    );
  }
  throw new Error(`unsupported JSON value ${typeof x}`);
}

/** One-line JSON, separators ", " and ": ", keys in insertion order. */
export function dumps(x: unknown): string {
  return enc(x, false);
}

/** The same, keys sorted (Python: dumps of a key-sorted copy). */
export function dumpsSorted(x: unknown): string {
  return enc(x, true);
}
