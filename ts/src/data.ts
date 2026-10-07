/**
 * The engine's data, generated from the Python reference by scripts/make_fixtures.py (CI
 * checks it is up to date): tool specs, the price and latency tables, the default policy and
 * the scenarios. One source of truth, so the two languages cannot drift.
 */
import data from "./engine_data.json";

export type Obj = { [k: string]: any }; // eslint-disable-line @typescript-eslint/no-explicit-any

export const VERSION: string = data.version;
export const TOOL_SPECS: Record<string, Obj> = data.tools as Record<string, Obj>;
export const PRICES: Record<string, Obj> = data.prices as Record<string, Obj>;
export const LATENCY: Record<string, Obj> = data.latency as Record<string, Obj>;
export const DEFAULT_POLICY: Obj = data.default_policy as Obj;
export const HUMAN_SEED_OFFSET: number = data.human_seed_offset;
export const SCENARIOS: Record<string, Obj> = data.scenarios as Record<string, Obj>;

/** A deep copy of a named scenario. */
export function scenario(name: string): Obj {
  const s = SCENARIOS[name];
  if (!s) throw new Error(`no scenario '${name}'`);
  return JSON.parse(JSON.stringify(s)) as Obj;
}
