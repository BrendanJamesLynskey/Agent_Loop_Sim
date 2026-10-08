/** The protocol scenarios, generated from the Python reference (protocols_data.json). */
import data from "../protocols_data.json";
import { clone, type Obj } from "./mcp";

export const SCENARIOS: Obj[] = data.scenarios as Obj[];
export const ENGINE_SCENARIOS: Obj[] = data.engine_scenarios as Obj[];

export function protocolScenario(name: string): Obj {
  for (const s of [...SCENARIOS, ...ENGINE_SCENARIOS]) if (s.name === name) return clone(s);
  throw new Error("no protocol scenario '" + name + "'");
}
