/** The orchestration module (engine 1.6.0): a graph runtime with LangGraph's execution model
 * (checked against LangGraph in the engine's CI), checkpoints, interrupts and time travel; durable
 * execution with replay and idempotency keys; and the views the site animates. A port of
 * agent_loop_sim/orchestration. */
import type { Obj } from "../data";
import data from "../orchestration_data.json";

export {
  START,
  END,
  REDUCERS,
  SOURCES,
  GraphError,
  InvalidUpdateError,
  NodeFailure,
  Thread,
  validate,
  channel,
  defaultValues,
  ordered,
  reduceValue,
  applyWrites,
  route,
  runOps,
  comparableHistory,
  outcome,
  runSession,
  clone,
  type InvokeOpts,
} from "./graph";
export { CKPT_MS, names, layout, folds, fmt, superstepFrames, timelineFrames, sessionSupersteps } from "./views";
export {
  WORKFLOW,
  TIMEOUT_MS,
  RESTART_MS,
  REPLAY_MS,
  MAX_ATTEMPTS,
  CHOOSE_MODES,
  DURABLE_SCENARIOS,
  runDurable,
  chargeSweep,
  durableScenario,
} from "./durable";

export const GRAPHS: Obj[] = data.graphs as Obj[];
export const SESSIONS: Obj[] = data.sessions as Obj[];

/** A deep copy of a named graph spec. */
export function graph(name: string): Obj {
  for (const g of GRAPHS) if (g.name === name) return JSON.parse(JSON.stringify(g)) as Obj;
  throw new Error(`no graph '${name}'`);
}

/** A deep copy of a named session. */
export function session(name: string): Obj {
  for (const s of SESSIONS) if (s.name === name) return JSON.parse(JSON.stringify(s)) as Obj;
  throw new Error(`no session '${name}'`);
}
