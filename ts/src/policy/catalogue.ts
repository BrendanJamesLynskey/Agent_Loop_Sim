/**
 * The abstract world of the policy module (engine 1.10.0): tools, destinations, plans, grants, sandbox
 * actions and layers, and the published measurements; since 1.11.0 the benign suite, the defences and the
 * utility-cost parameters. Generated from the Python reference
 * (agent_loop_sim/policy/catalogue.py) through policy_data.json, one source of truth for both languages.
 */
import type { Obj } from "../data";
import data from "../policy_data.json";

export const TOOLS: Record<string, Obj> = data.tools as Record<string, Obj>;
export const TOOL_ORDER: string[] = data.tool_order;
export const DESTS: Record<string, string> = data.dests as Record<string, string>;
export const DEST_ORDER: string[] = data.dest_order;
export const PLANS: Record<string, Obj> = data.plans as Record<string, Obj>;
export const PLAN_ORDER: string[] = data.plan_order;
export const BROAD: Obj = data.broad as Obj;
export const CIRCLES: Record<string, string> = data.circles as Record<string, string>;
export const CIRCLE_TOOLS: Record<string, string[]> = data.circle_tools as Record<string, string[]>;
export const PRIVILEGE: Obj = data.privilege as Obj;
export const ACTIONS: Obj[] = data.actions as Obj[];
export const LAYERS: Obj[] = data.layers as Obj[];
export const PUBLISHED: Obj[] = data.published as Obj[];
export const DEPTH: Obj = data.depth as Obj;

// since 1.11: the utility-cost model and the frontier
export const SUITE: Obj = data.suite as Obj;
export const DEFENCES: Obj[] = data.defences as Obj[];
export const DEFENCE_ORDER: string[] = data.defence_order;
export const UTILITY: Obj = data.utility as Obj;
export const ADAPTIVE: Obj = data.adaptive as Obj;
export const DUAL_PLANS: string[] = data.dual_plans;
export const DEPTH_DOTS: Obj = data.depth_dots as Obj;
