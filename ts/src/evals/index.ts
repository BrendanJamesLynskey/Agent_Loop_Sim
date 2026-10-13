/** The evals module (engine 1.8.0; graders, trajectories, scaffolds, cost and contamination since 1.9.0): exact special functions, the statistics of evaluating agents
 * (pass@k and pass^k, intervals, clustered errors, paired tests, power), a synthetic IRT benchmark,
 * one SWE-bench-style task end to end, and the chapter studies the Evals site animates. A port of
 * agent_loop_sim/evals. */
export * as mathx from "./mathx";
export * as stats from "./stats";
export * as generator from "./generator";
export * as swe from "./swe";
export * as study from "./study";
export * as judge from "./judge";
export * as trajectory from "./trajectory";
export * as scaffold from "./scaffold";
export * as contamination from "./contamination";
