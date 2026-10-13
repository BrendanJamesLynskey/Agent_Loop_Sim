/**
 * A synthetic benchmark (engine 1.8.0): tasks with a latent difficulty and discrimination, agents
 * with a latent skill (two-parameter logistic IRT), and seeded runs whose correlation within a task
 * is a parameter. A port of agent_loop_sim/evals/generator.py; see that file for the model.
 */
import type { Obj } from "../data";
import data from "../evals_data.json";
import { Rng } from "../rng";
import { logistic, normPpf } from "./mathx";

export const POPULATION: Obj = data.population;

const TWO32 = 4294967296.0;

export type Task = { b: number; a: number };

export function population(
  size: number = POPULATION.size,
  difficultySd: number = POPULATION.difficulty_sd,
  aLo: number = POPULATION.a_lo,
  aHi: number = POPULATION.a_hi,
  seed: number = POPULATION.seed,
): Task[] {
  const r = new Rng(seed);
  const out: Task[] = [];
  for (let i = 0; i < size; i++) {
    const u = (r.nextU32() + 0.5) / TWO32;
    const b = difficultySd * normPpf(u);
    const a = r.uniform(aLo, aHi);
    out.push({ b, a });
  }
  return out;
}

let POP: Task[] | null = null;
/** The default population, computed once. */
export function defaultPopulation(): Task[] {
  if (!POP) POP = population();
  return POP;
}

export function successP(task: Task, theta: number): number {
  return logistic(task.a * (theta - task.b));
}

export function mus(tasks: Task[], theta: number): number[] {
  return tasks.map((t) => successP(t, theta));
}

export function trueScore(tasks: Task[], theta: number): number {
  let s = 0.0;
  for (const t of tasks) s = s + successP(t, theta);
  return s / tasks.length;
}

export function sample(nTasks: number, size: number, r: Rng): number[] {
  const out: number[] = [];
  for (let i = 0; i < nTasks; i++) out.push(r.randint(0, size - 1));
  return out;
}

export function runs(muList: number[], m: number, rho: number, r: Rng): number[][] {
  const s = Math.sqrt(rho);
  const out: number[][] = [];
  for (const mu of muList) {
    const z = r.random() < mu ? 1 : 0;
    const row: number[] = [];
    for (let j = 0; j < m; j++) {
      const c = r.random();
      const own = r.random();
      row.push(c < s ? z : own < mu ? 1 : 0);
    }
    out.push(row);
  }
  return out;
}

export function benchmark(theta: number, nTasks: number, m: number, rho: number, seed: number, pop: Task[] | null = null): Obj {
  const p = pop ?? defaultPopulation();
  const idx = sample(nTasks, p.length, new Rng(seed));
  const tasks = idx.map((i) => p[i]!);
  const mu = mus(tasks, theta);
  return { theta, idx, mu, runs: runs(mu, m, rho, new Rng(seed + 1)), rho, m };
}

export function icc(rows: number[][]): number {
  let s = 0.0;
  let n = 0;
  let pairs = 0.0;
  for (const row of rows)
    for (const x of row) {
      s = s + x;
      n += 1;
    }
  const p = s / n;
  for (const row of rows)
    for (let j = 0; j < row.length; j++)
      for (let k = j + 1; k < row.length; k++) pairs = pairs + (row[j]! - p) * (row[k]! - p);
  const m = rows[0]!.length;
  const npairs = (rows.length * m * (m - 1)) / 2;
  return pairs / npairs / (p * (1.0 - p));
}
