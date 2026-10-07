/** Cache, cost and latency of a model call: the TS twin of `agent_loop_sim/accounting.py`. */
import type { Obj } from "./data";

export function lcp(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}

export class PromptCache {
  entries: { ids: number[]; t: number }[] = [];

  constructor(
    public enabled: boolean,
    public minTokens: number,
    public block: number,
    public ttlMs: number,
  ) {}

  lookup(ids: number[], now: number): number {
    if (!this.enabled) return 0;
    this.entries = this.entries.filter((e) => now - e.t <= this.ttlMs);
    let best = 0;
    let bestEntry: { ids: number[]; t: number } | null = null;
    for (const e of this.entries) {
      const n = lcp(e.ids, ids);
      if (n > best) {
        best = n;
        bestEntry = e;
      }
    }
    const cached = Math.floor(best / this.block) * this.block;
    if (cached < this.minTokens) return 0;
    if (bestEntry !== null) bestEntry.t = now;
    return cached;
  }

  store(ids: number[], now: number): boolean {
    if (!this.enabled || ids.length < this.minTokens) return false;
    this.entries.push({ ids, t: now });
    return true;
  }
}

export function callCost(price: Obj, input: number, cached: number, stored: boolean, output: number): number {
  const fresh = input - cached;
  const rate = stored ? price.cache_write : price.input;
  return (cached * price.cache_read + fresh * rate + output * price.output) / 1e6;
}

export function callLatency(profile: Obj, input: number, cached: number, output: number): [number, number] {
  const ttft = profile.overhead_ms + ((input - cached) * 1000) / profile.prefill_tps;
  return [ttft, ttft + (output * 1000) / profile.decode_tps];
}
