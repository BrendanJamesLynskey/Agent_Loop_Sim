/**
 * mulberry32, bit for bit the Python reference's `Rng` (and Disaggregated_Inference_Sim's
 * browser engine): 32-bit integer arithmetic only.
 */
export class Rng {
  state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  nextU32(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    const a = this.state;
    let t = Math.imul(a ^ (a >>> 15), 1 | a) >>> 0;
    t = ((t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0) ^ t;
    return (t ^ (t >>> 14)) >>> 0;
  }

  random(): number {
    return this.nextU32() / 4294967296;
  }

  /** An integer in [lo, hi], inclusive. */
  randint(lo: number, hi: number): number {
    return lo + Math.floor(this.random() * (hi - lo + 1));
  }

  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.random();
  }
}
