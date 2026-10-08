/** The context module (engines 1.4.0 and 1.5.0): corpus, chunkers, retrieval, evaluation, the window
 * simulation and the views. A port of agent_loop_sim/context. */
export { CHUNKERS, DEFAULT, FILES, Corpus, spansSha256, type Chunk } from "./corpus";
export { STOPWORDS, analyse, paragraphStarts, sentenceStarts, ranges, fixed, type Piece } from "./text";
export { PRECISIONS, decode, dot, cosine, toInt4, toBits, binarySim, prepare, similarity, nbytes } from "./vectors";
export { ln } from "./mathx";
export { fixedRanges, recursiveRanges, semanticBreaks, semanticRanges, spans } from "./chunking";
export { K1, B, RRF_K, BM25, rank, denseScores, rrf, minmaxTop, weighted, rerank, Retriever } from "./retrieval";
export { KS, DISCOUNT, METRICS, relevant, metrics, meanMetrics, evaluate } from "./evaluate";
export { POLICIES, MAX_READS, SYSTEM, taskQuestions, windowRun } from "./window";
export { tfCurve, lengthCurve, bm25View, denseFrames, hybridFrames, chunkView } from "./views";
export { SUMMARISERS, compactionStudy } from "./window";
export {
  PACKERS,
  PLACEMENTS,
  CANDIDATES,
  POSITION,
  gain,
  positionP,
  positionCurve,
  candidates,
  greedy,
  knapsackTable,
  knapsackPick,
  place,
  positions,
  useP,
  packAll,
  packingEval,
  packingView,
  type Cand,
} from "./packing";
export {
  SESSIONS,
  PER_SESSION,
  HOURS,
  DECAY,
  IMPORTANCE,
  POLICIES as MEMORY_POLICIES,
  SWEEP as MEMORY_SWEEP,
  recency,
  memoryPlan,
  memoryRun,
} from "./memory";
export { STRATEGIES, GAP_MS, OUT_TOKENS, MODELS, measuredSizes, runQuestions, tradeoff } from "./tradeoff";
