# Context module results, part 2 (engine 1.5.0)

Written by `scripts/make_fixtures.py` (CI checks it is up to date). Packing, compaction with a lossy summariser, memory across sessions, and long context against retrieval. Part 1 is `context_results.md`.

## The window task with a lossy summariser (12 questions; loss 0.25, seed 23)

| budget | policy | facts answered | input tokens spent | compactions |
|---|---|---|---|---|
| 1000 | compact (lossy) | 7 of 12 | 14631 | 3 |
| 1000 | compact+retrieve (lossy) | 12 of 12 | 27602 | 7 |
| 1500 | compact (lossy) | 7 of 12 | 16281 | 2 |
| 1500 | compact+retrieve (lossy) | 12 of 12 | 25125 | 3 |
| 2000 | compact (lossy) | 9 of 12 | 19092 | 1 |
| 2000 | compact+retrieve (lossy) | 12 of 12 | 28645 | 2 |
| 3000 | compact (lossy) | 8 of 12 | 27153 | 1 |
| 3000 | compact+retrieve (lossy) | 12 of 12 | 32359 | 1 |

## The long task (36 questions, six per article)

| budget | policy | facts answered | input tokens spent | model calls | compactions |
|---|---|---|---|---|---|
| 1500 | unbounded | 36 of 36 | 233502 | 47 | 0 |
| 1500 | truncate | 4 of 36 | 63744 | 47 | 0 |
| 1500 | compact | 32 of 36 | 98532 | 71 | 24 |
| 1500 | compact+retrieve | 36 of 36 | 241495 | 163 | 70 |
| 2000 | unbounded | 36 of 36 | 233502 | 47 | 0 |
| 2000 | truncate | 7 of 36 | 84643 | 47 | 0 |
| 2000 | compact | 36 of 36 | 89571 | 56 | 9 |
| 2000 | compact+retrieve | 36 of 36 | 89571 | 56 | 9 |

## What survives a lossy summariser (long task, 20 seeds each)

S(n): the share of facts still in the summary after their n-th compaction (product of each compaction's kept / faced), against the model (1 - loss)^n.

| budget | policy | loss | mean facts answered | mean input tokens | S(1) | S(2) | S(3) | S(5) | (1-loss)^5 |
|---|---|---|---|---|---|---|---|---|---|
| 1500 | compact | 0.1 | 16.45 of 36 | 79876 | 0.873 | 0.790 | 0.705 | 0.560 | 0.590 |
| 1500 | compact | 0.25 | 10.70 of 36 | 73856 | 0.746 | 0.557 | 0.411 | 0.251 | 0.237 |
| 1500 | compact | 0.5 | 5.60 of 36 | 70512 | 0.485 | 0.224 | 0.119 | 0.023 | 0.031 |
| 1500 | compact+retrieve | 0.1 | 36.00 of 36 | 151766 | 0.878 | 0.789 | 0.714 | 0.557 | 0.590 |
| 1500 | compact+retrieve | 0.25 | 36.00 of 36 | 149719 | 0.749 | 0.568 | 0.423 | 0.251 | 0.237 |
| 1500 | compact+retrieve | 0.5 | 36.00 of 36 | 142552 | 0.481 | 0.242 | 0.129 | 0.032 | 0.031 |
| 2000 | compact | 0.1 | 22.75 of 36 | 84661 | 0.900 | 0.806 | 0.721 | 0.577 | 0.590 |
| 2000 | compact | 0.25 | 15.15 of 36 | 80921 | 0.754 | 0.560 | 0.415 | 0.237 | 0.237 |
| 2000 | compact | 0.5 | 6.10 of 36 | 78179 | 0.474 | 0.223 | 0.101 | 0.032 | 0.031 |
| 2000 | compact+retrieve | 0.1 | 36.00 of 36 | 137812 | 0.901 | 0.811 | 0.719 | 0.583 | 0.590 |
| 2000 | compact+retrieve | 0.25 | 36.00 of 36 | 157827 | 0.751 | 0.560 | 0.424 | 0.230 | 0.237 |
| 2000 | compact+retrieve | 0.5 | 36.00 of 36 | 156344 | 0.478 | 0.243 | 0.127 | 0.036 | 0.031 |

## Packing (reranked top 20 candidates, value 1/log2(rank+1); means over all questions)

Position curve (illustrative): start 0.75, middle 0.55, end 0.65, trough at 0.50.

| budget | packer | answer in window | tokens used | value | p best-first | p best-last | p ends | p middle |
|---|---|---|---|---|---|---|---|---|
| 256 | top | 0.890 | 201.6 | 1.042 | 0.495 | 0.492 | 0.495 | 0.492 |
| 256 | density | 0.815 | 200.4 | 1.034 | 0.454 | 0.451 | 0.454 | 0.451 |
| 256 | optimal | 0.870 | 202.5 | 1.045 | 0.484 | 0.481 | 0.484 | 0.481 |
| 512 | top | 0.965 | 469.2 | 1.920 | 0.597 | 0.565 | 0.599 | 0.541 |
| 512 | density | 0.935 | 467.1 | 1.931 | 0.580 | 0.547 | 0.581 | 0.520 |
| 512 | optimal | 0.940 | 483.8 | 1.955 | 0.585 | 0.551 | 0.587 | 0.522 |
| 1024 | top | 0.995 | 980.0 | 3.089 | 0.667 | 0.607 | 0.670 | 0.551 |
| 1024 | density | 0.990 | 969.8 | 3.144 | 0.663 | 0.604 | 0.666 | 0.548 |
| 1024 | optimal | 0.985 | 1008.0 | 3.195 | 0.663 | 0.602 | 0.666 | 0.545 |
| 2048 | top | 0.995 | 1999.6 | 4.797 | 0.703 | 0.625 | 0.702 | 0.549 |
| 2048 | density | 0.995 | 1972.8 | 4.931 | 0.702 | 0.625 | 0.702 | 0.548 |
| 2048 | optimal | 0.995 | 2036.1 | 4.992 | 0.704 | 0.625 | 0.703 | 0.548 |

## Memory across sessions (6 sessions of 3 questions, then a probe-only session)

| policy | recalled | repeats | neighbours | memory tokens read | write tokens | stored at the end |
|---|---|---|---|---|---|---|
| none | 0 of 22 | 0 of 18 | 0 of 4 | 0 | 0 | 0 |
| transcript | 22 of 22 | 18 of 18 | 4 of 4 | 67498 | 0 | 4176 |
| scratchpad | 18 of 22 | 18 of 18 | 0 of 4 | 5981 | 372 | 374 |
| scratchpad-cap | 14 of 22 | 14 of 18 | 0 of 4 | 2553 | 372 | 123 |
| episodic | 19 of 22 | 17 of 18 | 2 of 4 | 12515 | 4198 | 3321 |
| episodic-cap | 12 of 22 | 11 of 18 | 1 of 4 | 12540 | 4198 | 1481 |
| semantic | 19 of 22 | 17 of 18 | 2 of 4 | 2231 | 5412 | 1103 |
| semantic-cap | 16 of 22 | 15 of 18 | 1 of 4 | 2121 | 5412 | 334 |
| episodic k=1 | 15 of 22 | 13 of 18 | 2 of 4 | 4134 | 4198 | 3321 |
| episodic k=2 | 19 of 22 | 17 of 18 | 2 of 4 | 8125 | 4198 | 3321 |
| episodic k=5 | 20 of 22 | 18 of 18 | 2 of 4 | 21221 | 4198 | 3321 |
| semantic k=1 | 16 of 22 | 15 of 18 | 1 of 4 | 829 | 5412 | 1103 |
| semantic k=5 | 20 of 22 | 18 of 18 | 2 of 4 | 3621 | 5412 | 1103 |
| semantic k=8 | 20 of 22 | 18 of 18 | 2 of 4 | 5693 | 5412 | 1103 |

## Long context or retrieval (50 questions, 60 s apart, 50 output tokens each)

Measured: system prompt 33 tokens, mean question 13.64, mean chunk 180.3 (208 chunks), corpus 37499 tokens. Retrieval: reranked top k; answer in the prompt = recall@k k=3 0.990, k=5 0.995, k=10 0.995, k=20 1.000.

| model | set (tokens) | k | strategy | first call $ | later call $ | all questions $ | first TTFT (s) | later TTFT (s) |
|---|---|---|---|---|---|---|---|---|
| claude-sonnet-4.6 | any | 3 | rag | 0.0025 | 0.0025 | 0.126 | 0.52 | 0.52 |
| claude-sonnet-4.6 | 37499 | - | long | 0.1134 | 0.1134 | 5.669 | 7.91 | 7.91 |
| claude-sonnet-4.6 | 37499 | - | long+cache | 0.1415 | 0.0121 | 0.733 | 7.91 | 0.40 |
| claude-sonnet-4.6 | any | 5 | rag | 0.0036 | 0.0036 | 0.180 | 0.59 | 0.59 |
| claude-sonnet-4.6 | any | 10 | rag | 0.0063 | 0.0063 | 0.315 | 0.77 | 0.77 |
| claude-sonnet-4.6 | any | 20 | rag | 0.0117 | 0.0117 | 0.585 | 1.13 | 1.13 |
| claude-sonnet-4.6 | 200000 | - | long | 0.6009 | 0.6009 | 30.045 | 40.41 | 40.41 |
| claude-sonnet-4.6 | 200000 | - | long+cache | 0.7509 | 0.0608 | 3.731 | 40.41 | 0.40 |
| claude-sonnet-4.6 | 1000000 | - | long | 3.0009 | 3.0009 | 150.045 | 200.41 | 200.41 |
| claude-sonnet-4.6 | 1000000 | - | long+cache | 3.7509 | 0.3008 | 18.491 | 200.41 | 0.40 |
| claude-haiku-4.5 | any | 3 | rag | 0.0008 | 0.0008 | 0.042 | 0.52 | 0.52 |
| claude-haiku-4.5 | 37499 | - | long | 0.0378 | 0.0378 | 1.890 | 7.91 | 7.91 |
| claude-haiku-4.5 | 37499 | - | long+cache | 0.0472 | 0.0040 | 0.244 | 7.91 | 0.40 |
| claude-haiku-4.5 | any | 5 | rag | 0.0012 | 0.0012 | 0.060 | 0.59 | 0.59 |
| claude-haiku-4.5 | any | 10 | rag | 0.0021 | 0.0021 | 0.105 | 0.77 | 0.77 |
| claude-haiku-4.5 | any | 20 | rag | 0.0039 | 0.0039 | 0.195 | 1.13 | 1.13 |
| claude-haiku-4.5 | 200000 | - | long | 0.2003 | 0.2003 | 10.015 | 40.41 | 40.41 |
| claude-haiku-4.5 | 200000 | - | long+cache | 0.2503 | 0.0203 | 1.244 | 40.41 | 0.40 |
| claude-haiku-4.5 | 1000000 | - | long | 1.0003 | 1.0003 | 50.015 | 200.41 | 200.41 |
| claude-haiku-4.5 | 1000000 | - | long+cache | 1.2503 | 0.1003 | 6.164 | 200.41 | 0.40 |
| gpt-5-mini | any | 3 | rag | 0.0002 | 0.0002 | 0.012 | 0.52 | 0.52 |
| gpt-5-mini | 37499 | - | long | 0.0095 | 0.0095 | 0.474 | 7.91 | 7.91 |
| gpt-5-mini | 37499 | - | long+cache | 0.0095 | 0.0010 | 0.061 | 7.91 | 0.41 |
| gpt-5-mini | any | 5 | rag | 0.0003 | 0.0003 | 0.017 | 0.59 | 0.59 |
| gpt-5-mini | any | 10 | rag | 0.0006 | 0.0006 | 0.028 | 0.77 | 0.77 |
| gpt-5-mini | any | 20 | rag | 0.0010 | 0.0010 | 0.051 | 1.13 | 1.13 |
| gpt-5-mini | 200000 | - | long | 0.0501 | 0.0501 | 2.506 | 40.41 | 40.41 |
| gpt-5-mini | 200000 | - | long+cache | 0.0501 | 0.0051 | 0.301 | 40.41 | 0.42 |
| gpt-5-mini | 1000000 | - | long | 0.2501 | 0.2501 | 12.506 | 200.41 | 200.41 |
| gpt-5-mini | 1000000 | - | long+cache | 0.2501 | 0.0251 | 1.481 | 200.41 | 0.42 |
