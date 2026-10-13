# Evals module results (engine 1.9.0)

Written by `scripts/make_fixtures.py` (CI checks it is up to date). Every difficulty, skill, correlation and benchmark here is synthetic and illustrative (see `evals/generator.py`); the statistics are exact under the stated assumptions.

Population: 4000 tasks, difficulty sd 1.3, discrimination uniform on [0.7, 1.9], seed 2501.

## Chapter 1: runs accumulating (40 tasks, 3 runs each, skill 0.6, within-task correlation 0.5)

| after | runs | rate | Wilson 95% (runs independent) | cluster-robust 95% (tasks the unit) |
|---|---|---|---|---|
| run 1 of every task | 40 | 67.5% | 52.0% to 79.9% | 52.8% to 82.2% |
| run 2 of every task | 80 | 67.5% | 56.6% to 76.8% | 54.1% to 80.9% |
| run 3 of every task | 120 | 67.5% | 58.7% to 75.2% | 54.2% to 80.8% |

True score of these 40 tasks: 71.2%; of the population: 62.1%. Design effect after all runs: 2.53.

## Chapter 3: pass@k and pass^k (30 tasks, n = 10 samples, skill -0.2, correlation 0.3)

| k | pass@k | true | plug-in | pass^k | true |
|---|---|---|---|---|---|
| 1 | 45.7% | 44.4% | 45.7% | 45.7% | 44.4% |
| 2 | 56.8% | 56.0% | 55.7% | 34.5% | 32.7% |
| 3 | 62.8% | 61.7% | 61.0% | 29.3% | 26.8% |
| 4 | 67.0% | 65.6% | 64.5% | 25.8% | 22.8% |
| 5 | 70.5% | 68.6% | 67.2% | 23.1% | 19.6% |
| 6 | 73.3% | 71.0% | 69.2% | 21.0% | 17.1% |
| 7 | 75.7% | 73.0% | 70.9% | 19.4% | 15.0% |
| 8 | 77.6% | 74.6% | 72.3% | 18.2% | 13.2% |
| 9 | 79.0% | 76.1% | 73.4% | 17.3% | 11.7% |
| 10 | 80.0% | 77.3% | 74.3% | 16.7% | 10.4% |

Bias check (one task, μ = 0.3, n = 10, k = 5, 4000 repeats): unbiased pass@k averages 0.8323, the plug-in 0.7596, the truth is 0.8319; pass^k averages 0.00260 against 0.00243.

## Chapter 4: repeating the benchmark (30 repeats, skill 0.4, one run per task, bootstrap 200 resamples)

| tasks | what changes | truth | Wald | Wilson | exact | bootstrap | mean Wilson width |
|---|---|---|---|---|---|---|---|
| 50 | tasks and runs | 58.1% | 93% | 93% | 97% | 97% | 26.0% |
| 50 | runs only | 55.6% | 97% | 97% | 100% | 93% | 26.3% |
| 200 | tasks and runs | 58.1% | 87% | 87% | 90% | 87% | 13.4% |
| 200 | runs only | 57.1% | 97% | 97% | 97% | 97% | 13.5% |
| 500 | tasks and runs | 58.1% | 93% | 93% | 93% | 93% | 8.6% |
| 500 | runs only | 57.9% | 100% | 100% | 100% | 100% | 8.6% |

Exact coverage of nominal 95% intervals (minimum over p = 0.02 … 0.98):

| n | Wald | Wilson | exact |
|---|---|---|---|
| 20 | 33.2% | 92.7% | 95.9% |
| 100 | 86.6% | 93.6% | 95.6% |

The tie: 341/500 (68.2%) against 339/500 (67.8%): z = 0.136, p = 0.892; Wilson 64.0% to 72.1% and 63.6% to 71.7%. Tasks per agent to detect that difference at 80% power: 213,489 (unpaired), 98,109 (paired, discordance 0.2). Smallest difference 500 tasks detect: 8.3% unpaired, 5.6% paired.

| margin ± | n at p = 0.5 | n at p = 0.8 |
|---|---|---|
| 5.0% | 385 | 246 |
| 3.0% | 1,068 | 683 |
| 2.0% | 2,401 | 1,537 |
| 1.0% | 9,604 | 6,147 |

| difference (base 0.6) | tasks per agent, unpaired | shared tasks, paired (ψ = 0.2) |
|---|---|---|
| 10.0% | 356 | 155 |
| 5.0% | 1,471 | 626 |
| 3.0% | 4,129 | 1,742 |
| 1.0% | 37,513 | 15,696 |

## Chapter 5: two agents on the same 300 tasks (skills 0.45 and 0.1, correlation 0.5)

| runs per task | A | B | difference | paired 95% | unpaired 95% | McNemar p (first runs) | SE² tasks (paired) | SE² tasks (unpaired) | SE² runs |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 61.7% | 54.3% | 7.3% | 1.2% to 13.5% | -0.6% to 15.2% | 0.0263 | 4.72e-06 | 5.23e-04 | 1.10e-03 |
| 3 | 61.9% | 57.3% | 4.6% | -0.9% to 10.0% | -2.4% to 11.5% | 0.9219 | 4.72e-06 | 5.23e-04 | 7.37e-04 |
| 5 | 59.9% | 54.5% | 5.4% | 0.4% to 10.4% | -1.4% to 12.2% | 0.2513 | 4.72e-06 | 5.23e-04 | 6.63e-04 |

True difference on these tasks: 7.2%.

## Chapter 2: one SWE-bench-style task

| patch | hidden tests | FAIL_TO_PASS | PASS_TO_PASS | resolved |
|---|---|---|---|---|
| the right fix | thorough | pass | pass | yes |
| the right fix | weak | pass | pass | yes |
| an overfit patch | thorough | fail | pass | no |
| an overfit patch | weak | pass | pass | yes |
| a fix that breaks sub() | thorough | pass | fail | no |
| a fix that breaks sub() | weak | pass | fail | no |
