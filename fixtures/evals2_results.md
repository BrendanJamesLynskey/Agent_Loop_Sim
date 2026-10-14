# Evals module results, part 2 (engine 1.11.0)

Written by `scripts/make_fixtures.py` (CI checks it is up to date). Every quality, bias, skill, rate, leak and task here is synthetic and illustrative; token counts and costs come from the harness's own accounting (the vendored Qwen2.5 tokenizer, the dated price table).

## Chapter 6: a simulated judge (200 pairs, noise τ = 0.3, ties within 0.3 of quality)

| biases | accuracy, one order | κ, one order | first shown wins | consistent across the swap | κ, swapped (ties when inconsistent) | accuracy where decided | longer-but-worse pairs right | own-family-but-worse pairs right |
|---|---|---|---|---|---|---|---|---|
| fair (pos 0.0, len 0.0, self 0.0) | 84.5% | 0.72 | 49.8% | 88.5% | 0.77 | 92.1% | 96.5% (n = 86) | 89.2% (n = 37) |
| position (pos 0.8, len 0.0, self 0.0) | 76.0% | 0.56 | 67.8% | 64.5% | 0.55 | 95.3% | 88.4% (n = 86) | 86.5% (n = 37) |
| verbosity (pos 0.0, len 0.6, self 0.0) | 80.0% | 0.64 | 48.3% | 86.5% | 0.64 | 86.1% | 81.4% (n = 86) | 83.8% (n = 37) |
| self (pos 0.0, len 0.0, self 0.8) | 80.5% | 0.65 | 49.8% | 88.5% | 0.67 | 86.4% | 93.0% (n = 86) | 67.6% (n = 37) |
| all (pos 0.8, len 0.6, self 0.8) | 76.0% | 0.56 | 64.5% | 71.0% | 0.49 | 86.6% | 76.7% (n = 86) | 73.0% (n = 37) |

Pass/fail grading (300 tasks, skill -0.4; judge Se 0.95, Sp 0.7): unit tests 40.7%, judge 58.3%, κ 0.62. Calibration (200 tasks labelled): Se 74/75, Sp 88/125. Corrected 41.6%, 95% 30.9% to 52.3% (SE 0.0543; 0.0412 if Se and Sp were known).

## Chapter 7: grading trajectories

| trajectory | status | outcome | self-report | trajectory | verified | wasted steps | unsafe run / stopped | process score | tokens |
|---|---|---|---|---|---|---|---|---|---|
| Scripted: fix, test, done | done | pass | pass | pass | yes | 0 | 0 / 0 | 1.00 | 4,539 |
| Scripted: one malformed call, then the fix | done | pass | pass | pass | yes | 1 | 0 / 0 | 0.90 | 5,398 |
| Scripted: the fix on a flaky shell | done | pass | pass | pass | yes | 0 | 0 / 0 | 1.00 | 4,539 |
| Scripted: the fix, then rm -rf build | done | pass | pass | fail | yes | 0 | 1 / 0 | 0.50 | 6,644 |
| Scripted: the fix, then installs and deletes, in a sandbox | done | pass | pass | fail | yes | 0 | 1 / 2 | 0.50 | 9,074 |
| Recorded (Qwen2.5-1.5B): native tool calls | done | fail | fail | fail | no | 0 | 0 / 0 | 0.80 | 1,127 |
| Recorded (Qwen2.5-1.5B): ReAct text | done | fail | pass | fail | no | 2 | 0 / 0 | 0.60 | 2,775 |
| Scripted: search, calculate, answer | done | pass | pass | pass | yes | 0 | 0 / 0 | 1.00 | 2,075 |
| Recorded (Qwen2.5-1.5B): search and answer | done | fail | pass | fail | no | 0 | 0 / 0 | 0.80 | 1,290 |

## Chapter 8: one model, many harnesses (12 tasks × 8 runs per configuration; within-task correlation 0.5; malformed-output rates 0.02 native, 0.12 ReAct)

| agent | tools | style | turns | resolved | Wilson 95% | cluster 95% | model calls | malformed / run | tokens / run | $ / run | $ / solved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | edit | native | 4 | 44.8% (43/96) | 35.2% to 54.7% | 25.8% to 63.7% | 3.07 | 0.07 | 1,358 | 0.00173 | 0.00386 |
| small | edit | react | 4 | 43.8% (42/96) | 34.3% to 53.7% | 25.5% to 62.0% | 3.26 | 0.34 | 1,375 | 0.00181 | 0.00413 |
| small | test | native | 4 | 53.1% (51/96) | 43.2% to 62.8% | 35.3% to 71.0% | 4.00 | 0.11 | 2,272 | 0.00286 | 0.00539 |
| small | test | native | 8 | 65.6% (63/96) | 55.7% to 74.4% | 46.8% to 84.5% | 6.11 | 0.14 | 4,318 | 0.00522 | 0.00796 |
| small | test | native | 16 | 74.0% (71/96) | 64.4% to 81.7% | 56.3% to 91.7% | 8.68 | 0.20 | 8,488 | 0.00982 | 0.01327 |
| small | test | react | 4 | 46.9% (45/96) | 37.2% to 56.8% | 29.8% to 64.0% | 4.00 | 0.54 | 2,043 | 0.00260 | 0.00554 |
| small | test | react | 8 | 64.6% (62/96) | 54.6% to 73.4% | 45.1% to 84.1% | 6.31 | 0.78 | 4,018 | 0.00491 | 0.00760 |
| small | test | react | 16 | 72.9% (70/96) | 63.3% to 80.8% | 55.4% to 90.5% | 9.07 | 1.10 | 7,974 | 0.00930 | 0.01276 |
| large | edit | native | 4 | 64.6% (62/96) | 54.6% to 73.4% | 48.7% to 80.5% | 3.07 | 0.07 | 1,358 | 0.00519 | 0.00804 |
| large | edit | react | 4 | 63.5% (61/96) | 53.6% to 72.5% | 48.3% to 78.8% | 3.26 | 0.34 | 1,375 | 0.00543 | 0.00854 |
| large | test | native | 4 | 68.8% (66/96) | 58.9% to 77.1% | 53.8% to 83.7% | 4.00 | 0.11 | 2,252 | 0.00842 | 0.01225 |
| large | test | native | 8 | 79.2% (76/96) | 70.0% to 86.1% | 64.6% to 93.7% | 5.43 | 0.14 | 3,622 | 0.01331 | 0.01682 |
| large | test | native | 16 | 86.5% (83/96) | 78.2% to 91.9% | 74.6% to 98.3% | 7.03 | 0.15 | 6,178 | 0.01561 | 0.01805 |
| large | test | react | 4 | 65.6% (63/96) | 55.7% to 74.4% | 50.2% to 81.0% | 4.00 | 0.52 | 2,033 | 0.00772 | 0.01176 |
| large | test | react | 8 | 77.1% (74/96) | 67.7% to 84.4% | 60.9% to 93.3% | 5.67 | 0.72 | 3,419 | 0.01260 | 0.01634 |
| large | test | react | 16 | 84.4% (81/96) | 75.8% to 90.3% | 73.0% to 95.7% | 7.48 | 0.91 | 5,946 | 0.01599 | 0.01895 |

Spread across scaffolds: small: 43.8% to 74.0% (30.2 points); large: 63.5% to 86.5% (22.9 points).

| paired comparison | A only | B only | McNemar exact p | difference | paired 95% |
|---|---|---|---|---|---|
| small|test|native|16 vs small|edit|native|4 | 28 | 0 | 0.0000 | 29.2% | 20.0% to 38.3% |
| small|test|native|16 vs large|edit|native|4 | 16 | 7 | 0.0931 | 9.4% | -0.3% to 19.0% |
| small|test|react|8 vs small|test|native|8 | 0 | 1 | 1.0000 | -1.0% | -3.1% to 1.0% |
| large|test|native|16 vs large|test|native|4 | 17 | 0 | 0.0000 | 17.7% | 10.0% to 25.4% |

## Chapter 10: cost

Pareto front (cheapest first): small|edit|native|4, small|test|react|4, small|test|native|4, small|test|react|8, small|test|native|8, large|test|native|4, small|test|react|16, small|test|native|16, large|test|react|8, large|test|native|8, large|test|native|16. Highest success: large|test|native|16. Fewest dollars per solved task: small|edit|native|4.

## Chapter 9: contamination (400 tasks, skill 0.0, memorise 0.7, paraphrased share 0.4, mutation 0.25, 8-grams flagged at 0.5)

Chance an 8-gram survives a paraphrase: 0.1001.

| leak rate | leaked (verbatim / paraphrased) | flagged verbatim | flagged paraphrased | canary | score | clean score | after dropping flagged | before cutoff | after cutoff | split p |
|---|---|---|---|---|---|---|---|---|---|---|
| 0% | 0 (0 / 0) | 0 | 0 | no | 48.3% | 48.3% | 48.3% | 50.0% | 46.4% | 0.4704 |
| 10% | 23 (15 / 8) | 15 | 0 | yes | 50.0% | 48.3% | 49.4% | 53.4% | 46.4% | 0.1613 |
| 30% | 62 (41 / 21) | 41 | 0 | yes | 53.8% | 48.3% | 50.7% | 60.7% | 46.4% | 0.0042 |
| 60% | 131 (80 / 51) | 80 | 0 | yes | 60.8% | 48.3% | 55.3% | 74.3% | 46.4% | 0.0000 |
