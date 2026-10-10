# Orchestration module results, part 2 (engine 1.7.0)

Written by `scripts/make_fixtures.py` (CI checks it is up to date). Every token size, accuracy, latency, rate limit and failure rate here is illustrative (see the module docstrings); prices are the engine's dated list prices (`accounting.PRICES`), latency is the engine's hosted profile.

## Multi-agent patterns on one task (4 sub-questions, 700-token tool results, uncontended, Claude Sonnet 4.6 prices)

| pattern | model calls | input tokens | cached | output | cost ($) | latency (s) | one call at a time (s) | P(success) |
|---|---|---|---|---|---|---|---|---|
| single | 5 | 12700 | 7620 | 690 | 0.0309 | 14.8 | 14.8 | 0.903 |
| supervisor | 13 | 19640 | 9120 | 1930 | 0.0704 | 34.6 | 34.6 | 0.817 |
| hierarchical | 17 | 23590 | 10350 | 2550 | 0.0902 | 44.5 | 44.5 | 0.777 |
| swarm | 8 | 24040 | 9480 | 1620 | 0.0810 | 29.6 | 29.6 | 0.840 |
| debate | 11 | 37020 | 16500 | 2190 | 0.1140 | 23.1 | 39.1 | 0.902 |
| map_reduce | 10 | 14480 | 4200 | 1930 | 0.0680 | 15.9 | 33.4 | 0.830 |

## Patterns as the task grows (1,500-token tool results, context penalty 0.01 per 1K tokens)

| sub-questions | pattern | model calls | input tokens | cost ($) | latency (s) | P(success) |
|---|---|---|---|---|---|---|
| 4 | single | 5 | 20700 | 0.0444 | 15.5 | 0.857 |
| 4 | supervisor | 13 | 22840 | 0.0824 | 35.3 | 0.817 |
| 4 | hierarchical | 17 | 26790 | 0.1022 | 45.2 | 0.777 |
| 4 | swarm | 8 | 36840 | 0.1124 | 31.2 | 0.764 |
| 4 | debate | 11 | 61020 | 0.1626 | 24.2 | 0.878 |
| 4 | map_reduce | 10 | 17680 | 0.0800 | 16.0 | 0.830 |
| 8 | single | 9 | 65340 | 0.0829 | 24.5 | 0.583 |
| 8 | supervisor | 25 | 49620 | 0.1529 | 64.1 | 0.681 |
| 8 | hierarchical | 29 | 49970 | 0.1717 | 74.0 | 0.648 |
| 8 | swarm | 16 | 133520 | 0.3412 | 66.4 | 0.326 |
| 8 | debate | 15 | 136860 | 0.2750 | 34.5 | 0.619 |
| 8 | map_reduce | 18 | 33320 | 0.1464 | 19.2 | 0.706 |
| 16 | single | 17 | 229500 | 0.1824 | 42.6 | 0.112 |
| 16 | supervisor | 49 | 118060 | 0.2984 | 121.8 | 0.464 |
| 16 | hierarchical | 53 | 103770 | 0.3128 | 131.7 | 0.451 |
| 16 | swarm | 32 | 506400 | 1.1622 | 154.7 | 0.007 |
| 16 | debate | 23 | 363420 | 0.5224 | 55.1 | 0.123 |
| 16 | map_reduce | 34 | 64600 | 0.2791 | 25.6 | 0.506 |

## Best pattern by P(success) across the grid

| sub-questions | tool result tokens | penalty per 1K | best | P(success) | single agent |
|---|---|---|---|---|---|
| 4 | 700 | 0.0 | single | 0.904 | 0.904 |
| 4 | 700 | 0.01 | single | 0.903 | 0.903 |
| 4 | 700 | 0.02 | single | 0.903 | 0.903 |
| 4 | 1500 | 0.0 | single | 0.904 | 0.904 |
| 4 | 1500 | 0.01 | debate | 0.878 | 0.857 |
| 4 | 1500 | 0.02 | debate | 0.846 | 0.812 |
| 8 | 700 | 0.0 | single | 0.834 | 0.834 |
| 8 | 700 | 0.01 | debate | 0.786 | 0.768 |
| 8 | 700 | 0.02 | debate | 0.735 | 0.707 |
| 8 | 1500 | 0.0 | single | 0.834 | 0.834 |
| 8 | 1500 | 0.01 | map_reduce | 0.706 | 0.583 |
| 8 | 1500 | 0.02 | map_reduce | 0.706 | 0.397 |
| 16 | 700 | 0.0 | single | 0.709 | 0.709 |
| 16 | 700 | 0.01 | map_reduce | 0.506 | 0.376 |
| 16 | 700 | 0.02 | map_reduce | 0.500 | 0.191 |
| 16 | 1500 | 0.0 | single | 0.709 | 0.709 |
| 16 | 1500 | 0.01 | map_reduce | 0.506 | 0.112 |
| 16 | 1500 | 0.02 | map_reduce | 0.500 | 0.012 |

## Fan-out under a rate limit (32 chunks of 2000 tokens, map-reduce, 50 RPM, 8000 OTPM)

| ITPM | width | run time (s) | speed-up | Amdahl | rate-limit ceiling |
|---|---|---|---|---|---|
| 40000 | 1 | 140.5 | 1.00 | 1.00 | 1.49 |
| 40000 | 2 | 102.1 | 1.38 | 1.88 | 1.49 |
| 40000 | 4 | 102.1 | 1.38 | 3.34 | 1.49 |
| 40000 | 8 | 102.1 | 1.38 | 5.49 | 1.49 |
| 40000 | 16 | 102.1 | 1.38 | 8.08 | 1.49 |
| 40000 | 32 | 102.1 | 1.38 | 10.58 | 1.49 |
| 80000 | 1 | 140.5 | 1.00 | 1.00 | 8.20 |
| 80000 | 2 | 74.9 | 1.88 | 1.88 | 8.20 |
| 80000 | 4 | 42.0 | 3.34 | 3.34 | 8.20 |
| 80000 | 8 | 25.6 | 5.49 | 5.49 | 8.20 |
| 80000 | 16 | 25.5 | 5.50 | 8.08 | 8.20 |
| 80000 | 32 | 25.5 | 5.50 | 10.58 | 8.20 |
| 200000 | 1 | 140.5 | 1.00 | 1.00 | 36.74 |
| 200000 | 2 | 74.9 | 1.88 | 1.88 | 36.74 |
| 200000 | 4 | 42.0 | 3.34 | 3.34 | 36.74 |
| 200000 | 8 | 25.6 | 5.49 | 5.49 | 36.74 |
| 200000 | 16 | 17.4 | 8.08 | 8.08 | 36.74 |
| 200000 | 32 | 13.3 | 10.58 | 10.58 | 36.74 |

| ITPM | one at a time (s) | parallel fraction | bound from the buckets (s) | requests | uncached input | output |
|---|---|---|---|---|---|---|
| 40000 | 140.5 | 0.935 | 94.3 | 34 | 102840 | 8510 |
| 80000 | 140.5 | 0.935 | 17.1 | 34 | 102840 | 8510 |
| 200000 | 140.5 | 0.935 | 3.8 | 34 | 102840 | 8510 |

## Reliability (detectable errors e = 0.05, silent errors w = 0.03, 2000 runs per point, verifier recall 0.8, false rejections 0.05)

| configuration | steps | closed form | simulated | 95% CI | mean attempts | closed form |
|---|---|---|---|---|---|---|
| none | 1 | 0.9200 | 0.9245 | [0.912, 0.935] | 1.00 | 1.00 |
| none | 2 | 0.8464 | 0.8580 | [0.842, 0.873] | 1.96 | 1.95 |
| none | 3 | 0.7787 | 0.7565 | [0.737, 0.775] | 2.84 | 2.85 |
| none | 5 | 0.6591 | 0.6610 | [0.640, 0.681] | 4.51 | 4.52 |
| none | 8 | 0.5132 | 0.4990 | [0.477, 0.521] | 6.74 | 6.73 |
| none | 10 | 0.4344 | 0.4380 | [0.416, 0.460] | 7.91 | 8.03 |
| none | 15 | 0.2863 | 0.2910 | [0.272, 0.311] | 10.91 | 10.73 |
| none | 20 | 0.1887 | 0.1715 | [0.156, 0.189] | 12.65 | 12.83 |
| none | 30 | 0.0820 | 0.0930 | [0.081, 0.107] | 16.10 | 15.71 |
| retry | 1 | 0.9683 | 0.9675 | [0.959, 0.974] | 1.05 | 1.05 |
| retry | 2 | 0.9376 | 0.9435 | [0.933, 0.953] | 2.10 | 2.10 |
| retry | 3 | 0.9079 | 0.9035 | [0.890, 0.916] | 3.17 | 3.16 |
| retry | 5 | 0.8512 | 0.8545 | [0.838, 0.869] | 5.27 | 5.26 |
| retry | 8 | 0.7728 | 0.7495 | [0.730, 0.768] | 8.41 | 8.42 |
| retry | 10 | 0.7246 | 0.7290 | [0.709, 0.748] | 10.53 | 10.52 |
| retry | 15 | 0.6168 | 0.5900 | [0.568, 0.611] | 15.75 | 15.77 |
| retry | 20 | 0.5250 | 0.5055 | [0.484, 0.527] | 21.07 | 21.03 |
| retry | 30 | 0.3805 | 0.4075 | [0.386, 0.429] | 31.48 | 31.52 |
| verify | 1 | 0.9915 | 0.9920 | [0.987, 0.995] | 1.13 | 1.13 |
| verify | 2 | 0.9830 | 0.9875 | [0.982, 0.992] | 2.27 | 2.27 |
| verify | 3 | 0.9746 | 0.9715 | [0.963, 0.978] | 3.40 | 3.40 |
| verify | 5 | 0.9581 | 0.9545 | [0.944, 0.963] | 5.64 | 5.65 |
| verify | 8 | 0.9337 | 0.9320 | [0.920, 0.942] | 9.04 | 9.02 |
| verify | 10 | 0.9179 | 0.9245 | [0.912, 0.935] | 11.25 | 11.26 |
| verify | 15 | 0.8794 | 0.8805 | [0.866, 0.894] | 16.84 | 16.81 |
| verify | 20 | 0.8425 | 0.8365 | [0.820, 0.852] | 22.42 | 22.32 |
| verify | 30 | 0.7733 | 0.7825 | [0.764, 0.800] | 33.18 | 33.19 |

## At scale (200 workflows per row, Poisson arrivals, seed 11; 50 RPM, 40000 ITPM, 8000 OTPM, concurrency 16, transient failure 0.03, 3 attempts)

| pattern | capacity (workflows/min) | limited by | arrivals/min | success | p50 (s) | p95 (s) | p99 (s) | $ per success | completed/min |
|---|---|---|---|---|---|---|---|---|---|
| single | 7.87 | input | 1.0 | 0.925 | 16.8 | 19.4 | 20.1 | 0.0334 | 0.99 |
| single | 7.87 | input | 2.0 | 0.925 | 17.0 | 19.2 | 20.0 | 0.0334 | 1.98 |
| single | 7.87 | input | 3.0 | 0.920 | 16.8 | 19.5 | 20.6 | 0.0336 | 2.95 |
| single | 7.87 | input | 4.0 | 0.925 | 16.6 | 18.8 | 20.2 | 0.0334 | 3.95 |
| single | 7.87 | input | 6.0 | 0.925 | 16.6 | 19.1 | 19.7 | 0.0334 | 5.91 |
| single | 7.87 | input | 8.0 | 0.875 | 93.0 | 158.7 | 177.7 | 0.0353 | 6.97 |
| supervisor | 3.80 | input | 1.0 | 0.810 | 39.6 | 42.8 | 44.5 | 0.0869 | 0.87 |
| supervisor | 3.80 | input | 2.0 | 0.785 | 39.9 | 43.4 | 45.2 | 0.0896 | 1.68 |
| supervisor | 3.80 | input | 3.0 | 0.805 | 40.5 | 66.1 | 75.2 | 0.0870 | 2.57 |
| supervisor | 3.80 | input | 4.0 | 0.850 | 478.2 | 644.8 | 687.3 | 0.0828 | 3.15 |
| supervisor | 3.80 | input | 6.0 | 0.835 | 1641.8 | 1871.5 | 1966.7 | 0.0843 | 2.96 |
| supervisor | 3.80 | input | 8.0 | 0.835 | 2111.5 | 2311.3 | 2437.5 | 0.0843 | 2.91 |
| map_reduce | 3.89 | input | 1.0 | 0.830 | 18.8 | 22.0 | 23.4 | 0.0819 | 0.89 |
| map_reduce | 3.89 | input | 2.0 | 0.835 | 18.9 | 21.2 | 22.9 | 0.0810 | 1.79 |
| map_reduce | 3.89 | input | 3.0 | 0.840 | 19.5 | 53.1 | 67.2 | 0.0808 | 2.70 |
| map_reduce | 3.89 | input | 4.0 | 0.835 | 279.7 | 433.5 | 489.4 | 0.0814 | 3.21 |
| map_reduce | 3.89 | input | 6.0 | 0.825 | 1178.1 | 1664.9 | 1703.7 | 0.0824 | 3.13 |
| map_reduce | 3.89 | input | 8.0 | 0.815 | 1785.7 | 2040.9 | 2219.1 | 0.0834 | 3.03 |
| debate | 1.95 | input | 1.0 | 0.905 | 28.1 | 46.6 | 56.5 | 0.1260 | 0.97 |
| debate | 1.95 | input | 2.0 | 0.900 | 634.4 | 911.8 | 983.4 | 0.1266 | 1.72 |
| debate | 1.95 | input | 3.0 | 0.900 | 2179.4 | 3100.7 | 3344.3 | 0.1266 | 1.71 |
| debate | 1.95 | input | 4.0 | 0.905 | 3248.7 | 4057.1 | 4477.7 | 0.1260 | 1.73 |
| debate | 1.95 | input | 6.0 | 0.920 | 4418.1 | 4916.7 | 5017.6 | 0.1239 | 1.74 |
| debate | 1.95 | input | 8.0 | 0.925 | 4862.5 | 5283.6 | 5362.6 | 0.1232 | 1.76 |

## Choosing a pattern (DES, 1,500-token tool results; arrivals per minute: 1.0 for 4 sub-questions, 0.25 for 16 sub-questions)

| sub-questions | pattern | success | closed form | p95 (s) | $ per success | completed/min | capacity (workflows/min) |
|---|---|---|---|---|---|---|---|
| 4 | single | 0.850 | 0.857 | 19.9 | 0.0522 | 0.89 | 4.83 |
| 4 | supervisor | 0.824 | 0.817 | 44.8 | 0.0999 | 0.86 | 2.92 |
| 4 | hierarchical | 0.788 | 0.777 | 57.2 | 0.1297 | 0.82 | 2.43 |
| 4 | swarm | 0.769 | 0.764 | 99.9 | 0.1461 | 0.80 | 1.77 |
| 4 | debate | 0.873 | 0.877 | 784.6 | 0.1862 | 0.91 | 1.23 |
| 4 | map_reduce | 0.834 | 0.829 | 22.5 | 0.0959 | 0.87 | 2.97 |
| 16 | single | 0.105 | 0.112 | 51.7 | 1.7370 | 0.03 | 1.48 |
| 16 | supervisor | 0.472 | 0.464 | 192.5 | 0.6318 | 0.12 | 0.82 |
| 16 | hierarchical | 0.452 | 0.450 | 217.5 | 0.6920 | 0.12 | 0.78 |
| 16 | swarm | 0.008 | 0.007 | 218889.6 | 145.1505 | 0.00 | 0.15 |
| 16 | debate | 0.131 | 0.123 | 993.4 | 3.9842 | 0.03 | 0.37 |
| 16 | map_reduce | 0.523 | 0.505 | 152.4 | 0.5335 | 0.14 | 0.84 |

| sub-questions | constraints | survivors | pick |
|---|---|---|---|
| 4 | success ≥ 0.8 | single, supervisor, debate, map_reduce | debate |
| 4 | success ≥ 0.8, p95 ≤ 30000 | single, map_reduce | single |
| 4 | success ≥ 0.4, p95 ≤ 180000 | single, supervisor, hierarchical, swarm, map_reduce | single |
| 4 | success ≥ 0.4, p95 ≤ 180000, cost_per_success ≤ 0.6 | single, supervisor, hierarchical, swarm, map_reduce | single |
| 4 | success ≥ 0.4, capacity ≥ 1.0 | single, supervisor, hierarchical, swarm, debate, map_reduce | debate |
| 16 | success ≥ 0.8 | none | none |
| 16 | success ≥ 0.8, p95 ≤ 30000 | none | none |
| 16 | success ≥ 0.4, p95 ≤ 180000 | map_reduce | map_reduce |
| 16 | success ≥ 0.4, p95 ≤ 180000, cost_per_success ≤ 0.6 | map_reduce | map_reduce |
| 16 | success ≥ 0.4, capacity ≥ 1.0 | none | none |
