# Orchestration module results (engine 1.6.0)

Written by `scripts/make_fixtures.py` (CI checks it is up to date). Node durations, the checkpoint write (15 ms) and every durable-execution timing are illustrative; the semantics are checked against LangGraph.

## Conformance with LangGraph

`fixtures/langgraph_recordings.json`: langgraph 1.2.14 (langgraph-checkpoint 4.2.0), 30 sessions, 55 operations, 178 checkpoints in the final histories. CI re-records them live and requires live = committed = engine for every checkpoint after every operation.

## Super-steps of the teaching graphs

| graph | super-steps | graph time (ms) | one node at a time (ms, a checkpoint per node) | speed-up |
|---|---|---|---|---|
| chain | 4 | 2860 | 2860 | 1.00 |
| fanout | 4 | 3260 | 4475 | 1.37 |
| react_loop | 6 | 3590 | 3590 | 1.00 |
| uneven_join | 5 | 1275 | 1590 | 1.25 |
| loop_join | 7 | 2905 | 3935 | 1.35 |

## Two branches, one key

| reducer | status | answer after the join |
|---|---|---|
| overwrite | error | "InvalidUpdateError" |
| add | done | [42, 41] |
| max | done | 42 |

## Checkpoint sessions

| session | operations | statuses | checkpoints | node runs | side effects |
|---|---|---|---|---|---|
| crash | 2 | error, done | 5 | 5 | 3 |
| crash_twice | 3 | error, error, done | 5 | 6 | 4 |
| approval | 2 | interrupted, done | 5 | 5 | 3 |
| approval_no | 2 | interrupted, done | 5 | 5 | 3 |
| gated | 2 | interrupt_before, done | 5 | 3 | 1 |
| gated_edit | 3 | interrupt_before, updated, done | 6 | 3 | 1 |
| fork | 3 | done, updated, done | 8 | 5 | 0 |
| replay | 2 | done, done | 8 | 5 | 0 |
| replay_interrupt | 3 | interrupt_before, done, done | 8 | 5 | 2 |

## Durable execution scenarios

| scenario | status | charges | £ charged | receipts | events | finished at (ms) |
|---|---|---|---|---|---|---|
| clean | completed | 1 | 40 | 1 | 10 | 1800 |
| crash_replay | completed | 1 | 40 | 1 | 11 | 6810 |
| crash_in_charge | completed | 2 | 80 | 1 | 11 | 7605 |
| crash_in_charge_key | completed | 1 | 40 | 1 | 11 | 7605 |
| lost_reply | completed | 2 | 80 | 1 | 12 | 4600 |
| lost_reply_key | completed | 1 | 40 | 1 | 12 | 4600 |
| nondeterministic | nondeterminism | 1 | 40 | 0 | 6 | 6110 |
| deterministic | completed | 1 | 40 | 1 | 11 | 6810 |

## Lost replies and retries (4000 orders per row, seed 24)

| q | attempts | mean charges, no key | closed form (1-q^A)/(1-q) | P(charged twice or more) | closed form q | P(all attempts lost) | closed form q^A | mean charges, key |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0560 | 0.0500 | 1.0000 |
| 0.05 | 2 | 1.0557 | 1.0500 | 0.0558 | 0.0500 | 0.0030 | 0.0025 | 1.0000 |
| 0.05 | 3 | 1.0590 | 1.0525 | 0.0560 | 0.0500 | 0.0000 | 0.0001 | 1.0000 |
| 0.05 | 5 | 1.0590 | 1.0526 | 0.0560 | 0.0500 | 0.0000 | 0.0000 | 1.0000 |
| 0.1 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.1025 | 0.1000 | 1.0000 |
| 0.1 | 2 | 1.1007 | 1.1000 | 0.1008 | 0.1000 | 0.0097 | 0.0100 | 1.0000 |
| 0.1 | 3 | 1.1100 | 1.1100 | 0.1003 | 0.1000 | 0.0010 | 0.0010 | 1.0000 |
| 0.1 | 5 | 1.1110 | 1.1111 | 0.1003 | 0.1000 | 0.0000 | 0.0000 | 1.0000 |
| 0.2 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.2137 | 0.2000 | 1.0000 |
| 0.2 | 2 | 1.2140 | 1.2000 | 0.2140 | 0.2000 | 0.0415 | 0.0400 | 1.0000 |
| 0.2 | 3 | 1.2580 | 1.2400 | 0.2152 | 0.2000 | 0.0083 | 0.0080 | 1.0000 |
| 0.2 | 5 | 1.2672 | 1.2496 | 0.2147 | 0.2000 | 0.0005 | 0.0003 | 1.0000 |
| 0.5 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.5072 | 0.5000 | 1.0000 |
| 0.5 | 2 | 1.5005 | 1.5000 | 0.5005 | 0.5000 | 0.2570 | 0.2500 | 1.0000 |
| 0.5 | 3 | 1.7668 | 1.7500 | 0.5062 | 0.5000 | 0.1253 | 0.1250 | 1.0000 |
| 0.5 | 5 | 1.9470 | 1.9375 | 0.5035 | 0.5000 | 0.0305 | 0.0313 | 1.0000 |
