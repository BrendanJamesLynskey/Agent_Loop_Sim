# Policy module results (engine 1.11.1)

Written by `scripts/make_fixtures.py` (CI checks it is up to date). The tools, plans, grants, sandbox actions and every layer miss rate marked illustrative are abstract placeholders; the decisions follow from the rules. The published measurements are quoted from their papers (arXiv IDs below).

## Plans under the broad grant (chapters 1 and 2)

| plan | classic agent: allow / ask / deny | planned in advance: allow / ask / deny | last call, classic | last call, planned |
|---|---|---|---|---|
| Summarise my notes into a file | 3 / 0 / 0 | 3 / 0 / 0 | allow | allow |
| Read my inbox and notes, then reply to a colleague | 3 / 0 / 1 | 3 / 0 / 1 | deny | deny |
| Send a colleague a summary of my notes, after looking up a page | 2 / 1 / 1 | 3 / 1 / 0 | deny | ask |
| Send me a digest of my inbox | 3 / 0 / 0 | 3 / 0 / 0 | allow | allow |
| Read a web page and save a summary | 2 / 1 / 0 | 2 / 1 / 0 | ask | ask |
| Share a public ticket's status with a colleague | 3 / 0 / 0 | 3 / 0 / 0 | allow | allow |

## The trifecta (chapter 3)

Over all 256 subsets of the 8 tools: 136 hold all three circles; the trifecta check and the exhaustive search for a flow out agree on 256 of 256.

| circles held | trifecta | flow out found | sequences checked |
|---|---|---|---|
| none | no | none | 0 |
| private | no | none | 0 |
| untrusted | no | none | 0 |
| private+untrusted | no | none | 0 |
| outbound | no | none | 2 |
| private+outbound | no | none | 6 |
| untrusted+outbound | no | none | 6 |
| private+untrusted+outbound | yes | read_notes → read_ticket → send_message (org) | 9 |
| read_inbox+send_message | yes | read_inbox → send_message (org) | 3 |
| read_notes+fetch_page | yes | read_notes → fetch_page → fetch_page (web) | 5 |

## Least privilege (chapter 4)

Plan: Summarise my notes into a file, and one call it does not need. Broad grant: 8 tools; least privilege: 3 (read_notes, summarise, write_file). Unneeded under the broad grant: read_inbox, read_ticket, search_docs, fetch_page, send_message.

| step | broad grant | least privilege |
|---|---|---|
| read_notes | allow | allow |
| summarise | allow | allow |
| write_file | allow | allow |
| send_message | ask | deny |

A token minted for notes-api presented to send_message: the token is for notes-api, not mail-api: rejected.

## Sandbox layers (chapter 5)

| layers | needed actions working | needed actions broken | unneeded stopped | unneeded still open |
|---|---|---|---|---|
| none | 3 | 0 | 0 | 5 |
| fs_scope | 3 | 0 | 2 | 3 |
| fs_scope + container | 3 | 0 | 3 | 2 |
| fs_scope + container + egress | 3 | 0 | 4 | 1 |

## Defence in depth (chapter 6)

Base rate 24.0% (InjecAgent, arXiv 2403.02691; up to 50% as in Spotlighting's baseline, arXiv 2403.14720). Layers: Provenance marking misses 4.0% (ratio of Spotlighting’s bounds); Input classifier misses 20.0% (illustrative); Output check misses 30.0% (illustrative). Independent, all three miss: 0.24%.

| shared blind spot c | layers | all miss | end to end | at least one of 1,000 gets through |
|---|---|---|---|---|
| 0.0% | 0 | 100.00% | 24.000% | 100.0% |
| 0.0% | 1 | 4.00% | 0.960% | 100.0% |
| 0.0% | 2 | 0.80% | 0.192% | 85.4% |
| 0.0% | 3 | 0.24% | 0.058% | 43.8% |
| 0.5% | 0 | 100.00% | 24.000% | 100.0% |
| 0.5% | 1 | 4.00% | 0.960% | 100.0% |
| 0.5% | 2 | 1.19% | 0.285% | 94.2% |
| 0.5% | 3 | 0.70% | 0.169% | 81.5% |
| 1.0% | 0 | 100.00% | 24.000% | 100.0% |
| 1.0% | 1 | 4.00% | 0.960% | 100.0% |
| 1.0% | 2 | 1.58% | 0.378% | 97.7% |
| 1.0% | 3 | 1.17% | 0.280% | 94.0% |
| 2.0% | 0 | 100.00% | 24.000% | 100.0% |
| 2.0% | 1 | 4.00% | 0.960% | 100.0% |
| 2.0% | 2 | 2.37% | 0.568% | 99.7% |
| 2.0% | 3 | 2.10% | 0.505% | 99.4% |
| 4.0% | 0 | 100.00% | 24.000% | 100.0% |
| 4.0% | 1 | 4.00% | 0.960% | 100.0% |
| 4.0% | 2 | 4.00% | 0.960% | 100.0% |
| 4.0% | 3 | 4.00% | 0.960% | 100.0% |

Simulation (20,000 attempts per c, seed 2601 + i): c 0.0%: 0.26%; c 0.5%: 0.70%; c 1.0%: 1.21%; c 2.0%: 2.17%; c 4.0%: 4.17%.

## Published measurements

| measurement | value | arXiv |
|---|---|---|
| InjecAgent: share of its test cases in which a ReAct-prompted GPT-4 agent was steered by text in a tool result (base setting; the enhanced setting nearly doubled it) | 24% | 2403.02691 |
| InjecAgent: test cases (17 user tools, 62 attacker tools) | 1,054 | 2403.02691 |
| Spotlighting: attack success rate without the defence (GPT-family models; a lower bound) | > 50% | 2403.14720 |
| Spotlighting: attack success rate with it (an upper bound) | < 2% | 2403.14720 |
| AgentDojo: realistic user tasks | 97 | 2406.13352 |
| AgentDojo: security test cases | 629 | 2406.13352 |
| CaMeL: AgentDojo tasks solved with provable security | 77% | 2503.18813 |
| CaMeL: the same tasks solved by the undefended system | 84% | 2503.18813 |
