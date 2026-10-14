# Policy module results, part two (engine 1.11.1)

Written by `scripts/make_fixtures.py` (CI checks it is up to date). The benign task suite, the detector false-positive rates of the classifier and the output check, the approval miss rates and fatigue are illustrative; the decisions follow from the rules. Base rates and provenance marking's rates come from the published measurements (arXiv IDs in `policy_results.md`); the adaptive setting follows arXiv 2503.00061 (all eight defences tested bypassed, attack success over 50%).

## The benign suite

120 tasks (seed 2611); 25 branch on what their last read returned. Sinks: answer the user: 26; send_message → org: 21; send_message → self: 27; send_message → web: 18; write_file: 28.

## The frontier: fixed attacks, base rate 24% (InjecAgent) (chapter 9)

| defence | tasks finished | blocked | prompts / task | extra calls / task | risk / task | unchecked route | on the frontier |
|---|---|---|---|---|---|---|---|
| No defence | 100.0% | 0.0% | 0.00 | 0.00 | 20.60% | 85.8% | no |
| Provenance marking | 100.0% | 0.0% | 0.00 | 0.00 | 0.82% | 85.8% | yes |
| Three detector layers | 95.1% | 0.0% | 0.00 | 2.09 | 0.24% | 85.8% | yes |
| Approve every write and send | 100.0% | 0.0% | 0.78 | 0.00 | 4.27% | 0.0% | no |
| Label policy | 72.5% | 27.5% | 0.30 | 0.00 | 12.32% | 50.0% | no |
| Label policy, no approvals | 45.8% | 54.2% | 0.00 | 0.00 | 11.60% | 48.3% | no |
| Label policy + least privilege | 72.5% | 27.5% | 0.30 | 0.00 | 7.96% | 30.8% | no |
| Plan-then-execute + policy + least privilege | 76.7% | 23.3% | 0.25 | 1.00 | 1.14% | 4.2% | no |

## The frontier: an adaptive attacker, base rate 50%, detectors passed (chapter 9)

| defence | tasks finished | blocked | prompts / task | extra calls / task | risk / task | unchecked route | on the frontier |
|---|---|---|---|---|---|---|---|
| No defence | 100.0% | 0.0% | 0.00 | 0.00 | 42.92% | 85.8% | yes |
| Provenance marking | 100.0% | 0.0% | 0.00 | 0.00 | 42.92% | 85.8% | yes |
| Three detector layers | 95.1% | 0.0% | 0.00 | 2.09 | 42.92% | 85.8% | no |
| Approve every write and send | 100.0% | 0.0% | 0.78 | 0.00 | 8.90% | 0.0% | yes |
| Label policy | 72.5% | 27.5% | 0.30 | 0.00 | 25.67% | 50.0% | no |
| Label policy, no approvals | 45.8% | 54.2% | 0.00 | 0.00 | 24.17% | 48.3% | yes |
| Label policy + least privilege | 72.5% | 27.5% | 0.30 | 0.00 | 16.59% | 30.8% | no |
| Plan-then-execute + policy + least privilege | 76.7% | 23.3% | 0.25 | 1.00 | 2.38% | 4.2% | yes |

## What blocks benign tasks (fixed attacks)

| defence | not-granted | scope | destination | untrusted-egress-private | private-egress | untrusted-write | needs-input |
|---|---|---|---|---|---|---|---|
| none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| marking | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| detectors | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| approve_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| policy | 0 | 0 | 0 | 33 | 0 | 0 | 8 |
| policy_refuse | 0 | 0 | 0 | 32 | 12 | 21 | 16 |
| least | 0 | 0 | 0 | 33 | 0 | 0 | 8 |
| planned | 0 | 0 | 0 | 28 | 0 | 0 | 0 |

## Approval thresholds over the label policy (chapter 8)

| threshold | tasks finished | prompts / task | risk / task | unchecked route |
|---|---|---|---|---|
| refuse | 45.8% | 0.00 | 11.60% | 48.3% |
| policy | 72.5% | 0.30 | 12.32% | 50.0% |
| consequential | 72.5% | 0.59 | 2.41% | 0.0% |
| every | 72.5% | 3.64 | 7.36% | 0.0% |

Approval fatigue (illustrative): a mistaken approval at prompt n of a task: 1: 10.0%; 2: 23.5%; 3: 35.0%; 4: 44.7%; 5: 53.0%; 6: 60.1%; 7: 66.1%; 8: 71.1%; 9: 75.5%; 10: 79.2%.

## Attempts through the layers (chapter 6)

400 attempts, seed 2613, all three layers.

| shared blind spot c | attempts | steer the model | + marking | + classifier | + output check |
|---|---|---|---|---|---|
| 0.0% | 400 (400.00) | 91 (96.00) | 3 (3.84) | 2 (0.77) | 2 (0.23) |
| 0.5% | 400 (400.00) | 91 (96.00) | 4 (3.84) | 3 (1.14) | 3 (0.68) |
| 1.0% | 400 (400.00) | 91 (96.00) | 3 (3.84) | 2 (1.51) | 2 (1.12) |
| 2.0% | 400 (400.00) | 91 (96.00) | 4 (3.84) | 3 (2.27) | 3 (2.02) |
| 4.0% | 400 (400.00) | 91 (96.00) | 3 (3.84) | 3 (3.84) | 3 (3.84) |

(Counts drawn; expected values from the closed form in brackets.)

## Plan-then-execute (chapter 7)

| plan | planned: allow / ask / deny | classic: allow / ask / deny |
|---|---|---|
| Send a colleague a summary of my notes, after looking up a page | 3 / 1 / 0 | 2 / 1 / 1 |
| Read a web page and save a summary | 2 / 1 / 0 | 2 / 1 / 0 |
| Read my inbox and notes, then reply to a colleague | 3 / 0 / 1 | 3 / 0 / 1 |
| File a ticket's summary wherever its category says | 2 / 1 / 0 | 2 / 1 / 0 |

Simulation of plan-then-execute (400 draws of each task, seed 2612): risk 1.04% (closed form 1.14%), tasks finished 76.7%.

## The checklist (chapter 10), with every sandbox layer

| design | passed | ch 1 | ch 2 | ch 3 | ch 4 | ch 5 | ch 6 | ch 7 | ch 8 | ch 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 2 / 9 | fail | fail | fail | fail | pass | fail | fail | pass | warn |
| none + approve writes and sends | 2 / 9 | fail | pass | fail | fail | pass | fail | fail | warn | warn |
| marking | 2 / 9 | fail | fail | fail | fail | pass | warn | fail | pass | warn |
| marking + approve writes and sends | 2 / 9 | fail | pass | fail | fail | pass | warn | fail | warn | warn |
| detectors | 2 / 9 | fail | fail | fail | fail | pass | warn | fail | pass | fail |
| detectors + approve writes and sends | 2 / 9 | fail | pass | fail | fail | pass | warn | fail | warn | fail |
| approve_all | 2 / 9 | fail | pass | fail | fail | pass | fail | fail | warn | warn |
| approve_all + approve writes and sends | 2 / 9 | fail | pass | fail | fail | pass | fail | fail | warn | warn |
| policy | 4 / 9 | pass | fail | warn | fail | pass | pass | warn | pass | fail |
| policy + approve writes and sends | 4 / 9 | pass | pass | warn | fail | pass | pass | warn | warn | fail |
| policy_refuse | 4 / 9 | pass | fail | warn | fail | pass | pass | warn | pass | warn |
| policy_refuse + approve writes and sends | 5 / 9 | pass | pass | warn | fail | pass | pass | warn | pass | warn |
| least | 5 / 9 | pass | fail | warn | pass | pass | pass | warn | pass | fail |
| least + approve writes and sends | 5 / 9 | pass | pass | warn | pass | pass | pass | warn | warn | warn |
| planned | 7 / 9 | pass | warn | warn | pass | pass | pass | pass | pass | pass |
| planned + approve writes and sends | 7 / 9 | pass | pass | warn | pass | pass | pass | pass | warn | pass |
