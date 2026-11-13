# Does a lane depend on its batch partner or position? Lane-layout check (plan; run 13 November 2026, results at the end)

Written 13 November 2026 (UTC) before any model call. The human asked for plans and for the result to be
"unimpeachable". Evidence class when run: prospective.

## Why

Every register run carried two lanes in one batch: the lane started on cycle 0 first, the lane started on cycle 1
second. The referee and Codex (Task 7 item 3) ask whether a lane's trajectory or outcome depends on that layout. The
earlier study found that batch layout can change chaotic loops even in float64.

## What is already known (from audited receipts; not rerun here)

- **Single lane against a pair, holds.** On 26 October two 600-update trajectories at H, run alone and in the pair
  instrument, differed by at most 4.9e-15, with zero token mismatches and the same final labels
  (`docs/receipts/qwen-batch-trajectory.json`). Single forward passes at H agreed to 7.4e-16
  (`docs/receipts/qwen-batch.json`).
- **Single lane against a pair, write of 1.** The pair instrument's write trajectories matched the original
  single-lane write runs to 1.0e-14 and 2.3e-14, with zero token mismatches and the same final labels
  (`docs/receipts/public-batch-writes.json`).
- **Not known.** Write 0 and the toggle have not been compared across layouts. Neither has any operation with the
  lanes reversed, or with a lane paired with a copy of itself.

## Why there is no single-lane arm here

The audited pair instrument checks that every batch has exactly two lanes: its return check, block format and cache
check all require shape (2, 896). The original single-lane instrument runs only its fixed write protocols and
refuses other controls. A single-lane toggle would need a modified or new instrument, with its own audit. That is not
done here. If wanted, it belongs with an independently written implementation (Codex's later step), as a separate
plan.

## What runs

- **Instrument.** The toggle check's pair instrument, unchanged, through a new runner,
  `tools/run_public_lane_layout.py`. It differs from `tools/run_public_toggle_check.py` only in its list of groups
  and in placing the starts.
- **Four operations**, each a move stage followed by 240 updates at H and the neutral readout:
  - W: 12 idle updates at H;
  - E0: the audited 120-update write of 0;
  - E1: the audited 120-update write of 1;
  - T: the 30-degree, radius-3.5, 12-update toggle.
- **Four layouts of the starts** (reference cycles at phase 0):
  - P, original: cycle 0 then cycle 1;
  - R, reversed: cycle 1 then cycle 0;
  - D0: cycle 0 twice;
  - D1: cycle 1 twice.
- **Calls.** 16 groups (four operations by four layouts): per layout 258 + 366 + 366 + 258 = 1,248, so 4,992 in
  all. One group per single-thread process, at most ten at a time, in the same session as the full-adder runs.

## Predictions (fixed now)

- **L1 (primary, outcomes).** Every one of the 32 lanes reads its expected cycle:
  - W keeps each lane's start;
  - E0 gives 0;
  - E1 gives 1;
  - T exchanges the start.

  The expected pairs:

  | Operation | P | R | D0 | D1 |
  |---|---|---|---|---|
  | W | (0,1) | (1,0) | (0,0) | (1,1) |
  | E0 | (0,0) | (0,0) | (0,0) | (0,0) |
  | E1 | (1,1) | (1,1) | (1,1) | (1,1) |
  | T | (1,0) | (0,1) | (1,1) | (0,0) |

- **L2 (primary, trajectories).** For each operation and start cycle, every lane with that start follows the same
  path in every layout and position: the largest coordinate gap at every update, over the move and the hold, is at
  most 1e-10. This is the threshold of the 26 October comparison. Each start occurs four times (P, R and both lanes of
  its duplicate layout), and the other three are compared with the P lane: exactly 24 comparisons in all. Missing
  coverage fails L2; it never passes vacuously. L1 and L2 are reported separately: right labels with an excess path
  gap count as a functional success and an equivalence failure.
- **Reported, not scored:**
  - whether the paths are bitwise identical across layouts;
  - the largest gap per operation;
  - the gap between P and earlier runs of the same operation from the same starts, each bound by hash: the toggle
    against the phase-0 toggle-check run (pod, 12 + 240 updates) and against Part B's full-offset run (this machine,
    its 6 pulse updates); the writes against the sweep's pod baselines (120 + 240); the hold against the sweep's hold
    baseline over the first 252 updates at H (both at H throughout, but the baseline changes stage at update 120 and
    the W run at update 12);
  - the measured cycle and phase at the end of every stage.

A failure of L1 or L2 is reported as found, and the paper says so.

## Rules

- Plan, runner and launcher are committed and pushed before the first call; the go-ahead names that commit.
- Preflight with no model load. One run; no rerun or replacement; every outcome and every operational stop is
  reported, partial evidence included.
- **Failure categories.** A missing job, a timeout, or a failed instrument check is an operational or instrument
  failure, reported separately from a qualified wrong label or an erasure. Instrument checks cover job identity,
  cache, precision, controls and encodings (the hold's H included), lengths, the ledger, references, model inputs
  and the source commit.
- Scoring: `tools/audit_public_lane_layout.py`, with zero calls. It rehashes every file, checks starts, controls and
  stage lengths, recomputes every readout from saved states, and computes the gaps. Codex is asked to audit the
  returned runs independently before the paper uses them.

## What this can and cannot show

It can show whether, on this machine and with this instrument, a lane's path and outcome depend on what it is paired
with or on its position in the batch, for the four operations the paper uses. It cannot show the single-lane case for
write 0 or the toggle, another computer or implementation, or anything about batch sizes above two.

## Execution, fixed before any call (13 November 2026)

- **Code.**
  - Runners: `tools/run_public_full_adder.py` (the audited adder runner; the only differences are one-bit digits
    from `--x`/`--y` and a preflight stop before the model loads) and `tools/run_public_lane_layout.py`.
  - Launcher: `tools/launch_full_adder_and_lane_layout.sh` runs 20 single-thread jobs (4 adder, 16 lane), at most
    ten at once.
  - Audits: `tools/audit_public_full_adder.py` and `tools/audit_public_lane_layout.py`. Both are written before any
    run, so the scoring code is fixed too.
- **Preflight passed for all 20 jobs** with no model load and zero calls. Planned calls: adder 882, 774, 774 and
  882; lane jobs 258 (W and T) and 366 (E0 and E1); 8,304 in all.
- **Freeze.** Plans, runners, launcher and audits are committed and pushed together before the first call. The
  go-ahead names that commit.
- **Wall time.** About 2 to 3 hours on this machine (two waves of up to ten processes).

## Revision after Codex's pre-run review (13 November 2026, before any call)

Codex's review (`docs/audits/2026-11-13-task8/review.md`) reproduced audit and launcher defects. The repairs are those
listed in `docs/public-full-adder-truth-table.md` (same section), plus three for this audit:

- the schedule is literal: job k is operation k // 4 under layout k % 4, and each job's identity is checked against
  its directory, so sixteen copies of one job no longer pass;
- L2 requires complete coverage, as above;
- the Part B comparison is implemented with its files bound.

**Single lane (open, the human's decision).** These four layouts answer partner and position at batch size two
only. Codex points out a bounded route to a true single-lane arm. `benchmark_qwen_batch.QwenBackend.forward` already
runs one state through the scalar loop, so a thin harness with strict single-lane checks would not modify the pair
instrument: the four operations from both phase-0 starts, eight trajectories, about 2,496 more calls. It needs its
own code, tests and review. Until the human decides, this check's claim is limited to batch size two.

## Second revision after Codex's recheck (13 November 2026, before any call)

The shared repairs are listed in `docs/public-full-adder-truth-table.md` (same section). For this audit:

- **Scaffold (R4).** Each job's result keeps the runner's whole plan: 32 stages and 32 rows, of which only its own two
  are executed. All are checked in order against the literal scaffold. The job's own stages must be complete. The
  other 30 must be unstarted and carry no evidence. Rows are checked the same way, and the job's two rows must hold
  the recomputed bits.
- **Missing baselines (R3).** A comparison path is built only when both P occurrences of its operation exist and
  qualified. A missing or failed P job leaves coverage incomplete and that operation's earlier-run comparisons
  unavailable, and the receipt is still written.
- **Earlier runs (R6).** Each comparison qualifies on its own: one group file with the literal group name, the
  manifest, an ordered stage list, stage names, controls, lengths and completion, and every file read bound. A
  comparison that does not qualify is reported unavailable with its reasons, and L1 and L2 are untouched. While
  repairing this I found that the Part B comparison read stage 0 of shard-04. The pulse that ran there is stage 4
  (offset 1.00); stage 0 is an unstarted record. It now reads stage 4 by its literal identity. On the real run, the
  old code would have produced no Part B comparison, only a recorded problem. All five earlier runs used here qualify under the strict
  checks (zero calls).

## Single-lane arm (added 13 November 2026 at the human's request, before any call)

The human approved adding it ("Yes add it"). It answers what the four two-lane layouts cannot: whether a lane's path
depends on being computed in a batch at all. Earlier single-lane evidence covers only the hold (4.9e-15) and write 1
(2.3e-14). This arm adds write 0 and the toggle, and repeats the other two under this check's controls.

- **Design.** Eight groups: each of the four operations (W, E0, E1, T, with the controls and lengths above and the
  same 240-update hold at H and neutral readout) from each phase-0 reference start, one state at a time. Job k
  (`single-00` to `single-07`) is operation k // 2 from cycle k % 2. One group per job, one single-thread process per
  job. Planned calls: 258 for W and T, 366 for E0 and E1; 2,496 in all.
- **Instrument.** `tools/run_public_lane_singleton.py` reuses the pair instrument unchanged: the backend and its
  control setting and prefix call; the guarded factory (native64, one thread; no update unless the stage's cached and
  full comparisons agree); the budget, recorder, block format, call ledger, group closing and readout. The backend's
  forward sends a one-state input through `FeedbackMap.forward`, the canonical scalar map. The pair code fixes the
  batch at two in three places, so those are written out for one lane rather than changed. The return check requires
  (1, 896) and (1, vocabulary). A cached call uses the scalar map's own copy of the stored prefix cache, so each call
  must show that no batch expansion happened, and the stored cache must be unchanged before and after. The archive
  and its reader hold (n, 1, 896) blocks.
- **Predictions, fixed now.**
  - S1: each single lane ends on its expected cycle: W on its start cycle, E0 on 0, E1 on 1, T on the other cycle.
  - S2: each single-lane path is within 1e-10, at every update, of all four two-lane occurrences of the same
    operation and start cycle in the lane-layout run (P, R and the duplicate layout). That is 32 comparisons, each
    two-lane job qualified by the lane audit's own checks. Missing coverage makes S2 incomplete, not passed.
  - Reported: the largest gap and how many of the 32 are bitwise identical.
- **Audit.** `tools/audit_public_lane_singleton.py`, written before any run, with the same instrument checks as the
  other two audits at one lane. It also checks the runner's whole 16-stage, 8-row plan in order and the scalar path
  on every comparison and update. S2 takes `--lane-run`, the two-lane execution root.
- **Failure categories** as above: a missing job or failed instrument check is an operational or instrument failure,
  not an outcome.
- **What it can show.** Whether, on this machine and with this instrument, the four operations give the same paths
  and outcomes at batch size one as at batch size two. It cannot show another computer or implementation, batch
  sizes above two, or other operations.
- **Execution.** `tools/launch_full_adder_and_lane_layout.sh MODE SOURCE FROZEN ROOT single-lane` runs the eight jobs
  under the same controller, in their own root, after the two-lane launch has finished. The controller holds a lock,
  so two launches can never run at once. Two controllers would each count the same free memory.
- **Preflight passed** for all eight jobs in this environment, with no model load and zero calls; the planned
  calls total 2,496.
- **Tests.** `tests/test_lane_singleton.py` runs the runner end to end with a fake one-state backend for every job.
  It also checks the refusal of a batch expansion, a two-lane return and a wrong shard count, and the audit's
  positive, fault, coverage and divergence cases; 25 pass.
- **Freeze.** The same pushed commit as the rest; the go-ahead names it.

## Results (13 November 2026)

The jobs ran from the frozen commit 39e855e (see the research log); the audits were run with the frozen code.

- **Two lanes.** All 16 jobs exited cleanly, with 4,992 calls. The audit is `docs/receipts/public-lane-layout.json`.
  - Every instrument check passed.
  - **L1 passed:** all 32 lanes ended on their expected cycles.
  - **L2 passed:** coverage was complete (24 comparisons), and every path is bit-for-bit identical to its P path
    (largest gap 0.0).
  - Against earlier runs, Part B on this machine is identical (0.0). The pod's runs differ by 2.5e-13 (toggle
    check), 1.0e-13 (write-0 baseline), 3.0e-12 (write-1 baseline) and 4.9e-14 (hold baseline).
- **Single lane.** All 8 jobs exited cleanly, with 2,496 calls. The audit is `docs/receipts/public-lane-singleton.json`.
  - Every instrument check passed.
  - **S1 passed:** all 8 lanes ended on their expected cycles.
  - **S2 passed:** all 32 comparisons are within 1e-10. The largest gap is 1.448e-12, for write 1 from cycle 0; the
    others are 7e-15 to 2e-13. None is bit-for-bit identical, since one state through the scalar map and two through
    the batched call round differently.
- **What this shows.** On this machine and instrument, for the four operations, a lane's path at batch size two does
  not depend on its partner or position, down to the last bit. Batch size one gives the same outcomes, with paths
  within 1.5e-12 of batch size two.
