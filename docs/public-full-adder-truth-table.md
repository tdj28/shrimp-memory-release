# All eight input cases of a full adder in the register (plan; run 13 November 2026, results at the end)

Written 13 November 2026 (UTC) before any model call. The human asked for plans and for the result to be
"unimpeachable". Evidence class when run: prospective, with the move set and pulse selected earlier (after the
toggle sweep), which Table III already records.

## Why

The paper's addition result is one fixed case, 11 + 6, run on two computers with all 24 readouts correct. That case
exercises only some of the eight input combinations (carry c, digits x and y) and never x = y = 0. The referee and
Codex (Task 7 item 3) ask for the complete truth table. This plan runs every combination once, from fresh starts,
with the instrument, moves, readout and gates of the 11 + 6 run unchanged.

## What runs

- **Instrument.** `tools/run_public_full_adder.py` is a copy of the audited `tools/run_public_adder.py` (corrected
  after Codex's review: a copy, not an import). Only the digits (one bit each, from `--x` and `--y`), metadata and a
  preflight stop before the model loads differ. A regression test checks that its `moves`, `expected`, `decode`,
  `controls` and `plans` functions are identical to the audited file's.
- **Moves**, as in Stage 3 of `docs/public-toggle-sweep.md`. Each move is followed by 240 updates at H and the
  neutral readout:
  - T: the 30-degree, radius-3.5, 12-update toggle;
  - W: 12 idle updates at H;
  - E0 and E1: the audited 120-update writes.
- **Digit rule:**
  - step 1: T if x = 1, else W;
  - step 2: T if y = 1, else W (the register then holds the sum bit);
  - step 3: E_x if x = y, else T (the register then holds the carry out).

  Moves depend on x and y only; the register is never read to choose a move.
- **Four groups**, one per digit pair (x, y) = (0,0), (0,1), (1,0), (1,1). Each has two lanes sharing every move:
  lane 0 starts on cycle 0 at phase 0 (carry in 0), lane 1 on cycle 1 at phase 0 (carry in 1). The four groups
  cover all eight (c, x, y). Every move plus hold is 252 or 360 updates, a whole number of periods, so each move
  starts at phase 0.
- **Calls:** (0,0) is W, W, E0, 882 calls; (0,1) is W, T, T, 774; (1,0) is T, W, T, 774; (1,1) is T, T, E1, 882.
  Total 3,312, including three cache-check calls per stage. One group per single-thread process on this machine, in
  the same wave as the lane-layout check (at most ten processes at once; each needs about 5.5 GB).

## Predictions (fixed now)

Register after steps 1, 2 and 3 (c XOR x, then the sum c XOR x XOR y, then the carry maj(c, x, y)):

| (x, y) | lane 0 (c = 0) | lane 1 (c = 1) |
|---|---|---|
| (0, 0) | 0, 0, 0 | 1, 1, 0 |
| (0, 1) | 0, 1, 0 | 1, 0, 1 |
| (1, 0) | 1, 1, 0 | 0, 0, 1 |
| (1, 1) | 1, 0, 1 | 0, 1, 1 |

- **A1 (primary).** All 24 readouts are qualified and equal the table. Any mismatch or unresolved readout fails the
  truth table, and is reported with its group, lane and step.
- **A2 (reported).** The full-adder table read from the sum and carry readouts. For each move and lane, the settle
  update: the first hold update after which the lane stays within 1e-6 of its final cycle through the end of the
  hold. Distances are phase-aligned by counting back from the measured end phase; a lane not within 1e-6 at the end
  is reported as censored. Also reported: the measured cycle and phase at the end of every move and every hold. Group (1, 0) repeats the first three moves of the 11 + 6 run
  exactly (T, W, T from the same starts), so its states are compared with that run's saved states on this machine,
  and the largest gap is reported.

## Rules

- Plan, runner and launcher are committed and pushed before the first call; the go-ahead names that commit.
- Preflight with no model load first. One run; no rerun or replacement after any outcome. Every result is reported,
  including failures and operational stops. A stopped job's partial evidence is kept and reported as partial,
  with no silent retry.
- **Failure categories.** A missing job, a timeout, or a failed instrument check is an operational or instrument
  failure, reported as such and not as a wrong truth-table row. Instrument checks cover cache, precision, controls
  and their encodings, stage lengths, the call ledger, references, model inputs and the source commit. A1 requires
  every instrument check to pass.
- Scoring: an audit script, `tools/audit_public_full_adder.py`, with zero calls. It rehashes every file, checks
  starts, controls and stage lengths, and recomputes every readout from saved states with the readout function of
  `tools/audit_public_adder.py`. Codex is asked to audit the returned runs independently before the paper uses them.

## What this can and cannot show

It can show that each of the eight full-adder input cases, run once from the phase-0 cycles, gives the right sum and
carry. It cannot show reliability (one run per case), robustness to start phase or perturbations, or behavior on
another computer. The toggle used was chosen after the sweep (Table III).

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

Codex's review (`docs/audits/2026-11-13-task8/review.md`, with model-free fault probes) found that the first versions
of the audits and the launcher could accept malformed or incomplete evidence. Changes:

- **Audits.** Both are rebuilt on a shared module (`tools/task8_audit_common.py`) with literal controls, steps and
  call shapes. The truth table and job list are literal, not taken from the producer. They check every job's
  identity, the exact stage set, every step count, controls and bit encodings (the hold's H included), block
  controls, cache and final-cache gates, failed outputs, the call ledger and float64 prefix evidence, native64
  one-thread readiness, model-input hashes against the canonical audited run, reference cycles against the canonical
  file, and the frozen source commit with a clean checkout. A missing or broken job is recorded and the receipt is
  still written.
- **Launcher.** Separate preflight and execution roots; an existing root is refused; it runs only from a clean,
  isolated checkout of the frozen commit. Each job's exit status is kept and returned, and a summary lists every
  scheduled job. There is an external hard deadline per job, a stop trap for its own jobs, and memory admission
  before each job (at most ten at once).
- **Regression tests.** `tests/test_task8_revised.py` covers every fault Codex reproduced, plus missing and broken
  jobs and launcher refusals and failures; 23 pass.
- **Freeze.** One pushed commit contains the plans, runners, launcher, audits, tests and every module they import
  (24 modules from `tools/`, nine of them untracked until then), and the run uses an isolated checkout of that
  commit.

## Second revision after Codex's recheck (13 November 2026, before any call)

Codex's recheck (`docs/audits/2026-11-13-task8-recheck/review.md`) verified the first repairs and found seven
remaining contract gaps (R1 to R7). None changes a prediction, a tolerance or the moves. Changes:

- **Launcher (R1, R2).** `tools/launch_full_adder_and_lane_layout.sh` now starts one controller,
  `tools/launch_controller.py` (standard library only), in place of the `xargs` queue. The controller makes every
  start decision itself. It admits a job only while fewer than `MAX_JOBS` run and MemAvailable still holds a whole job
  budget plus the reserve after subtracting the part of each running job's budget that is not yet resident (budget
  minus the anonymous resident memory of the job's process group). Jobs are re-admitted as others finish. Each job runs
  in its own session. A stop signal reaches only those sessions: TERM, KILL after the grace period, then a record for
  every scheduled job and the summary. The per-job deadline is enforced the same way. This machine's `timeout` is not
  GNU coreutils, so no external timeout program is used. Limits are validated as positive integers, and the memory
  sensor can be pointed at a controlled file for tests.
- **Audits (R4, R5, R7).** Both audits check the complete ordered stage list against literal identities. They also
  check the literal ordered call ledger: each call's number, stage, group, category, step, cached flag, status and its
  own evidence. Every saved array is checked on its own before anything is joined, so a float32 block cannot be
  promoted unseen. The initial state is checked against its stage-record binding (size, SHA-256, array digest, shape,
  dtype), as the blocks are. Reconstruction returns nothing if any of its checks fails.
- **The 11 + 6 comparison (R6).** It is secondary and qualified on its own. It must be the saved `public-adder` run of
  11 + 6, with its manifest and stage records checked and every file it reads bound. Its value is reported only if all
  of its own checks pass. Otherwise it is reported unavailable with the reasons, and A1 is untouched. Both saved 11 + 6
  runs qualify under these checks (zero calls).
- **Regression tests.** `tests/test_task8_revised.py` now builds its fixtures from the runners' own planned scaffolds
  and turns each of Codex's recheck probes into a rejection or preservation test, besides the earlier ones; 86 pass.

## Results (13 November 2026)

The jobs ran from the frozen commit 39e855e: commit e27594d plus one unchanged file (see the research log). All four
jobs exited cleanly, with 3,312 calls between 20:14 and 20:41 UTC. The audit, `docs/receipts/public-full-adder.json`,
was run with the frozen code.

- Every instrument check passed.
- **A1 passed.** All 24 readouts match the literal truth table: the sum and the carry are right for all eight
  combinations of carry in, x and y, at every step.
- **Group (1, 0)** is bit-for-bit identical to the first three moves of the saved 11 + 6 run on this machine
  (largest gap 0.0).
- Settle updates and end phases are in the receipt.
