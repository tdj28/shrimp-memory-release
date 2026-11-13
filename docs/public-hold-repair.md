# Public repair from early and settled hold states

This adapted exploratory test compares nominal repair with equal-duration
continued hold, separately from two literal ages of a failed write history.
It concerns explicit public numerical memory and control. It is not a language
experiment, an online feedback controller or an error-correcting code.

## Adaptation and fixed inputs

The completed endpoint-continuation comparison found all four unchanged
perturbed continuations wrong and all four nominal restorations correct
([receipt](receipts/public-control-continuation.json)). Those interventions
started before any common-hold update. The present design asks whether the
same nominal action works after the failed write has spent 36 or 240 updates
at the hold. Earlier readout analyses motivated age 36; visibility of a label
does not establish readiness for another write.

Use only the failed full-offset group of the original two-phase challenge,
source commit `739a4fd678e35364236799d3701463a4d7b0009d`. Its endpoint is
stage 2 and hold is stage 3. Each requested-bit process has two origins:
td phases 0/1 for WRITE0 and ad phases 0/1 for WRITE1. These labels describe
the references before the original endpoint, not the current transient phase.
Extract literal post-update hold states 36 and 240, with state zero the
endpoint handoff. No new settling or reference replacement is permitted.

The immutable input packets bind all 39 consumed source files, the complete
independent source audit, exact model/reference identity, origin ordering and
all four selected row-array hashes. NPZ keys are `states` (float64[2,2,896]),
`hold_ages` (int64[2], 36 and 240), `origin_reference_index` and
`origin_reference_phase` (int64[2]). The first array axis selects hold age;
the second retains source-origin phase order 0/1.

The packets also preserve the unchanged geometric/strict observers applied to
each truncated hold prefix. Every input is retained regardless of those
measured labels or qualification. Runtime qualification rechecks their prefix
hashes and metadata; the independent audit checks the observations numerically.
No requested or future label enters the prefix observer.

## Four fixed groups per requested bit

| ID | Group | Literal input age | First stage, 120 updates | Second stage, 240 updates |
|---|---|---:|---|---|
| 0 | hold36-continue-hold | 36 | Common hold | Common hold |
| 1 | hold36-restore-nominal | 36 | Nominal write endpoint | Common hold |
| 2 | hold240-continue-hold | 240 | Common hold | Common hold |
| 3 | hold240-restore-nominal | 240 | Nominal write endpoint | Common hold |

The first stage is called `intervention` in every group; the second is `hold`.
Even the two continued-hold stages use separately constructed, checked caches.
Each group initializes its exact age batch once, independently of prior group
outcomes. There are four hold-state initializations, three between-group
resets, zero reference initializations, and no reset within a group. Mutation
counters and initial hashes are recorded before any group-specific file write.
A finite wrong or unresolved result never selects or skips a later group.

The parsed public WRITE0/WRITE1 action remains canonical (120 endpoint followed
by 240 hold); the separate `public-hold-repair` profile explicitly records the
realized control stages, including the continued-hold override. The nominal
controls remain alpha 0.07293697809562831, log10 temperature -0.1535 for WRITE0,
and alpha 0.07299785589927693, log10 temperature -0.15395151820007266 for WRITE1.
The common hold is alpha 0.07293697809562831 and log10 temperature
-0.15395151820007266. Shared controller modules and globals are unchanged.

## Gates, file contracts and failure preservation

Before model load, freeze every group, row, stage, call slot and parsed action.
`hold-inputs.json` records the verified packet and selected arrays; the initial
freeze event hashes the whole age cube. Each group-start event hashes only its
selected batch. Result rows retain source row 2/3, source stage 3, age index,
historical endpoint/hold updates, and the actual executed stages.

Use the existing one-thread native-float64 backend, effective normalization
and rotary patches, CPU math attention, and per-stage cached/full-prefix
coordinate/probability comparisons at the unchanged 1e-12 tolerance. Preserve
finite diagnostic disagreement as a distinct invalid outcome and finish the
fixed schedule; structural, nonfinite, precision, cache-integrity, execution
or resource failures abort with all eight row and 1,464 call slots retained.
Read terminal labels neutrally using lag-six closure, proper-divisor exclusion
and unique reference matching, then compare them with the explicit requested
bit. No branch is required to fail or succeed.

Stage initial files, nonoverlapping 12-step blocks, trajectory descriptors,
per-call diagnostics, completed row shards and final row/result/manifest files
retain the existing batch format. `group-00.json` through `group-03.json` are
immutable last-written markers binding every closed group artifact for the
existing incremental collector. A failed marker write cannot mark its group
completed. Managed signals are held through final records, then restored.

These are new interventions. No full-trajectory scalar equivalence or
historical continuation is assumed for any group. The earlier 240-hold wrong
finals are contextual evidence, not a substitute for the fresh control groups.

## Accounting and fixed limits

Per process, four groups require 1,440 new batch updates, eight prefix calls
and sixteen comparison calls: **1,464 new model calls**, or 2,880 row updates.
Across both processes, all sixteen row outcomes are retained. Historical
updates are 120 endpoint plus age 36 or 240; each new branch adds 360 updates,
so complete path lengths are 516 or 720 updates. The histories are shared by
the forks, are not recomputed, and must remain separate from new-call counts.
Counterfactual branches are not sixteen independent transmitted payload bits.

Per process: one thread; 8 GiB RSS, 16 GiB VM; 1,800 seconds internal,
2,100 seconds external plus 30 seconds termination grace; 256 MiB raw output
including an 8 MiB finalization reserve; 352 MiB archive, 448 MiB reserved
transfer, and 8 MiB source/input upload. `derive_design()` derives a conservative
working envelope from four copies of the tested complete batch-group format,
4 MiB input metadata, one failed native-shape return and 16 KiB event headroom.
The producer must fit this below the 248 MiB working allowance without assuming
compression; final records use the separate reserve.

The completed 732-call two-phase runs took 478.8218201809941 and
478.80444974899 seconds ([receipt](receipts/public-phase-challenge.json)).
Scaling the larger total by 1,464/732 gives approximately 958 seconds as a
planning comparison. It is not a measured runtime, deadline guarantee
or speedup. Actual cold load, stages, diagnostics, total time, peak RSS and
all admitted/returned calls will be retained. New source, tests, actual-input
preflight, run cards and independent review must precede any launch. Fresh
resource admission and the shared 7 TB transfer accounting remain required.

## Interpretation and stopping

A difference between repair and equal-duration hold at the same age supports
a benefit of the tested nominal action from those states. Early repair failing
while late repair succeeds would refute that proposed early handoff for the
sampled histories. Both ages passing would establish only these fixed writes;
no result grants general early-stop readiness. If continued hold changes the
bit, preserve it rather than imposing the prior wrong outcome. There is no
extra retry, pulse extension, phase scan or outcome-selected alternative.

The tests do not establish a general online verify-and-retry policy, coding
redundancy, fresh-state reliability, language recovery, achieved speedup or an
advantage specific to shrimp geometry. Corrective feedback using a remembered
public target is distinct from error-correcting coding. Generic control and
ordinary digital-memory comparisons remain necessary for added-value claims.

The CLI uses the standard model/reference paths and hashes plus
`--requested-bit`, `--hold-packet`, `--hold-packet-sha256`, `--hold-states`,
`--source-run`, `--source-audit` and a fresh `--output-dir`. `--preflight-only`
checks saved evidence without loading a model or invoking a forward. It saves
its complete result or failure as `input-preflight.json`; it is not a run.
