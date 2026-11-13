# Public controller challenge at two literal reference phases

This is preparation for one adapted exploratory pair of numerical runs. It
tests whether an observed control sensitivity persists when a public write
starts from a neighboring saved phase of the same reference cycle. It does
not optimize a hidden channel or test a language interface. No numerical
execution is authorized by this protocol alone: tested source, independent
input qualification, a committed run card, current resource and transfer
admission, and verified return workers precede any launch.

## Fixed question and inputs

The WRITE0 process starts from the saved td reference phases 0 and 1. The
WRITE1 process starts from the saved ad reference phases 0 and 1. These are
literal float64 rows of the hash-bound reference array, not states obtained
by an extra model update or settling period. Batch row 0 is phase 0; batch
row 1 is phase 1. Both rows receive the same explicit public command.

Each process has exactly two groups, in this order:

| Group | WRITE0 endpoint change | WRITE1 endpoint change |
|---|---|---|
| `nominal` | none | none |
| `full-negative-offset` | log10 temperature minus 0.0001 | alpha minus 0.00001 |

These offsets are the original full failed offsets in the preceding
axis-stencil experiment. They are fixed independently of the intervening
fractional refinement outcomes. The adaptation is disclosed: the source
cycle is the one that failed the original perturbed switch, and only its
first two saved phases are challenged. Offset failure is not required.

Each group reinitializes the same two literal states, runs 120 endpoint
updates, and carries its actual endpoint state through 240 updates at the
unchanged common hold coordinates. There is one reset between groups and
no reset within a group. The canonical parsed public action remains the
original 120+240 action; realized endpoint offsets are separately recorded.
No requested label enters the neutral numerical readout.

Both original scalar phase-zero controls are requalified for prerequisite
provenance. Only nominal batch row 0 is then compared with the matching
scalar trajectory: old p6-writes trial 1 for WRITE0 (td phase 0), or trial 2
for WRITE1 (ad phase 0). All 120 endpoint and 240 hold updates, their initial
states and token IDs are compared at coordinate tolerance 1e-10 and zero
token mismatches. Phase 1 has no historical scalar comparator. Perturbed
groups have no scalar equivalence claim. Historical timing is descriptive.

## Precision, gates and outcomes

Use the existing pinned Qwen backend, native float64 normalization and
rotary patches, CPU math attention and one thread. Each of the four stages
has its own prefix construction and cached/full batch diagnostic; their
coordinate and probability tolerance is 1e-12. Save effective precision
observations and cache/input integrity checks. This is the existing finite
verification scope, not a new derivative-convergence or global precision
proof.

For every row, independently evaluate the unchanged neutral terminal gate:
lag-six closure, exclusion of proper divisors, and unique reference-cycle
agreement. Store the qualified label, all gate values and the separate
requested-bit comparison. Finite wrong or unresolved labels, finite
cached/full disagreements, or finite nominal scalar disagreements retain
their outcomes and continue the fixed schedule. Structural, nonfinite,
effective-precision, cache-mutation, resource or execution failures abort
and retain completed, failed, partial and unstarted slots. A finite
diagnostic disagreement is not the same event as invalid or mutated cache
state. No retry, horizon extension, state substitution or control adjustment
is permitted after an outcome.

## Exact budgets and file contracts

Each process has 4 rows, 4 stages, 720 batched updates, 4 prefix calls and
8 diagnostic calls: **732 forward calls**. Across the pair there are 8 rows
and at most 1464 calls. A process has a 900-second internal deadline and a
1080-second external deadline with 30 seconds for termination and evidence
finalization. It uses one thread, at most 8 GiB RSS and 16 GiB virtual memory.

The raw cap is 160 MiB, including an 8 MiB finalization reserve. The explicit
uncompressed envelope is twice the existing 28,627,392-byte complete-group
envelope, or 57,254,784 bytes, plus one 2,473,984-byte failed-return allowance
and 16,384 bytes of event allowance. This is below the 152 MiB working
allowance without assuming compression. The outer archive cap is 224 MiB,
transfer reservation 320 MiB per process, and source upload cap 8 MiB.
Admission must also count incremental group returns, the final return and
both concurrent processes under the shared monthly ledger.

A historical proportional planning estimate is about 431 seconds per
process: the larger preceding five-group total, 1076.368245236008 seconds,
times 732/1830 is 430.5472980944032 seconds. Both totals (the other is
1075.6707983649976 seconds) are recorded in the bound
[axis-stencil receipt](receipts/public-control-tolerance.json). Cold
initialization and changing sustained host performance are not isolated by
that proportional estimate. It is not a runtime prediction, measured
speedup or matched timing result; the 900/1080-second caps remain fixed.

`design.json`, all four `planned-rows.json` slots, the two groups, four
stages and 732 calls are written before model construction. `source.json`
binds the new runner, this protocol, its tests, imported numerical helpers,
historical receipts and dependency lockfiles. `qualified-inputs.json` binds
the reference/map inputs. `historical-inputs.json` retains the original
pair qualification and explicitly selects one trial. `phase-inputs.json`
binds both literal phase-array hashes, shape/dtype and their combined hash.
`frozen-rows.json` retains each unchanged parsed public action.

Stage IDs are globally 0 through 3. Initial arrays and nonoverlapping
12-update blocks use the existing stage filename contract; a stage
trajectory manifest reconstructs every complete or partial trajectory.
`row-XX-completed.json` is immutable; final `row-XX.json` preserves every
planned slot. Closed `group-00.json` and `group-01.json` markers are written
last with sorted bindings for all of their immutable group artifacts.
Events and mutable global files are excluded from those markers. Existing
collectors can inspect absent group IDs 2 through 4 without treating their
absence as an execution failure. Final return still requires `result.json`,
the full producer manifest, completion and transfer closeout.

The CLI requires `--requested-bit 0|1`, `--cache-dir`, `--vectors`,
`--references`, `--reference-manifest`, `--source-hold-results`,
`--recovery-results`, `--historical-run`, `--output-dir`, and the three
SHA-256 arguments `--references-sha256`, `--reference-manifest-sha256`,
`--recovery-results-sha256`. The fixed profile is
`public-two-phase-challenge`; there is no profile-selection argument.
`--preflight-only` verifies saved inputs and phase hashes, writes an
exclusive `input-preflight.json`, and makes zero model calls or loads.

These two phases and two discrete controls cannot establish all-phase
robustness, a continuous control tolerance, held-out reliability, basin
geometry, or general composability. A phase difference would motivate a
separate explanation; it is not assumed in advance. Every phase outcome
and numerical diagnostic remains in the denominator.
