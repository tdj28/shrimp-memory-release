# Public control continuation from saved endpoint states

This adapted exploratory experiment compares extra time at the perturbed
endpoint with restoration of the nominal endpoint. It is a public numerical
control study. It does not test a language interface, hidden encoding or
error-correcting code.

## Adaptation and selection

The design was selected on 26 October 2026 after all eight outcomes of the
public two-phase challenge were available. At nominal controls, both tested
origins passed for both requested bits. At the full negative offset, all four
rows qualified as the original, wrong bit; none was unresolved or diagnostic
invalid. This selection is disclosed rather than described as a fresh
confirmatory test. The phase challenge used source commit
`739a4fd678e35364236799d3701463a4d7b0009d`.

A preceding design review proposed a continuation of one selected bit. This
preparation extends that proposal to both requested bits under the human's
standing authorization for bounded parallel public-control experiments.
Both bits and both arms are fixed before new numerical outcomes. Neither arm
is required to improve the write or to reproduce an earlier failure.

The input packets were prepared from the full, independently audited phase
runs. Their SHA-256 hashes are fixed in the producer and checked against the
CLI declaration. Each packet binds the complete source audit, source result,
18 consumed source artifacts, literal endpoint array and origin ordering.
The saved endpoint is the terminal state after exactly 120 updates of the
full-offset endpoint stage (source stage 2), before any common-hold update.
It is also byte-identical to the initial state of source stage 3. It is not
the settled wrong attractor observed at the end of that hold.

## Fixed arms and state continuity

The profile is `public-endpoint-continuation`. Each requested-bit process
uses a batch of two states whose origins are the earlier reference phases
0 and 1: td for WRITE0 and ad for WRITE1. These are **origin labels**, not a
claim that the current endpoint states lie on those reference cycles.

| Group ID | Arm | New endpoint updates | New hold updates |
|---|---|---:|---:|
| 0 | `continue-perturbed-endpoint` | 120 at the same full-offset endpoint | 240 |
| 1 | `restore-nominal-endpoint` | 120 at the nominal endpoint | 240 |

Both arms initialize from exactly the same two saved endpoint rows. Arm 1
never inherits arm 0's terminal state. There is one reset between arms,
zero fresh reference-state initializations and no reset between an arm's
endpoint and hold stages. A fresh prefix/cache construction is not a state
reset.

WRITE0's nominal endpoint is alpha 0.07293697809562831 and log10 temperature
-0.1535; its full offset subtracts 0.0001 from log10 temperature. WRITE1's
nominal endpoint is alpha 0.07299785589927693 and log10 temperature
-0.15395151820007266; its full offset subtracts 0.00001 from alpha. The common
hold is unchanged: alpha 0.07293697809562831 and log10 temperature
-0.15395151820007266.

The public commands remain the exact parsed WRITE0/WRITE1 commands. Their
canonical action is retained separately from the realized experimental
continuation schedule. The explicit protocol determines the arm controls;
the outcome does not choose the arm or change its duration. There is no
scalar trajectory equivalence claim for either arm. The existing original
120-endpoint plus 240-hold branch is a historical no-extension control,
not a newly rerun arm.

## Input qualification and evidence contract

Before loading weights, `--preflight-only` performs the actual-file checks:
pinned packet, independent source audit, all selected source hashes, source
stage controls and update count, nonoverlapping 120-update reconstruction,
reference-origin equality, endpoint NPZ dtype/shape/order, trajectory and
row-array hashes, and equality to the saved pre-hold handoff. It does not
load the model or run a model forward. Model/reference arguments are the
existing fixed Qwen inputs. The added arguments are `--endpoint-packet`,
`--endpoint-packet-sha256`, `--endpoint-states`, `--source-run` and
`--source-audit`; there is no scalar `--historical-run` argument.

The run saves `qualified-inputs.json` for the reference qualification and
`continuation-inputs.json` for the complete packet and actual consumed
bindings. All planned group, row, stage and call slots are saved before
backend creation. The fixed inventory per process is two groups, four rows,
four stages and 732 call slots. Rows record `origin_reference_index`,
`origin_reference_label`, `origin_reference_phase`, `source_result_row_id`
(2 or 3), `source_stage_id` (2) and `historical_endpoint_updates` (120).

Raw artifacts retain the established stage initial/block/trajectory format,
per-call observations, cache diagnostics, row shards and immutable group
markers. `group-00.json` and `group-01.json` are written once, last, after
all group artifacts are closed; their sorted inventories support incremental
return. Events, mutable result/row shards and the marker itself do not enter
that immutable inventory. Structural, precision, cache-integrity, execution
and resource failures abort subsequent computation but retain every planned
slot and every saved partial output. Finite wrong, unresolved or diagnostic
failures are preserved and do not select later arms.

## Gates, accounting and limits

Use the pinned native float64 Qwen backend, normalization/rotary patches,
CPU math attention and one thread. Every new stage constructs a prefix and
checks cached versus full-prefix results before fixed updates. The existing
1e-12 coordinate/probability diagnostic tolerance and neutral readout gates
are unchanged. Neutral readout checks lag-six closure, proper-divisor
exclusion and unique reference agreement before the requested-bit comparison.
Correct, qualified wrong, unresolved, diagnostic-invalid and unexecuted rows
remain distinguishable. This is finite numerical evidence, not a global
precision or exact-dynamics proof.

Each process admits at most 720 new batch updates, four prefix calls and
eight comparison calls: 732 new model calls and 1,440 new row updates.
Each branch comprises 120 historical endpoint updates plus 120 new endpoint
and 240 new hold updates, or 480 state updates along its full path. The
historical 120 updates are shared by the fork and are not executed again.
The input packet separately records the original 732-call source run and
its selected 123-call endpoint stage. Do not double count those histories
as new calls or omit them from end-to-end cost discussion.

Per process: 900 seconds internal, 1,080 seconds external plus 30 seconds
termination grace; one thread; 8 GiB RSS and 16 GiB virtual memory; 160 MiB
raw output including an 8 MiB finalization reserve; 224 MiB archive cap;
320 MiB transfer reservation; 8 MiB source upload cap. The two completed
732-call phase runs took about 479 seconds each. That is a descriptive
planning comparison, not a runtime guarantee or speedup claim. A launch
still requires committed tested source, actual-input preflight, a run card,
measured capacity, shared transfer admission and verified return workers.

A difference between arms could support a benefit of the tested endpoint
restoration at matched additional duration. Both succeeding could reflect
extra time, ordinary rewriting or both; both failing would rule out only
these tested extensions. Phase disagreement concerns these two histories.
The experiment establishes no continuous tolerance, all-phase reliability,
generic error correction, online feedback policy, language success or
special advantage of shrimp geometry. No longer hold is tested.
