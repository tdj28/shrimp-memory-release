# A fixed public toggle feasibility pilot

This exploratory pilot asks whether one common control pulse can exchange the
two qualified numerical states: original H0 must finish as H1, while original
H1 must finish as H0. The same pulse acts on both lanes. A reset that sends both
lanes to one value is not a toggle. No observed readout selects a control,
duration, inclusion rule or retry.

Use the original qualified references in their saved order, at literal phase
zero: `references[0,0]` and `references[1,0]`. Each group starts with this pair.
Within a group, the hold consumes the actual terminal pulse state. There are
eleven initializations, ten between-group resets and eleven carried boundaries.

## Fixed controls and ledger

Let H be the established hold, E0 the WRITE0 endpoint, and E1 the WRITE1
endpoint. Construct four diagonal corners with the exact binary64 operations

```
alpha = H.alpha + sign_alpha * (E1.alpha - H.alpha)
log10_temperature = H.log10_temperature + sign_temperature * (E0.log10_temperature - H.log10_temperature)
```

The signs are `(-,-), (-,+), (+,-), (+,+)`, in that order. The geometry explores
combined perturbations along the two previously used control directions and
their reflections. It is not derived from a predicted basin exchange, an
identified bifurcation or an expectation that a toggle exists.

Run the three baselines first, followed by each corner at 12 then 36 updates.
All groups finish with 240 updates at H. The plan saves the actual controls,
their hexadecimal representations and binary64 bits; names alone are not
evidence of the coordinates used.

| Groups | Pulse | Final hold | Calls per group |
|---|---|---|---:|
| 0 | H120 | H240 | 366 |
| 1 | E0(120) | H240 | 366 |
| 2 | E1(120) | H240 | 366 |
| 3,5,7,9 | Corresponding corner, 12 updates | H240 | 258 |
| 4,6,8,10 | Corresponding corner, 36 updates | H240 | 282 |

The exact maximum is 3,258 attempted calls: 3,192 batch updates, 22 prefix
preparations and 44 cached/full batch comparison calls. This is 6,384 lane
updates, 22 stages and 22 terminal rows. It uses the previously validated batch
size two, with no extra scalar checks or derivative calls. There is no dynamic
phase, control or duration CLI.

## Neutral qualification and reporting

Reuse the established neutral decoder. At each final hold, compare states
235–240 with states 229–234. Every one of the six L2 residuals must be at most
1e-8. Exclude proper divisors 1, 2 and 3 using their maximum final-cycle phase
gap greater than 1e-4. Require exactly one original reference within 1e-6 under
the best cyclic alignment. Preserve all residuals, divisor gaps and all six
alignments for both references.

Classify the ordered pair only after both neutral readouts:

| Qualified pair in original H0/H1 lane order | Mapping |
|---|---|
| [1,0] | toggle |
| [0,1] | identity |
| [0,0] | reset0 |
| [1,1] | reset1 |
| At least one unqualified finite readout | unresolved |

An incomplete or unrun pair is unavailable, not unresolved. The three expected
baseline pairs are [0,1], [0,0] and [1,1]. Their qualification affects the
panel interpretation, never the recorded probe mapping or execution order.
With qualified baselines, a qualified probe toggle yields `toggle_observed`;
without one, any unresolved probe yields `toggle_unresolved`, otherwise
`no_toggle_observed`. A baseline mismatch yields `baseline_unqualified`.
All finite mappings continue all eleven groups. A positive result identifies
only a candidate pulse at these two initial phases; it does not establish
repeated toggling, other-phase operation or robustness.

## Reused instrument and evidence contract

Reuse `run_public_mechanism` and its `run_public_control_continuation.execute_stage`,
`run_public_batch_writes` evaluator, block archive and reconstruction routines.
The new wrapper adds no model algorithm. The model remains the pinned CPU
Qwen2.5-0.5B-Instruct map with effective float64 normalization and rotary
operations, math attention and one thread. All four numerical thread limits
must equal one before startup; record the 49 normalization patches, one rotary
patch and Torch thread settings. Per-call fresh precision observations,
unchanged inputs, scalar/expanded cache integrity and cache lengths retain
their existing gates. Cached/full output and probability gaps must be at most
1e-12, with identical top tokens. A finite comparison mismatch is saved and
stops before any update in that stage.

The compact format intentionally matches the earlier mechanism engine:
every admitted state, top token and update timing is saved in nonoverlapping
12-update blocks, with a separate literal initial state. Every successful
update records input/output/probability hashes and precision/cache diagnostics.
Full probability arrays are saved for all 44 comparisons and any returned
failure, not for every successful update. A saved-array audit can verify the
stored arrays and continuity; it cannot reconstruct omitted probabilities or
recompute every model forward from their hashes alone.

Files include `design.json`, planned groups/stages/rows/calls, `source.json`,
qualified inputs, `reference-cycles.npz`, `reference-starts.npy`, every stage's
initial/cache/block/trajectory files, diagnostic/failed call NPZs, row JSONs,
`events.jsonl`, `result.json` and `manifest.json`. Atomic immutable
`group-00.json` through `group-10.json` bind disjoint raw artifacts and immutable
stage/call/row summaries. Completed, partial and unrun groups all close. Mutable
events and final result belong to the final return, not an earlier shard.

Resource failures are `incomplete`; numerical/cache/precision defects are
`instrument_failed`; other execution and persistence errors are
`execution_failed`. Original exceptions and inherited stage/call statuses
remain visible. The run-local guard converts inherited raw-cap RuntimeErrors
to resource failures without changing any older module. No failure is rerun
or converted to a scientific null.

## Bounds, CLI and independent audit

Bound execution at 3,600 seconds, 8 GiB RSS and 16 GiB virtual memory. The raw
cap is 512 MiB, including a 32 MiB finalization reserve. The native uncompressed
payload calculation, plus explicit container/metadata/failure allowances, is
336,247,616 bytes. This is a design estimate, not measured model output or
latency. The orchestration owner must measure the complete source/input upload,
package/archive/return envelope and fresh host/transfer headroom before launch.
The monthly transfer ceiling remains unchanged.

The path CLI is `--cache-dir --vectors --references --reference-manifest
--source-hold-results --recovery-results --output-dir`, with
`--references-sha256 --reference-manifest-sha256 --recovery-results-sha256` and
optional `--preflight-only`. The three hashes must be the original fixed pins
in the runner. Source-hold and recovery results retain their required colocated
metadata and raw source trajectories. Preflight loads no model and closes
all planned groups as unrun.

Before interpretation, independently rehash every returned file, reconstruct
both lanes and every carried boundary, verify fixed controls and call counts,
inspect every comparison and failure, and recompute all neutral readouts and
pair mappings. Preserve unavailable rows and baseline failures. The checker
must verify closed return and source/input provenance. No language transport,
concealment interface, agent sandbox, causal-hysteresis claim, new payload
capacity or general reliability claim is part of this numerical pilot.
