# A shortened WRITE0 from the next cycle phase

This adapted exploratory experiment asks whether the fresh 36-update WRITE0
and subsequent carried 120-update WRITE1 also work from phase one of both
reference cycles. The preceding phase-zero run qualified all four requested
labels. A conditional proposal naming phase one was saved before that result;
its success branch is now selected. This extends one sampled initial phase,
not a claim of general phase independence. The experiment is public numerical
memory research.

## Fixed input and schedule

Keep all seven historical input pins from
[the phase-zero protocol](public-short-write0-chain.md) and verify their
identities, effective float64 arrays and historical provenance. Add the saved
prospective pair `prospective-phase1-pair.npz`, 15170 bytes, SHA256
`7e146c00fced5d85310e1d3dc1d6e332957c9d38921f5ffc0306fd302a095bc4`.
It was prepared with the pre-outcome proposal. Require its initial pair to
equal `references[:,1]` byte for byte; the array is float64 with shape (2,896).
The original scalar histories begin at `references[:,0]`. Check that distinct
identity too; never treat them as runs from the new initial conditions.

Initialize the two lanes as `[references[0,1], references[1,1]]`. Keep those
literal arrays, with no rotation, renormalization, settling, state search or
replacement. The schedule is fixed:

| Group | Control | Updates | Attempted calls | Readout |
|---|---|---:|---:|---|
| 00 | E0 | 36 | 41 | Fresh complete WRITE0 trajectory |
| 01 | H | 240 | 245 | Both neutral labels; then compare with zero |
| 02 | E1 | 120 | 125 | WRITE1 from the exact preceding terminal pair |
| 03 | H | 240 | 245 | Both neutral labels; then compare with one |
| Total | | 636 | 656 | All four rows retained |

Every boundary carries the exact returned state bytes. The two lanes are
starting-state controls, not two independently transmitted payload bits.
Each group charges one prefix call, one cached batch comparison, one full
batch comparison and two full scalar comparisons before the updates.
Comparison outputs do not advance the state. The complete ledger therefore
has four prefix, eight batch-comparison, eight scalar-comparison and 636
batch-update calls, representing 1272 state-lane updates. Attempted failed
calls count toward the same bound.

Use the unchanged pinned Qwen model, prefix, steering vector, token position,
effective float64 normalization and rotary operations. In
(alpha, log10_temperature) order, the binary64 controls are
H=(0.07293697809562831, -0.15395151820007266),
E0=(0.07293697809562831, -0.1535), and
E1=(0.07299785589927693, -0.15395151820007266).
Require 49 normalization patches and one rotary patch. All numerical threads
are one. There is one model process and two lanes; no configurable input phase,
duration, control or payload interface is introduced.

## Readout and diagnostics

At every stage start, require cached/full-prefix/both-scalar maximum coordinate
and probability discrepancies at most 1e-10. Preserve every full probability
vector, token, input/input-after/output state, precision observation, original
and expanded cache, cache hash and call timing. Preserve full trajectories,
immutable closed groups and partial or failed attempts.

For each H readout, retain the last 13 states and all six lag-six residuals
between states[235:241] and states[229:235]. Require maximum L2 residual at most
1e-8. Exclude proper periods 1, 2 and 3 by maximum phase gaps greater than
1e-4. Require exactly one reference cycle to match within maximum phase
distance 1e-6 after alignment. Retain all six shift distances to both
references. Desired labels are unavailable to this neutral decoder; compare
with the requested value only afterward.

Retain a descriptive comparison of all 37 new E0 states and 36 token outputs
per lane against the old phase-zero scalar histories. This is an explicitly
**cross-phase comparison between different initial conditions**, with all
coordinate and L2 gaps, byte-equality and token-equality flags. It has no
replay tolerance, pass/fail threshold or scientific gate. No state is borrowed
from the historical histories, and no numerical repeat is triggered by their
differences. The old phase-zero result is historical context only; there is no
fresh phase-zero comparison arm in this experiment.

Both lanes and all groups continue after finite wrong or unresolved readouts.
Only an instrument, execution or resource failure stops the run; every unrun
stage is explicit. Preserve all four rows, all gates and all requested-write
comparisons. The finite outcomes are `chain_qualified` for [0,0,1,1],
`chain_unresolved` if any row is unresolved, and `chain_wrong` otherwise.
Instrument and execution failure remain separate categories. No longer pulse,
extended hold, alternate phase, retry or retrospective threshold change is
allowed.

## Bounds and interpretation

The first write costs 286 calls and the full chain 656. An otherwise identical
120-update first pulse would prescribe 740 calls. The 84-call difference is
arithmetic, not measured wall-time acceleration. Record actual total elapsed,
forward time, memory, output and return costs, including failed outcomes.
The 1300 full probability vectors require 1,580,134,400 bytes. A new phase-zero
comparison arm would double the run; it is outside the present bounds.

Run in a separate committed source checkout on the existing standing server.
Keep bounds of 2400 seconds internal, 2460 seconds external plus 30 seconds
grace, 2640 seconds outer and 4800 seconds for return. Limits are 8 GiB RSS,
16 GiB virtual memory, 2304 MiB numerical raw including a 32 MiB finalization
reserve, 2560 MiB package, 3072 MiB archive, 8 MiB upload and an 8 GiB shared
transfer reservation including fallback. Measure the actual complete
656-call serialization, incremental and fallback return, and early and late
failure paths before launch. No model is used in these contract tests.

Fresh admission requires at least 24 GiB available RAM, 16 GiB free disk,
load average at most four, no conflicting task-owned model or collector,
an idle human trial, a current shared transfer ledger and a ready verified
return worker. Return all evidence to `/data2/shrimp-memory-data`; verify all
files and hashes before removing any remote evidence. The monthly maximum
remains 7 TB. Commit and push the tested producer, independent checker,
protocol and run card before generating numerical outcomes. No new rental or
external paid call is needed.

Success adds this one literal reference phase to the phase-zero observation.
It does not establish all-phase or off-cycle reliability, a minimum write
duration, asymptotic attraction, causal hysteresis, or language communication.
A failure would identify a limitation at the tested starts and stopping rule.
Diagnose saved arrays first; do not retry until a preferred outcome appears.
