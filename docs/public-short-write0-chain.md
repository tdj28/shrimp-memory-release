# Fresh shortened WRITE0 followed by a carried WRITE1

This adapted exploratory test asks whether a fresh 36-update WRITE0 works from
both phase-zero reference states, and whether its actual returned states can
then support the canonical 120-update WRITE1. The preceding historical-state
experiment returned H1 from x12 and H0 from x36, with both controls qualified.
That observation motivated this test; it did not execute a shorter writer or
establish that its result could be reused. The conditional proposal was
finalized after those outcomes were disclosed. This is public numerical memory
research, with no language transport or concealed coordination component.

## Inputs and literal state carry

Retain the six original input pins used in
[the historical handoff protocol](public-nominal-handoffs.md): the H1 endpoint
trajectory, fixed references, original metadata, trial plan, complete audit and
final manifest from `p6-writes/20261025T222347Z-writes`. Add the unchanged H0
endpoint trajectory `run/trial-00-endpoint-final-120.npz` (833281 bytes, SHA256
`dce4af2b24d392d08589a7e747e832f1a690ca11172d544e7f7d5878e5045f67`).
All seven byte/hash pins must be checked before using the inputs. Require
finite float64 historical states of shape (121,896), reference cycles of shape
(2,6,896), and literal equality of each old initial state to its corresponding
reference phase zero. Bind the accepted nominal-handoff result and independent
audit as decision provenance in the preparation receipt.

Initialize the two lanes as `[references[0,0], references[1,0]]`. Keep the
original phase. Every subsequent stage receives the exact terminal pair from
the preceding stage. No renormalization, phase rotation, recentering, reference
replacement or state selection is allowed at a boundary.

| Group | Control | Updates | Attempted invocations | Readout |
|---|---|---:|---:|---|
| 00 | E0 | 36 | 41 | Complete fresh WRITE0 trajectory |
| 01 | H | 240 | 245 | Two neutral labels, then compare with requested 0 |
| 02 | E1 | 120 | 125 | WRITE1 starting from the literal group 01 terminal pair |
| 03 | H | 240 | 245 | Two neutral labels, then compare with requested 1 |
| Total | | 636 | 656 | Four retained readouts |

The two lanes are starting-state controls, not twice the payload capacity.
Each group charges one prefix call, one cached batch comparison, one full-prefix
batch comparison and two full-prefix scalar comparisons before its prescribed
updates. Comparison outputs do not advance the state. Thus the complete ledger
contains 4 prefix, 8 batch-comparison, 8 scalar-comparison and 636 batch-update
invocations, representing 1272 state-lane updates. Failed attempts also count.

Use the unchanged pinned Qwen model, prefix, steering vector and token position.
In (alpha, log10_temperature) order, the literal binary64 controls are
H=(0.07293697809562831, -0.15395151820007266),
E0=(0.07293697809562831, -0.1535), and
E1=(0.07299785589927693, -0.15395151820007266).
Keep the effective-float64 patches, with 49 normalization patches and one
rotary patch. Intra-op, inter-op, OMP, MKL, OpenBLAS and NumExpr threads are all
one. The runner exposes this named schedule only; there is no arbitrary
control, duration or payload interface.

## Instrument checks and readout

At every stage start, require cached/full-prefix/both-scalar maximum coordinate
and probability discrepancies to be at most 1e-10. Save the full probabilities,
tokens, input/input-after/output states, precision observations, original and
expanded caches, cache hashes and per-call timings. Save complete trajectories
and immutable group closures, including partial attempts and late failures.

Compare all 37 fresh group 00 states in both lanes with the corresponding
original scalar trajectories, without substituting either trajectory for the
other. Retain maximum coordinate and L2 differences, exact byte-equality flags
and every token comparison. The fixed replay allowance is a maximum coordinate
difference of 1e-10. This historical replay diagnostic is reported separately
from neutral qualification of the fresh writes. A discrepancy does not select
a different duration, replace a state or cause a numerical rerun.

At each H readout, retain the final 13 states and all six lag-six residuals
between states[235:241] and states[229:235]. Require their maximum L2 residual
to be at most 1e-8. Exclude proper periods 1, 2 and 3 using maximum phase gaps
greater than 1e-4. Require a unique match to one reference cycle within maximum
phase distance 1e-6 after cyclic alignment. Preserve all six alignment
distances to both reference cycles. Decode without access to the desired bit;
only afterward compare with the requested value. A nearer reference alone is
not a qualified readout.

Both lanes and all four stages execute after finite wrong or unresolved
scientific outcomes. Only an instrument, execution or resource failure stops
the run; preserve it and mark every unrun stage explicitly. Report all four
rows, each gate and the two requested-write comparisons. Qualification of all
four requested labels supports this one short-write-and-reuse chain. Wrong
qualified labels and unresolved labels remain distinct outcomes. No automatic
longer pulse, longer hold, alternate phase, binary search or retry is allowed.
The fixed finite outcome names are `chain_qualified` for labels [0,0,1,1],
`chain_unresolved` when any label is unresolved, and `chain_wrong` otherwise.
An instrument or execution failure is reported separately from these outcomes.

## Cost and evidence limits

The first write alone prescribes 286 invocations; the full chain prescribes
656. An otherwise identical four-stage schedule with a 120-update first pulse
would prescribe 740. The difference of 84 calls is arithmetic, not a measured
wall-time speedup or a contemporaneous comparator. Earlier full-length tests
provide historical context only. Record actual prefix, diagnostic, endpoint,
hold, total elapsed and resource costs, including wrong and unresolved writes.

Success would establish a finite numerical observation at these two initial
phases, these controls and these stopping rules. It would not establish a
minimum pulse duration, monotonic switching, arbitrary phases or messages,
asymptotic attraction, general reliability, causal hysteresis, or a language
interface. The previous artificial fixed-H resets remain relevant competing
evidence when discussing the writer's mechanism.

## Execution and verified return

Run one CPU model process with batch width two and one numeric thread in an
isolated committed checkout. Bounds are 2400 seconds internal, 2460 seconds
external with 30 seconds termination grace, 2640 seconds outer job and 4800
seconds for the return worker; 8 GiB peak RSS and 16 GiB virtual memory.

Candidate evidence ceilings are 2304 MiB numerical raw including a 32 MiB
finalization reserve, 2560 MiB package, 3072 MiB archive, 8 MiB upload and an
8 GiB shared transfer reservation including fallback. The 1300 probability
vectors alone require 1,580,134,400 bytes. Measure the complete four-stage,
656-call serialization and return contract before freeze; a smaller call count
than a previous run does not prove that its cache layout fits. Exercise early
failure, late failure, incremental return, fallback, full independent audit,
termination and hash verification using the actual result filenames.

Fresh admission requires at least 24 GiB available memory, 16 GiB free disk,
load average at most four, no conflicting task-owned model or collector,
and an idle human-controlled trial. Use a verified ready collector before
upload, exact process ownership, a bounded shared reservation and an exact
source/input upload proof. The monthly transfer maximum remains 7 TB.
Return all evidence to `/data2/shrimp-memory-data` and verify it before any
remote deletion. Commit and push the tested producer, independent checker,
protocol and run card before numerical outcomes. This run uses the existing
standing server and no external paid call or new rental.
