# Exact-parameter WRITE0 fold and matched pulses

This is the next case authorized by the human after the hysteresis consult.
It is adapted exploratory numerical research. Existing results already support
fold-informed hysteretic memory. This case closes the interpolation gap for the
temperature-directed fold at the writer's exact alpha and compares local pulses.
It does not test a language channel or restart automatic continuation.

## Map and literal inputs

Use the pinned Qwen2.5-0.5B-Instruct revision, 37-token prefix, layer-12
philosophy steering, one feedback slot, effective native float64 and CPU math
attention from `validate_qwen_map.py`. Keep all numerical libraries at one thread.
The physical alpha is 0.07293697809562831. H has log10 temperature
-0.15395151820007266; E0 is -0.1535; P is -0.15361287955001818 and
N is -0.1535987696062659. Temperature affects feedback softmax, not the prefix.
A changed alpha rebuilds the prefix. The immutable cache is reused only at
identical alpha.

`prepare_public_fold_inputs.py` hash-checks parent curve rows 21/22, both
original P histories and the H cycles. It copies literal states into a compact
packet without running a model. P phases are terminal 480, then 475 through 479.
Replay each parent representative at its own controls before aligning its six
phases to the P1 cycle and interpolating. A phase roll acts on cycle states,
never on embedding coordinates. Interpolation supplies a seed, not a result.

The fixed scale r is exactly the L2 distance between the two saved P terminal
states. This is a numerical conditioning choice; it is not a basin width or the
minimum over phase alignments. Use y=(x/r,(logT-H)/(E0-H)) throughout correction.
Do not rephase states, recenter this scaling or renormalize the null vector
inside a corrector.

## Geometry and acceptance

First replay both P anchors and the parent folds in the current CPU map. The
budgeted feasibility stage must finish within 1,800 seconds including loading.
Compare mixed state/temperature forward AD with a five-step centered-difference
ladder, repeated and full-prefix primals, and separate state/temperature actions
under a mixed linearity check. Require two successive approximately quadratic
halvings, best relative error at most 1e-6, cache/primal maximum error at most
1e-10, bitwise replay and action linearity at most 1e-10. This stage records
actual time, calls, memory and output volume. Stop on failure.

Let Phi=F^6 and A=D_x Phi. Solve the full extended system
R/r=0, (A-I)v=0, c^T v-1=0, at fixed physical alpha. The fixed covector c is
the normalized real critical mode of the replayed row-21 cycle. Obtain it with
eight Arnoldi directions and freshly evaluate the leading mode. Do not infer
full spectral coverage from this finite-subspace diagnostic.

The extended Newton action uses exact first-order state/temperature AD and a
centered difference of A(y)v along the state/parameter direction. This avoids a
dense 896-direction monodromy and higher-order activation graphs. Each augmented
action uses at most four six-step JVPs, all charged individually. Solve once
with scaled difference step 1e-3 and again from the same original interpolated
seed with 5e-4. Before each solve, compare its action with a smaller-step action,
an independent centered difference of the complete residual, and a mixed
linearity check; each relative discrepancy must be at most 1e-4.

Each solve permits six Newton iterations, one ten-vector GMRES cycle per
iteration, at most twelve linear actions including fresh full-space validation,
and damping 1, 1/2, 1/4. GMRES must report success and the fresh linear residual
must satisfy max(1e-14,1e-3 times right-hand-side norm). Acceptance requires
the full scaled residual at most 1e-8, physical orbit L2 residual at most 1e-10,
null residual at most 1e-8 and normalization error at most 1e-10. The independent
final gate additionally checks all six phase closures and proper divisors.
A tiny orbit residual alone never qualifies a fold.

Trace from literal P1 with pseudo-arclength correction until the first qualified
positive-to-negative temperature tangent. The initial step is one fifth of the
scaled distance from P1 to the corrected fold; the only rejected-step alternatives
are half and quarter of that step. Bound the main trace by twelve accepted,
sixteen attempted and four rejected steps. Every point must satisfy the orbit
and arc equations, a fresh tangent equation, a correction-to-step ratio no
greater than one, and consecutive tangent cosine at least 0.5.

Refine the turn from the last pre-turn point with half the accepted crossing
step, at most four further accepted, eight attempted and four rejected points.
Use the saved endpoints and tangents for a local cubic-Hermite maximum estimate.
The two continuation estimates and both extended solutions must agree within
1e-6 in log10 temperature. Cubic-Hermite state estimates at the two maxima
must also agree with each other and the fixed-representative fold state within
1e-6 in physical L2 norm; equal temperatures alone cannot link two folds.
This is numerical refinement spread, not a rigorous
enclosure. The exact-alpha phrase refers to the fixed parameter; no result is
claimed to have an exact real-valued fold location.

Fresh target-mode checks require the near-+1 mode and the next two largest
observed modes to have normalized eigenpair residuals at most 1e-6; require the
critical multiplier within 1e-6 of +1, other checked moduli below 1-1e-6 and
observed separation from other modes of at least 0.01. Preserve ambiguous modes
as unresolved. No generic saddle-node normal-form or global-stability claim is
made without additional nondegeneracy/coverage evidence.

Fresh eight-direction mode diagnostics at both refined turn-bracket endpoints
must show a residual-qualified real critical mode below 1-1e-5 on the incoming
side and above 1+1e-5 on the returning side, with absolute dot product above
0.5 against c and the other checked modes inside the unit circle. The numerical
margin is an operational distinction, not a nonnormal eigenvalue error bound.

Set margin=max(1e-6,5 times the largest refinement spread). Require P+margin
below the located fold and N-margin above it. Only then define B=(P+fold)/2.
Correct native B roots from both P histories and the N survivor from the B0
root; save and qualify their primitive-six cycles. B0/B1 must remain distinct
under every cyclic alignment by more than 1e-4. These controls, the branch-link
gate and full raw artifact checks must pass before pulse admission.

## Conditional matched pulses

After accepted geometry, hold the source and duration fixed across three arms:
P960→P240 (sham), B960→P240 (subfold), and N960→P240 (postfold). Run the two
original P histories as separate lanes at each of the six literal phases.
Classify with the original native P cycles, never H bit labels. Record every
state and retain unresolved outputs. Phase zero is a development phase;
phases one through five are withheld from tuning this case, not independent
random samples or historically untouched data.

Include one phase-zero pair initialized at the corrected native B roots,
B960→P240, and one original H phase-zero pair with the nominal E0(120)→H240
write as a positive control. These are separate controls, not payload capacity.
The fixed main matrix costs 21,708 ordinary model invocations with the existing
compact engine's cache checks. The two controls bring the total to 23,280.
If geometry does not qualify, do not relabel N postfold or run this panel.

The conditional panel is partitioned into two one-thread workers. The first
uses phases zero through two and the native-B control (12,060 ordinary calls);
the second uses phases three through five and the nominal-H control (11,220).
Each retains ten closed groups, twenty stages and twenty final lane readouts.
The second stage carries the actual state produced by the first. No decoded
label selects or alters a later control. P-history indices remain separate
from the H control's bit labels in both data and independent audit.

Pulse admission consumes the independent geometry audit's byte-bound result,
native controls, original input packet and full process/transfer closure.
Each worker is limited to eight GiB RSS, twelve GiB virtual memory, 512 MiB raw
evidence, one GiB return package and a four GiB transfer reservation. The
twenty-hour case deadline is elapsed wall time from the geometry job's original
start, shared by both workers; it is not renewed at pulse launch. Retrieval has
its own reserved interval. An optional one-time case controller may perform
this conditional handoff, then audit both returned workers. It neither retries
failed work nor starts another research case.

## Resource and evidence contract

Whole-case limits are 32,000 ordinary calls, 1,200 six-step JVPs, 200 augmented
actions, twelve hours geometry and twenty hours total numerical time. Reserve
8,720 ordinary calls for geometry and 23,280 for the conditional panel. Calls
spent on failed corrections, all nested derivatives, prefix rebuilding,
refinements and fresh checks debit these same limits. Total model invocations
are reported separately from ordinary calls and tangent lanes. These ceilings
bound a search; they do not guarantee its convergence.

Run geometry as one one-thread model process with a 28,000,000,000-byte aggregate
RSS ceiling, measured free memory of at least 30,000,000,000 bytes, two GiB raw
evidence cap, three GiB package/archive cap and twelve GiB transfer reservation.
Initial timing is part of the geometry allowance. Preserve compact state and
tangent arrays, controls, top-five probability diagnostics, full solver
residuals and every failed/partial return. Atomic closed checkpoints support
incremental hash-checked return to `/data2/shrimp-memory-data`; keep a full
failure-return reserve. No new rental, API call or automatic successor is part
of this case.

The planned figure plots continued cycle state versus temperature, the located
fold and P/B/N controls, alongside native-label pulse outcomes by phase. Its
source is the saved cycles, tangents, mode residuals and every pulse row. Do not
draw an absent branch or invent a basin geometry. A positive result supports
the local branch-loss account of these matched pulses; it does not alone prove
that fold loss is necessary for every possible reset or explain every transient
of the nominal H-to-E0 jump.

## Disclosed equation-order amendment

The first exact-parameter attempt stopped before a Newton trial: its first
ten-direction GMRES cycle failed the unchanged fresh linear residual gate.
The complete return and independent audit are recorded in
`receipts/public-e01-first-geometry.json`. This is numerical nonconvergence,
not evidence of a missing fold. The failed initial audit and its repair for a
host-dependent, unsaved norm reduction remain in the same evidence record.

One adapted repetition changes only the equation order at the extended square
solver boundary. The unknown order is `(x/r, temperature, v)`. Permute residual
and action rows from `(orbit, null, normalization)` to
`(orbit, normalization, null)`, with the corresponding residual-block indices.
This orthogonal permutation preserves roots, singular values and full Euclidean
residual norms while aligning the dominant minus-identity vector block.
Saved actions and a separately recorded exact saddle-node control motivated
this choice. They do not establish convergence of the amended Qwen solve.

Keep the original fold residual, derivative and physical gates outside the
solver boundary. Retain the original seed, fixed covector, two difference
steps, ten-direction linear allowance, damping choices and acceptance
tolerances. Record the actual direction-norm scalar used to construct each
finite-difference pair so that its two signed inputs can be reconstructed
exactly across hosts. Repeat the bounded feasibility checks and charge every
repeated call. An unresolved amended solve stops; it does not silently enlarge
the Krylov space or change tolerances.

The amended attempt must consume a byte-bound context derived from the accepted
first audit, result and original start record. Record fresh-attempt and
cumulative counts separately. Subtract previous geometry calls, raw bytes,
package/archive bytes and charged transfers from their original envelopes.
Retain the original absolute geometry and case deadlines, rather than starting
new twelve- and twenty-hour windows. The new wrapper's actual start time is
also preserved. A separate source checkout and completed return record retain
the distinction between attempts. Conditional pulses require qualification of
the amended geometry and enough remaining time under these same case limits.
