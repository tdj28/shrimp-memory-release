# Matched-displacement intervention at the common hold

This adapted exploratory experiment tests whether an explicit finite change to
the model state can switch the stored label while the controls remain at H.
It does not identify the cause of the nominal endpoint writer. In particular,
generic reset sufficiency and loss of an endpoint attractor are not mutually
exclusive. No communication interface is involved.

## Literal inputs and interventions

Use the original scalar WRITE0 trajectory, `trial-01-endpoint-final-120.npz`
(833183 bytes, SHA256
`c8e4d37d64765734d3b23b3fb3b38e22337d8af1a003cc7bc235cccc3ea51f9a`),
and its `fixed-trial-inputs.npz` (96827 bytes, SHA256
`0ad3846d11ff65d7616b5fc6d125f6b30c16ea56ed490da3bfcd27f510586b44`).
Both originate in the returned `p6-writes/20261025T222347Z-writes/run`.
The producer and independent checker must verify these file pins, array shapes,
float64 finiteness, the phase-zero identity, and historical control values.
Preserve and bind the original metadata, trial plan, complete audit and final
manifest as provenance. Historical endpoint preparation is a prior cost, not
part of a new end-to-end write replication.

Let h1 be `references[1,0]`, h0 be `references[0,0]`, and x60 be
`states[60]` in that literal endpoint trajectory. Require its `states[0]` to
equal h1 bitwise. Set d=x60-h1 and r=norm(d). Generate one PCG64 standard-normal
896-vector using seed 2026102902. Twice subtract its projection parallel to d,
then normalize that direction only to obtain u. Preserve the raw draw, both
projected vectors, u, d, r, all six initial states and their actual displacement
norms. Abort on a zero or nonfinite direction. Do not normalize, clip or project
a model state, select another direction, or vary the radius after outcomes.

Execute three groups in this order, with two independent state lanes each:

| Group | Lane 0 | Lane 1 |
|---|---|---|
| source-and-replay | h1 | x60 |
| opposite-and-positive | h1-d | h1+r*u |
| negative-and-target | h1-r*u | h0 |

H is fixed at alpha=0.07293697809562831 and
log10_temperature=-0.15395151820007266. The pinned Qwen revision, prefix,
steering vector and native64 normalization/rotary patches are unchanged.
Each group prepares the prefix once, compares a cached batch and full-prefix
batch at its initial state, then checks both lanes with separate full-prefix
scalar calls. Each comparison must agree in coordinates and probabilities
to Linf <= 1e-10. Comparisons do not advance the state. Carry the literal returned
state through exactly 240 cached two-lane H updates. There is no within-group
reset or outcome-dependent extension.

## Calls, evidence and qualification

Each group has exactly 245 attempted model invocations: one prefix, two batch
comparisons, two scalar comparisons and 240 batch updates. Total 735; the 720
two-lane updates represent 1440 state-lane updates. Count failed invocations as
attempts. Record per-call input, input-after, output, full probability arrays,
tokens, precision observations, elapsed time and cache hashes; retain the
actual prefix and expanded-prefix caches for independent hash reconstruction.
Archive complete state trajectories and every partial return on failure.
Never overwrite an immutable completed group. The return inventory must bind
every artifact, including unrun groups after an execution failure.

Readout uses only saved arrays. For each lane retain the last 13 states and
compare the final six with the preceding six, as in the existing H gate:
trajectory[235:241] minus trajectory[229:235]. Keep all six phase lag-six
residuals; their maximum must be <= 1e-8. The extra state 228 is archived, not an
additional seventh gate. The final six states must
have maximum phase gaps > 1e-4 for shifts 1, 2, 3 and a unique cyclic reference
match <= 1e-6. Retain distances for every cyclic shift of both references. These
are finite numerical qualification gates, not certified attractor proofs.
Readout never uses a desired answer. Instrument, cache, precision, nonfinite
or resource failure aborts execution and preserves partial evidence. Finite
unqualified controls or interventions do not skip the remaining fixed groups.

The three controls are h1->H1, x60->H0 and h0->H0. If any is unqualified or
wrong, report control-unqualified. Otherwise an H0 outcome for h1-d or either
orthogonal intervention establishes generic reset sufficiency for that literal
intervention. If all three interventions return H1, report no tested reset;
unresolved intervention rows remain unresolved, never inferred negative.
Report each row separately regardless of the panel classification.

The x60 control is a replay of a known successful handoff. The radius was
chosen from that known success. One direction, radius and phase cannot estimate
basin volume, general reliability, a switching threshold or shrimp specificity.
The artificial displacements assume direct access to the continuous state;
they need not be realizable by a token-probability mixture or a text interface.
No projection into such a realizable set is performed or implied.
No outcome establishes continuous branch identity or hysteresis-caused writing.

## Execution envelope and prelaunch gates

Use one CPU model process, batch width 2, with OMP, MKL, OpenBLAS and NumExpr
thread limits set to 1 before Python starts. Limit model invocations to 735,
internal wall time to 2400 seconds, the numerical wrapper to 2460 seconds plus
30 seconds termination grace, the outer job to 2640 seconds and the return worker
to 4800 seconds. Peak RSS cap 8 GiB, virtual memory 16 GiB. Fresh admission requires
at least 24 GiB available RAM, 16 GiB free disk, load average <= 4, no conflicting
task-owned model/collector and an idle human-controlled trial.

Provisional evidence caps are 2304 MiB raw including 32 MiB finalization reserve,
2560 MiB package,3072 MiB archive,8 MiB upload and 8 GiB reserved transfer. Before
freezing for execution, measure the actual maximum 735-call serialization and
complete return, including all scalar and batch probability arrays and the
fallback reserve; amend these caps prospectively if that fixture does not fit.
The monthly human maximum remains 7 TB. Use the shared ledger and reserve a
complete fallback before spending incremental transfer. Do not remove remote
evidence before verified return.

The producer, independent checker, actual-file integration tests, finite
failure tests and private return/termination helpers must pass before launch.
Commit and push the exact sources and run card before outcomes. Use a fresh
isolated checkout and a verified ready local collector. No automatic retry or
numerical extension is permitted. Record any preparation defect and correction;
the fixture envelope supplements, and does not replace, runtime caps.
