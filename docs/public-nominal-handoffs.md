# Earlier handoffs from the recorded nominal WRITE0 pulse

This adapted exploratory test asks whether two earlier literal states from
the nominal WRITE0 trajectory already return to H0 after the fixed common-hold
readout. The preceding matched-displacement panel found three artificial
resets at unchanged H. This new test concerns the actual writer trajectory;
it does not expand that intervention search or extend an unsuccessful root
search. No communication interface is involved.

## Fixed inputs and schedule

Use the original returned scalar trajectory
`p6-writes/20261025T222347Z-writes/run/trial-01-endpoint-final-120.npz`
(833183 bytes, SHA256
`c8e4d37d64765734d3b23b3fb3b38e22337d8af1a003cc7bc235cccc3ea51f9a`)
and `fixed-trial-inputs.npz` (96827 bytes, SHA256
`0ad3846d11ff65d7616b5fc6d125f6b30c16ea56ed490da3bfcd27f510586b44`).
Retain the original metadata, trial plan, complete audit and final manifest
with the same byte/hash pins as the preceding state-intervention protocol.
Check finite float64 states of shape (121,896), references of shape (2,6,896),
and bitwise equality of states[0] with references[1,0].

Let h1=references[1,0], x12=states[12], x36=states[36] and x60=states[60].
Keep each as a literal unmodified state, including its original phase.
There is no new endpoint preparation, normalization, phase rotation, clipping
or replacement with a reference state. The historical preparation is a prior
cost; this is not a new end-to-end replication of the original write.

| Group | Lane 0 | Lane 1 |
|---|---|---|
| source-and-replay | h1 | x60 |
| early-handoffs | x12 | x36 |

Both parameters remain at H: alpha=0.07293697809562831 and
log10_temperature=-0.15395151820007266. Use the unchanged pinned Qwen model,
prefix, steering vector, token position and effective float64 normalization
and rotary patches. Each group prepares its prefix once, compares a cached
batch and a full-prefix batch at its initial states, and compares each lane
against its own full-prefix scalar invocation. Maximum coordinate and
probability discrepancies must each be <=1e-10. These checks do not advance
the state. Then carry the literal output through exactly 240 cached two-lane
updates at H, without resets or outcome-dependent continuation.

This is two groups of 245 attempted invocations: one prefix, two batch
comparisons, two scalar comparisons and 240 updates. The maximum is 490
calls, with 480 batch updates representing 960 state-lane updates. Failed
attempts count. Keep complete input/input-after/output states, full probability
arrays, tokens, precision observations, cache arrays and hashes, per-call
timing, complete trajectories and all partial failures. Completed group
artifacts are immutable.

## Numerical qualification and outcomes

Use the unchanged finite H readout: retain the last 13 states; compare
states[235:241] with states[229:235], and require the largest of the six
lag-six L2 residuals to be <=1e-8. The final six states must exclude proper
periods 1, 2 and 3 by maximum phase gaps >1e-4, and uniquely match one of the
two reference cycles within maximum phase distance <=1e-6 after cyclic
alignment. Retain all six cyclic distances to both references. These are
finite numerical gates, not certified attractor proofs. Readout must not
consult an intended answer.

The fresh h1 control must qualify as H1 and x60 replay as H0. A wrong or
unqualified control yields `control_unqualified`. Otherwise report
`early_handoff_unresolved` if either early state fails finite qualification;
with both qualified, use `both_early_handoffs_h0`, `both_early_handoffs_h1`,
or `mixed_early_handoffs` according to their actual labels. Always report
every row separately. Finite control or readout failures do not skip remaining
groups. Instrument or resource failure aborts, preserves evidence and marks
unrun groups explicitly; it is not a scientific negative.

The two early handoffs are fixed before their new H readouts. Related outcomes,
including the successful x60 replay, are already known. This adapted test can
describe the return property of these two points. It cannot identify first
acquisition of a writable state, a monotone switching threshold, basin volume,
branch loss, text reachability, general write reliability or causal hysteresis.
Do not infer monotonicity or use the result to justify binary search. There is
no automatic longer pulse, longer hold, additional handoff or numerical retry.

## Bounded execution and data return

Use one CPU model process, one numeric thread and batch width two. Set OMP,
MKL, OpenBLAS and NumExpr thread limits to 1 before Python starts. Retain the
tested conservative envelope: 2400 seconds internal numerical time, 2460
seconds external plus 30 seconds termination grace, 2640 seconds outer job,
4800 seconds return worker, 8 GiB peak RSS and 16 GiB virtual memory.
The call cap is 490, not the preceding panel's 735.

Retain ceilings of 2304 MiB raw evidence including a 32 MiB finalization
reserve, 2560 MiB package, 3072 MiB archive, 8 MiB upload and 8 GiB transfer
reservation including fallback. These are bounds, not expected sizes. Measure
the actual complete serialization and return of this two-group schedule before
source freeze. Reuse tested transport components without changing the older
frozen sources. Test actual filenames, incremental return, incomplete groups,
late failure, fallback, verified hashes and termination before launch.

Fresh admission needs at least 24 GiB available RAM, 16 GiB free disk, load
average <=4, no conflicting task-owned model or collector and an idle
human-controlled trial. The shared human maximum remains 7 TB per month.
Use an isolated committed checkout, a verified ready local collector, a
bounded shared reservation and exact process ownership. Return and verify
every artifact under `/data2/shrimp-memory-data` before any remote deletion.
Commit and push producer, independent checker, tested orchestration and the
run card before numerical outcomes. Preserve preparation failures and all
subsequent outcomes; no new rental or external paid call is part of this test.
