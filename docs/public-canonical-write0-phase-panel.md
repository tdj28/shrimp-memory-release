# Canonical WRITE0 at the four remaining literal reference phases

This adapted exploratory panel compares the canonical 120-update E0 pulse
with the accepted 36-update pulse at the same original reference phases. The
short-pulse panel returned qualified H1 after WRITE0 from initial H1 phases
3, 4 and 5; phase 2 and all H0 starts returned H0. Its subsequent WRITE1
successes do not repair those wrong WRITE0 outcomes. The independently
accepted zero-call census found no earlier qualified zero for those three
failed rows at checkpoints 12 through 240. Their first qualified wrong labels
occurred at 188, 181 and 180; no common requested checkpoint exists for all 24
trajectories. These observations motivated this design. They remain retained.

Decision evidence is bound in
`public-canonical-write0-phase-panel/20261029T172047Z/root-selection.json`
under the private preparation root. The accepted receipts are
[the short panel](receipts/public-short-write0-phase-panel.json) and
[the prefix census](receipts/public-short-write0-prefix-census.json).
The previous six-phase round trip starts from H0, writes H1, then writes H0
from the actual carried terminal state. It does not test this literal
two-reference initial-state panel.

## Fixed inputs and controls

Keep all eight pinned inputs of the
[short-pulse panel](public-short-write0-phase-panel.md), including its original
references and prospective phase packet. Add these two immutable local inputs:

| File | Bytes | SHA256 |
|---|---:|---|
| accepted-short-prefix.npz | 2126228 | `810b396212c88bad19225d363e53ac2e3327cb315f8a0e003de4fe3bea68db72` |
| accepted-short-prefix-provenance.json | 46218 | `b22f602afa95a0b321c432bb966cf89502da3b2f42f5463bd65d6177c35d1d4b` |

The packet contains float64 `states[4,37,2,896]`, int64
`token_ids[4,36,2]`, `reference_phases=[2,3,4,5]`,
`reference_indices=[0,1]`, `state_indices=0..36` and
`update_indices=1..36`. Its pinned provenance binds the accepted short-panel
receipt, independent audit, result, run/final manifests and all 148 source
files: four state trajectories and 144 raw update NPZs. Extraction verified
each input/input-after/output chain and token against the saved probabilities.
Independent local preparation must reconstruct every packet array from those
sources. Remote preflight verifies the packet/provenance pins and complete
shape, dtype, array hash, phase and literal-start contracts; it does not
download the old bulky run. None of the packet states supplies a fresh state.

Execute phases 2, 3, 4 and 5 in order with one loaded CPU model of batch size two. Each case
starts from the exact pair `references[:,phase]`. Carry actual returned bytes
from E0 into H. Reset only between cases to the next declared literal pair;
never settle, rotate, normalize or substitute a state. The original eight
inputs plus the two new files total 4,134,553 bytes before source packaging.

In (alpha, log10_temperature) order, E0 is
(0.07293697809562831,-0.1535) and H is
(0.07293697809562831,-0.15395151820007266). Save literal binary64 values and
bits on every call. There is no production phase, control, duration, payload
or state selector. Use the pinned original Qwen weights, prefix, steering
vector and token position. Keep effective float64, all 49 native normalization
patches and one rotary patch, and intra-op/inter-op plus all four numerical
thread environment limits equal to one at startup and runtime.

## Eight groups and eight readouts

For case index c=0..3, group 2c is `phase{c+2}-canonical-write0` and
group 2c+1 is `phase{c+2}-hold-after-write0`. Both lanes retain their original
reference indices; they are starting-state comparisons, not two payload bits.

| Per-case group | Control | Updates | Attempted calls | Readout |
|---|---|---:|---:|---|
| 2c | E0 | 120 | 125 | Full trajectory and first 36 updates as prefix diagnostic |
| 2c+1 | H | 240 | 245 | Two neutral labels, then compare with requested 0 |

The exact maximum is 1,480 attempted calls: 8 prefix, 16 cached/full batch
comparisons, 16 scalar comparisons and 1,440 batch updates, representing 2,880
state-lane updates. Comparison outputs never advance the carried state.
Each stage performs its own prefix and cached/full/both-scalar comparisons
with maximum coordinate and probability gaps <=1e-10 before updates. Preserve
all full probability vectors, token IDs, input/input-after/output states,
precision observations, original/expanded caches, hashes and timings.

At each H readout save `states[228:241]`. Require maximum L2 residual over the
six pairs `states[235:241]` minus `states[229:235]` to be <=1e-8. Exclude proper
periods 1, 2 and 3 with maximum phase gaps >1e-4, and require exactly one reference
cycle within maximum aligned phase distance 1e-6 over all six cyclic shifts.
Save every residual, divisor gap and alignment. Decode without requested-bit
access; only then classify each row correct, wrong or unresolved. The gates
and neutral decoder are unchanged.

After each completed E0 stage, compare all 37 fresh states 0..36 and 36 token
pairs with that same phase's accepted short-pulse prefix. Retain every
coordinate and L2 difference, byte equality and token equality. The diagnostic
passes only if maximum coordinate difference <=1e-10 and all tokens match
exactly. State-byte inequality is disclosed even when within tolerance.
Report this instrument diagnostic separately; finite mismatch does not change
the readout gate, replace states, select another pulse, stop remaining finite
cases or cause a retry. A mismatch limits attribution solely to pulse duration.

Each case is `case_qualified`, `case_wrong` or `case_unresolved`; the panel is
`panel_qualified` only for eight qualified zeros. Any unresolved row takes
precedence as `panel_unresolved`; otherwise a wrong label is `panel_wrong`.
All finite wrong/U labels continue. Instrument, execution and resource
failures retain their own status, partial calls, all planned rows, unrun
reasons and immutable group 00..07 closures. A completed group is completed
regardless of finite row labels. Prefix diagnostics do not determine these
scientific outcome enums. There is no automatic retry or hold extension.

## Resource, source and return gates

Keep the selected bounded envelopes: 9 GiB numerical raw including 128 MiB
finalization reserve; 10 GiB complete package; 12 GiB archive; 8 MiB upload;
32 GiB shared transfer reservation; 4 MiB final-manifest metadata. These are
caps, not estimated usage. Keep 8 GiB RSS, 16 GiB virtual memory, 3,600 s internal,
3,660 s external plus 30 s grace, 3,840 s outer and 7,200 s return-worker bounds.
The schedule requires 2,936 full float64 probability vectors of 151,936 entries,
exactly 3,568,672,768 probability bytes before caches and metadata. Measure the
complete actual 1,480-call serialization and resource/return contracts before
source freeze. Exercise early, pre-invocation, late and finalization failures.

Fresh admission, isolated committed source, exact source/input upload proof,
one ready return worker and shared transfer accounting are prerequisites to
the sole root launch. Preserve immutable incremental closures and all late
files in final return; after the first incremental failure use one bounded
final fallback. No duplicate collector or model process. Reuse the prior
reviewed transport guards, including exact process ownership and cleanup.
The monthly maximum remains 7 TB; a per-run reservation is not permission to
exceed it. No new rental or external paid call is part of this work.

Before launch, the independent checker must pass real file-contract fixtures
and rederive every neutral gate, literal start, all four carry/three reset
boundaries, complete call ledger, precision/cache evidence, prefix diagnostic
and raw/return hashes. Commit tested producer/checker/protocol and run card
before outcomes. Keep old frozen sources, failures and nulls unchanged.

Success would support only the longer pulse at these eight literal starts
under fixed H240. Compare the three earlier wrong rows and the five earlier
correct rows explicitly. Retain regressions as well as corrections. No
monotonic duration response, minimum duration, general reliability, speedup,
asymptotic attraction, causal hysteresis, subsequent reuse or language
interface is established by this panel. The longer pulse adds 336 calls to
the corresponding short-WRITE0-only panel; any runtime result must include
loading, comparison, serialization, unsuccessful outcomes and return costs.
