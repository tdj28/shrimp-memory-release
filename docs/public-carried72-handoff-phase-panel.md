# A fixed carried two-write word after a 144-update hold

This adapted exploratory panel asks whether the actual state after E0(72),
H144 supports the carried E1(120), H240 write. It follows the independently
accepted matched-prefix census: all six 72-update starts had requested zero
persistently from H117 through H240. This saved-array result motivated the
fixed H144 handoff; it did not execute a second write at that state.

The prior E0(36), H240, E1(120), H240 panel used a different first pulse and
different carried states. The accepted E0(72), H240 panel tested only the
first write. Neither establishes the proposed two-write word. The selection
also requires the accepted 72-versus-120 E0 prefix to have maximum coordinate
gap at most 1e-10 and all 432 individual tokens equal. This is a disclosed
prospective decision condition, separate from the descriptive census, whose
metrics and neutral gates remain unchanged. Root's private selection binds
the accepted receipts, memo and decision calculation before this run.

## Literal inputs and fixed schedule

Use the same ten pinned inputs as the intermediate-duration panel: the original
two historical trajectories, fixed reference inputs, metadata, plan, audit and
manifest, the four-phase prospective packet, and the accepted short-prefix
packet and its extraction proof. Verify all four packet entries, then select
indices [1,2,3] for literal `references[:,3]`, `references[:,4]` and
`references[:,5]`. Do not rotate, settle or replace these starting states.
The initial pair, reference cycles, selected prefixes and explicit phase and
packet indices are retained in `literal-inputs.npz`.

The fixed controls are E0 = (0.07293697809562831, -0.1535),
H = (0.07293697809562831, -0.15395151820007266), and
E1 = (0.07299785589927693, -0.15395151820007266), in
(alpha, log10 temperature) coordinates. Save their exact binary64 values,
hex representations and bits on each call. There is no configurable control,
duration, state, phase, adaptive hold or retry interface.

For each case c=0,1,2, run these four stages in order:

| Group | Name suffix | Control | Updates | Attempted calls |
|---|---|---|---:|---:|
| 4c | intermediate-write0 | E0 | 72 | 77 |
| 4c+1 | hold-after-write0 | H | 144 | 149 |
| 4c+2 | carried-write1 | E1 | 120 | 125 |
| 4c+3 | hold-after-write1 | H | 240 | 245 |

Group names begin with `phase3-`, `phase4-` or `phase5-`. Carry the actual
terminal state bytes through all nine within-case boundaries. Reset only at
groups 4 and 8, to the next literal pair. Every group performs its own prefix,
cached batch, full batch and two full scalar comparison calls before updates.
The comparison outputs do not advance the state. Maximum coordinate and
probability comparison differences must be at most 1e-10.

The exact bound is 1,788 calls: 1,728 batch updates, 12 prefix calls, 24 batch
comparison calls and 24 scalar comparison calls. This represents 3,456
state-lane updates and 3,540 full float64 probability vectors. At vocabulary
151,936, probabilities alone occupy 4,302,827,520 bytes. These counts include
instrumentation, but are not measured latency or a speedup. A corresponding
three-case word with H240 at both holds prescribes 2,076 calls; the conditional
288-call difference is schedule arithmetic only.

## Readouts and diagnostics

Read both lanes at groups 4c+1 and 4c+3. At H144 retain states[132:145] and
compare the six pairs states[139:145] minus states[133:139]. At H240 retain
states[228:241] and compare states[235:241] minus states[229:235]. The first
retained state is context, not a seventh closure pair. Every six-pair L2
residual must be at most 1e-8. Exclude proper divisors 1, 2 and 3 when each
divisor's maximum final-cycle phase gap exceeds 1e-4. Require exactly one
original reference cycle within maximum phase-aligned L2 distance 1e-6 across
all six cyclic shifts. Retain every residual, divisor gap, shift and tie.

The decoder receives no requested bit. Compare its neutral result afterward
with [0,0,1,1] for each case. All twelve rows remain visible, including
qualified wrong values and finite unresolved U. Each case is
`case_qualified`, `case_wrong` or `case_unresolved`; the panel uses the
corresponding `panel_` outcomes. Any U takes precedence over wrong, and every
finite outcome continues the entire prescribed word and subsequent cases.

At the end of each E0 stage, compare only its first 37 states and 36 token
pairs with the accepted same-phase short E0 packet. Across three cases these
are 222 individual lane states and 216 individual tokens. Save all coordinate
and L2 gaps, state-byte equalities and token equalities. `prefix_replay_passed`
requires coordinate gap at most 1e-10 and exact token agreement; state-byte
differences remain separately visible. This instrument diagnostic never
substitutes states, gates a neutral label or triggers another run. A finite
mismatch completes the plan and prevents attributing a difference solely to
the prescribed pulse/hold changes.

## Precision, evidence and bounds

Reuse one loaded CPU Qwen2.5-0.5B-Instruct backend, batch size two and one
numerical thread. All four thread environment limits must already equal one
at startup. Preserve the pinned weights, vectors, configuration and tokenizer,
49 effective float64 normalization patches, one rotary patch, and recorded
float64 operations, probabilities, states and caches. Save full probability
arrays for every returned invocation. Save the original and expanded prefix
caches once per group, hashes before and after each call, and anomalous cache
arrays when instrumentation fails.

The flat evidence contract retains `design.json`, `source.json`,
`input-preflight.json`, `literal-inputs.npz`, `events.jsonl`, every call NPZ
and JSON, group trajectories, six H tails, three prefix diagnostic NPZs,
`result.json` and `manifest.json`. Immutable `group-00.json` through
`group-11.json` markers bind disjoint group artifacts and summaries.
Late mutable events/results belong to the final return. Every call, group,
row, trajectory and tail records case, stage, phase and packet identity.

Instrument defects, nonfinite returns, execution errors and resource failures
stop the execution, retain attempted and blocked-before-invocation evidence,
and explicitly close every partial or unrun group and row. They are not
scientific U. Preflight loads no model and closes all twelve groups as unrun.
Final resource accounting includes result and manifest bytes. Retain every
failed preparation or numerical attempt without silent retry.

The bounds are 3,600 s internal, 3,660 s external plus 30 s grace, 3,840 s
outer and 7,200 s return; 8 GiB RSS and 16 GiB virtual memory; 9 GiB raw
including a 128 MiB finalization reserve, 10 GiB complete package, 12 GiB
archive, 32 GiB cumulative transfer reservation, 8 MiB upload and a 4 MiB
final manifest. These are envelopes, not projected consumption. They do not
relax the shared monthly 7 TB limit.

Before root's sole launch, test the actual ten input pins and preflight,
neutral gates at H144 and H240, all nine carries and two resets, finite
wrong/U continuation, early/late and blocked failures, and one complete
1,788-call serializer with actual probability and cache dimensions. Independently
check the retained fixture and incremental/fallback return contracts; freeze
source and admit measured host and transfer capacity with one return worker.

Success would establish only this fixed two-write word at six literal starts.
Failure would show insufficiency somewhere in that word. Without a fresh
matched E0(72), H240 carried control, it cannot uniquely attribute a later
failure to the shorter first hold. Neither result establishes arbitrary reuse,
off-cycle robustness, minimum duration, monotone duration response, attraction,
causal hysteresis, general reliability, extra register capacity or a text
interface. A follow-up comparator requires a new disclosed decision.
