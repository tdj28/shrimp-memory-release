# A fixed intermediate WRITE0 duration

This adapted exploratory panel tests whether a 72-update E0 pulse recovers
the three previously wrong H1-origin writes at original reference phases 3,
4 and 5 while preserving the H0 controls. The accepted 36-update panel gave
[0,1] at each of these phases; the matched 120-update panel gave [0,0]. Both
used H240. Keep those outcomes and every earlier wrong or unresolved write.

The accepted canonical prefix census found no primitive-six qualification
through E0 update 120 for the H1-origin rows, although all final H readouts
qualified zero. H0-origin endpoint tails first qualified at updates 90–95;
the eight H returns qualified zero by updates 112–117. These saved prefixes
do not validate an earlier handoff. H stays at 240 here to isolate pulse
duration rather than changing two parts of the schedule simultaneously.
Neither endpoint recurrence nor an earlier neutral label gates this panel.

Selection is bound by the private
`public-intermediate-write0-phase-panel/20261029T183414Z/root-selection.json`
(SHA256 `c7572a5c16a337c3124eaafa7e98965ed9e01393470463e944e7e3b3997d9fb6`).
The accepted sources are the [short panel](receipts/public-short-write0-phase-panel.json),
[canonical panel](receipts/public-canonical-write0-phase-panel.json), and
[canonical census](receipts/public-canonical-write0-prefix-census.json).
The retained duplicate review found no matching E0(72), H240 test from these
six literal starts. Earlier 60-update, repaired-state, carried-state and
192-hold experiments use different starting states or schedules.

## Fixed input population

Use the same ten pinned input files as the
[canonical panel](public-canonical-write0-phase-panel.md). Validate the entire
original prospective packet and same-phase short-prefix packet, including
their unused phase-2 entry, before selecting the fixed indices [1,2,3].
The original packet phases are [2,3,4,5]; the selected literal phases are
[3,4,5]. No command-line phase, duration, control or state selector exists.

Each case starts from the exact pair `references[:,phase]`, one H0 and one
H1 lane. The literal artifact retains `initial_states[3,2,896]`,
`short_prefix_states[3,37,2,896]`, `short_prefix_token_ids[3,36,2]`,
`packet_indices=[1,2,3]`, `packet_reference_phases=[2,3,4,5]` and
`reference_phases=[3,4,5]`, together with original references and historical
phase-zero arrays. All selected arrays are copied only after full-file,
dtype, shape, phase, array-hash and literal-start validation. The pinned
short-prefix proof continues to bind all 148 original source files. Remote
preflight uses the fixed packet/proof; independent local audit reconstructs
their source arrays. No historical output replaces a fresh state.

The ten original files total 4,134,553 bytes before source packaging. Keep
the pinned Qwen weights, steering vector, prefix and token position; obtain
no weights through source upload. In (alpha, log10_temperature) order E0 is
(0.07293697809562831,-0.1535), and H is
(0.07293697809562831,-0.15395151820007266). Every call retains exact binary64
controls and bits. Effective float64, 49 normalization patches, one rotary
patch, and intra-op/inter-op plus all four numerical thread limits remain
unchanged. Use one loaded CPU model with validated batch size two.

## Six groups, six neutral rows

Run phases 3, 4 and 5 in order. For case c=0..2, group 2c is
`phase{c+3}-intermediate-write0`; group 2c+1 is
`phase{c+3}-hold-after-write0`. Each raw call, trajectory, tail, marker,
summary and readout retains case, stage, reference phase and source-packet
index. The three endpoint-to-hold boundaries carry actual bytes. Only the
two between-case boundaries reset to the next declared literal pair.

| Group within each case | Control | Updates | Attempted calls |
|---|---|---:|---:|
| 2c | E0 | 72 | 77 |
| 2c+1 | H | 240 | 245 |

The exact maximum is **966 attempted calls**: 936 batch updates, six prefix
calls, 12 cached/full batch comparisons and 12 scalar comparisons. Those
updates represent 1,872 state-lane updates. Every stage performs its own
prefix and cached/full/both-scalar checks before updates. Maximum coordinate
and probability differences must be at most 1e-10. Comparison outputs never
advance the carried state. Preserve full probabilities, tokens, state
input/input-after/output arrays, precision observations, original/expanded
caches once per group, per-call cache hashes and timings.

After each complete E0 stage, compare its first 37 states (indices 0–36) and
36 token pairs with the same phase's accepted short-pulse prefix. The three
diagnostics cover 222 individual lane states and 216 individual tokens
(111 two-lane state pairs and 108 token pairs). Retain all coordinate
and L2 differences, state-byte and token equalities. A diagnostic passes only
when maximum coordinate gap is at most 1e-10 and every token matches exactly.
State-byte disagreement remains visible even within tolerance. Report this
separately from neutral readout; a finite mismatch does not substitute states,
alter a label, stop later finite cases or trigger a retry. It prevents
attribution of a difference solely to pulse duration.

At H240 retain `states[228:241]`. Require all six L2 differences between
`states[235:241]` and `states[229:235]` to be at most 1e-8. Exclude proper
periods 1, 2 and 3 by maximum final-cycle phase gaps greater than 1e-4. Require
exactly one original reference cycle within maximum phase-aligned L2 distance
1e-6 over all six cyclic shifts. Preserve every residual, divisor gap, shift
distance and tie. The neutral decoder receives no requested bit. Compare
its result with zero only afterward; unqualified finite rows remain U.

Each case is `case_qualified`, `case_wrong` or `case_unresolved`. All six
qualified zeros give `panel_qualified`; any U gives `panel_unresolved`;
otherwise any qualified one gives `panel_wrong`. All finite wrong/U rows
continue through the fixed plan. Instrument, execution and resource failures
remain separate, retaining partial attempts and all planned rows/groups with
explicit unrun reasons. A completed group is completed regardless of its
finite readout. There is no automatic retry, longer hold or duration search.

## Evidence and bounded resources

Use new producer/checker/test files under the single
`public-intermediate-write0-phase-panel` profile. Preserve earlier frozen
sources. The flat file contract retains `result.json`, `manifest.json`,
`events.jsonl`, `source.json`, `design.json`, `input-preflight.json`,
`literal-inputs.npz`, every attempted `call-NNN-*.npz` and JSON record,
`group-00.json` through `group-05.json`, their immutable summaries and
trajectories, H tails, and `case-00-prefix-replay.npz` through case 02.
Closed markers exclude mutable late result/events files. Final return must
include those late files and every failed or unrun record.

The schedule has 1,914 full float64 probability vectors of 151,936 entries,
exactly 2,326,444,032 probability bytes before caches and other evidence.
Measure one complete actual 966-call serializer with scripted outputs, full
cache geometry and exact final file accounting. Test actual input pins,
preflight with no model, early/late/pre-invocation failures, all carry/reset
boundaries and neutral wrong/U retention before freezing source. A serializer
fixture is not a scientific model run.

Keep the conservative caps: 9 GiB raw including 128 MiB finalization reserve,
10 GiB complete package, 12 GiB archive, 8 MiB source/input upload, 32 GiB
transfer reservation and 4 MiB final manifest. Limits are 3,600 s internal,
3,660 s external plus 30 s grace, 3,840 s outer and 7,200 s return worker;
RSS is bounded at 8 GiB and virtual memory at 16 GiB. These are envelopes,
not projected usage or permission to consume the shared monthly 7 TB limit.

Fresh capacity/process/idle-human-trial/transfer admission, isolated committed
source, measured upload, complete return rehearsal and one ready return worker
are prerequisites to root's sole launch. After the first incremental-return
failure, stop collection and use one bounded final fallback. Preserve task-only
process ownership and cleanup; do not duplicate a model or collector. Return
and hash-check every artifact in the local data root, preserving remote
originals. No new rental or paid review is part of this design.

The matched 120-update schedules for these same three cases prescribe 1,110
calls, versus 966 here: 144 fewer calls, conditional on successful writes.
This is not a measured runtime speedup; do not compare with the four-case
1,480-call total as if all starting states were matched. H still consumes
720 of 936 updates. Record full loading, forwards, serialization, failed
outcomes and return costs. Success would establish only this fixed schedule
at these six starts, not a minimum duration, monotone response, general
reliability, off-cycle robustness, carried reuse, attraction, causal
hysteresis, a new register count or a language interface.
