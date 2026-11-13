# Remaining reference phases of the shortened write-and-reuse chain

This adapted exploratory panel tests the same finite two-write word from the
four remaining original reference phases. The phase-zero and phase-one runs
both qualified [0,0,1,1]. A conditional proposal for this panel was sealed
before the phase-one result; its success branch is now selected. The earlier
runs remain historical observations, not fresh comparison arms. This is
public numerical memory research.

## Fixed inputs and case boundaries

Keep the seven original historical input pins and provenance checks from
[the phase-zero protocol](public-short-write0-chain.md). Add
`prospective-phase-panel.npz`, 55637 bytes, SHA256
`513ebc3988dd50502324986dbe2b2647b0a78e12b056031b6986505373e197a3`.
Its only arrays are `initial_states` (float64, shape 4 by 2 by 896),
`reference_phases` (int64 [2,3,4,5]) and `reference_indices` (int64 [0,1]).
Require each initial pair to equal the original `references[:,phase]` byte
for byte. In phase order, the raw C-order pair hashes are:

| Phase | Initial-pair SHA256 |
|---|---|
| 2 | `87bf7f61d3c06d8b47545b97c7ab8d8c89551f31e0dc985d2d571fff0efe83d4` |
| 3 | `a6f116dfe2c09f5b2ec9fbac29e4633e40bccdbd3ddc9324b69dad68dac0ff45` |
| 4 | `9cfc5c30be03cdaac065a774947e2754049311ef032f8c169e9c1e5ba9c40820` |
| 5 | `7fd2448b6669a1afe03abb1a8a0cee8448ac0a098a2490d3738a65e9ddc0db49` |

Execute phases 2,3,4,5 in that order with one loaded model, one numerical
thread and batch size two. Each case starts from its declared literal pair.
Within each case, carry the actual returned bytes across all three stage
boundaries. Reset only between cases, to the next declared literal pair.
Never settle, rotate, normalize, search for or substitute a favorable state.
The lanes represent two initial stored values, not two payload registers or
statistical replicates. There is no phase, duration or payload selector in
the production command line.

## Schedule and instrumentation

Each case has four stages. Use global group numbers 0 through 15, with
`case_index = group_id // 4` and `stage_index = group_id % 4`. Preserve both
identifiers and the reference phase in every group and readout record.

| Stage within case | Control | Updates | Attempted calls | Readout |
|---|---|---:|---:|---|
| 0 | E0 | 36 | 41 | Complete fresh WRITE0 trajectory |
| 1 | H | 240 | 245 | Decode both lanes, then compare with zero |
| 2 | E1 | 120 | 125 | WRITE1 from the actual preceding terminal pair |
| 3 | H | 240 | 245 | Decode both lanes, then compare with one |

The complete panel has 2624 attempted calls: 16 prefix, 32 batch-comparison,
32 scalar-comparison and 2544 batch-update calls. Those updates represent
5088 state-lane updates. Every stage charges its own prefix and
cached/full/both-scalar checks; no prefix saving or larger-batch equivalence
is assumed. Failed attempted calls count toward the same bound.

Use the original pinned Qwen model, prefix, steering vector and token position.
Require effective float64, all 49 normalization patches and the single rotary
patch, and all four numerical thread environment limits equal to one at
startup and runtime. In (alpha, log10 temperature) order the exact binary64
controls are H=(0.07293697809562831,-0.15395151820007266),
E0=(0.07293697809562831,-0.1535) and
E1=(0.07299785589927693,-0.15395151820007266).

At each stage start, require cached/full-prefix/both-scalar maximum coordinate
and probability discrepancies at most 1e-10. Preserve every full probability
vector, token, input/input-after/output state, precision observation, original
and expanded cache, cache hash and timing. Preserve complete trajectories,
immutable closed groups, failed or partial calls and explicit unrun stages.

For each H return, retain the last 13 states and all six lag-six residuals
between states[235:241] and states[229:235]. Require maximum L2 residual at most
1e-8, proper periods 1,2,3 excluded by maximum phase gaps greater than 1e-4,
and exactly one reference cycle within 1e-6 after evaluating all six cyclic
alignments. Preserve every alignment distance to both references. Desired
bits are unavailable to the neutral decoder and are compared only afterward.

Retain the descriptive comparison of each new E0 trajectory's 37 states and
36 token outputs per lane with the old phase-zero scalar histories. These
are different initial conditions. The comparison has no replay tolerance,
pass/fail threshold or scientific gate and cannot supply a replacement state.
No fresh phase-zero or phase-one comparator is included.

## Outcomes and evidence

Record sixteen readout rows in case order, each case requesting [0,0,1,1].
Continue all later stages and cases after finite wrong or unresolved readouts.
Only instrument, execution or resource failure can stop the panel. Preserve
all planned cases and rows even when unrun. Classify each completed case as
qualified, wrong or unresolved; the panel qualifies only if all sixteen
requested labels qualify. Any unresolved row gives an unresolved panel;
otherwise any qualified wrong label gives a wrong panel. Execution failure
remains distinct from a scientific null.

Use one panel producer, one independent saved-array checker, one protocol and
run card, and one compact four-by-four result/gate table. Do not make a new
runner or figure for each phase. Older frozen scientific sources and evidence
remain unchanged. The checker must independently reconstruct all readout
gates, call ledgers, literal starts, twelve within-case carry boundaries and
three between-case resets, and verify every raw and returned-file binding.

## Resource and return bounds

The complete schedule has 5200 full float64 probability vectors of 151936
entries, exactly 6,320,537,600 probability bytes. The prior synthetic serializer
projects about 6,602,261,868 total raw bytes; this projection is not a measured
upper bound. Retain at least 128 MiB of numerical finalization reserve.
Candidate caps are 9 GiB numerical raw, 10 GiB complete package, 12 GiB
archive, 8 MiB source/input upload and a 32 GiB transfer reservation. Prove
these limits with a new actual full-size serializer, complete incremental
and fallback return rehearsal, and early/late failure cases before launch.
Bound final-manifest metadata at 4 MiB, allowing the larger inventory and
full-fallback file list; include it in the return reserve. Bound each transfer
or collector operation at 600 seconds, each packer at 300 seconds and the final
return helper at 1800 seconds, all inside the total return-worker limit.
Charge attempted transfers cumulatively and preserve the final-return reserve
before each shard. Never multiply the whole archive cap by sixteen when
summing disjoint group payloads. One failed incremental attempt ends
collection and uses one bounded final fallback; no duplicate collector.

Keep 8 GiB RSS, 16 GiB virtual memory, 4800 seconds internal, 4860 seconds
external plus 30 seconds grace, 5040 seconds outer and 9600 seconds for the
return worker. Fourfold extrapolation from the older phase-zero elapsed time
is about 1979 seconds; it is a planning estimate, not observed performance.
Measure actual loading, forwards, diagnostics, serialization, total elapsed,
RSS, raw volume and transfer consumption, including unsuccessful outcomes.

Admission requires at least 24 GiB available RAM and 64 GiB free disk on the
standing host, load average at most four, no conflicting owned model or
collector, an idle human trial, current transfer accounting and a verified
ready return worker. Measure local test and return capacity separately; use
the sibling data root for bulky fixtures. The monthly maximum remains
7,000,000,000,000 bytes with the existing shared warning, stop and return
reserves. This reservation is a per-run envelope, not a monthly cap.

Commit and push tested code, checks, protocol and run card before outcomes;
use an isolated checkout and fixed wall/call/resource limits. Return and
hash-check every output under `/data2/shrimp-memory-data` before considering
remote evidence deletion. No new rental or paid review is needed.

If all cases qualify, combining this panel with the historical phases 0 and 1
completes a finite census of the twelve original literal starts for this
specific two-write word. It does not establish off-cycle or noisy robustness,
long-term repeated reuse, a probability of error, minimum pulse duration,
asymptotic attraction, causal hysteresis or a language interface. Diagnose a
wrong or unresolved row from saved arrays first; no automatic numerical retry,
longer hold, alternate phase or retrospective threshold change is allowed.
