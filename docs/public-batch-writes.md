# Two-state public write equivalence

This is one adapted exploratory engineering test. Earlier batch trajectories
and readout used fixed hold controls. This test asks whether the same batch
implementation preserves a complete write, including the control change and
prefix-cache rebuild. It neither changes the public command grammar nor tests
independently addressed registers.

The visible message for each row is exactly `Write bit 1 to register R0.`.
Row zero starts at saved ad phase zero; row one starts at saved td phase zero.
Both use the unchanged canonical action: 120 endpoint updates followed by 240
hold updates. The endpoint alpha is 0.07299785589927693, the hold alpha is
0.07293697809562831, and log10 temperature is -0.15395151820007266 throughout.
Rows remain in this order and evolve together. No reset, replacement, row
removal, early stop, adaptive extension or retry is permitted.

The implementation uses the pinned Qwen2.5-0.5B-Instruct snapshot, fixed
37-token prefix and philosophy vector at layer twelve. A local adapter retains
the original vector explicitly. For each stage it uses the existing
`set_controls` helper and rebuilds the scalar prefix before repeating its
cache into two separate batch rows. Historical module globals are unchanged;
the vector is never recovered by dividing an earlier steering tensor. Native
float64 normalization and rotary instrumentation must be fresh for each call;
CPU math SDPA and one thread remain fixed. This instrumentation is not a new
global precision or derivative-convergence proof.

Each stage executes one prefix preparation, a cached/full batch diagnostic
pair which never advances the carried state, and its fixed updates. This is
366 calls: 360 batch updates, two prefixes and four diagnostics; the two rows
undergo 720 updates. Cached/full output-coordinate and probability maximum
absolute gaps must each be at most 1e-12 with identical top tokens. A finite
diagnostic disagreement is retained and the complete schedule still executes.
Mutation of the scalar cache or input, wrong cache dimensions/length,
nonfinite or non-float64 tensors, missing fresh precision observations,
execution errors or resource limits abort with planned and partial slots saved.
In incomplete attempts, `model_calls` counts admitted slots against the cap.
Each nonprefix slot separately records whether its forward was attempted;
a failed pre-forward cache inspection can consume an admitted slot without
executing a model forward. Returned update counts and validated carried update
counts are recorded separately.

Before model loading, qualify the immutable local/returned references and
recovery prerequisites, then bind metadata, results and both stages from
historical isolated scalar trials 2 and 3. The historical source is
`f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4`. Model, configuration, vector,
reference, prerequisite, mathematical source, prefix, thread and control
identities must agree. Historical files use 121/241 float64 states and 120/240
int64 top tokens. Their initial bytes and interstage continuity are checked.
The first batch initial is saved before loading; the hold initial is the actual
carried endpoint, saved once on arrival at that stage.

Compare every stage state and top token with the saved scalar trajectory.
The engineering gate is maximum absolute coordinate error at most 1e-10,
zero token mismatches and matching qualified terminal labels. Report byte
equality separately; previous scalar/batch results were not bitwise identical.
The two stage arrays include their common boundary twice for the stage checks,
but their union contains 361 distinct carried time points per row and 360
tokens. A finite trajectory mismatch does not select, replace or stop a row.
The terminal readout independently applies the full period-six recurrence,
proper-divisor exclusion and unique all-phase reference-match gates, without
receiving a desired bit. Correctness relative to the visible request is a
separate independent evaluation. An intact run may complete while its
numerical-equivalence gate fails.

Evidence is compact without relying on compression. Save each stage initial
and nonoverlapping blocks of twelve post-update batch states, top tokens and
forward durations. Preserve a shorter last block on interruption. A final
trajectory JSON binds the initial and all blocks; do not duplicate the complete
trajectory into another NPZ. Each stage retains its initial scalar prefix and
the actual repeated prefix from its cached diagnostic. All calls record input,
output, probability and cache hashes, lengths, fresh dtype observations and
timing. Final scalar-cache hashes are retained without duplicate snapshots.
The four diagnostics retain complete probability arrays; an invalid or failed
returned map call retains its available arrays separately. Ordinary probability
hashes and intermediate cache observations remain producer evidence without
model replay.

The planned uncompressed envelope is 28,627,392 bytes: 5,189,632 for states,
8,640 for tokens/times, 10,911,744 for cache tensors, 9,895,936 for diagnostic
arrays, 524,288 for containers and 2,097,152 for metadata. This fits a 30 MiB
working allowance while reserving 2,473,984 bytes plus a small event allowance
before every model call for a failed native-shape return. A further 2 MiB is
reserved for final manifests and the pending block, within 32 MiB total raw.
Actual admission is checked before each write and forward. Bounds are 600
internal seconds, 780 external seconds plus 30 seconds termination grace,
8 GiB RSS, 16 GiB external virtual memory, one thread, 48 MiB return archive,
64 MiB transfer reservation and at most 8 MiB source upload. Model weights are
already remote and are never uploaded from the local cache.

The source is `tools/run_public_batch_writes.py`. Its CLI requires
`--cache-dir --vectors --references --reference-manifest --source-hold-results
--recovery-results --historical-run --output-dir`, plus the three corresponding
`--references-sha256 --reference-manifest-sha256 --recovery-results-sha256`
values. The historical run directory is
`audits/p6-writes/20261025T222347Z-writes/run` under the data root. Source,
protocol and inputs must be committed and a run card recorded before the one
model run. A zero-forward actual-input preflight and model-free corruption,
failure and archive-reconstruction tests precede launch. Every output,
including a failed or incomplete attempt, returns to the local data root and
is hash checked before closing its transfer reservation.

Report measured absolute process/update times, initialization, calls, retained
row updates, correctness, erasures and equivalence failures. Historical scalar
timing is descriptive. The prescribed comparison of 366 with 732 calls is
not a runtime speedup measurement, single-register latency reduction or useful
payload throughput. Success would extend finite batching equivalence to two
known starts under one common write. It would not establish arbitrary
addresses, a cell bank, reliability across phases, natural-language semantics,
logic or hidden communication.
