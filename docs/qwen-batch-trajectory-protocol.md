# Two-state Qwen trajectory validation

This exploratory engineering test asks whether batch size two preserves two
complete 600-update trajectories of the existing scalar reference. Rows zero
and one, horizon, gates and call schedule were fixed before batch outcomes.
Preliminary scalar decisions for those rows had already appeared in incremental
collector output visible to the coordinating agent. The implementation and
audit authors did not inspect scalar outcomes, but the team was not blinded;
this is an adapted implementation check with a disclosed exposure. The prior
one-step benchmark does not establish long-term
agreement; this experiment can find accumulated drift even when one-step errors
are small. No result is asserted in this protocol.

Select rows zero and one, in original order, from the untouched fresh-readout
packet with SHA-256
`db143ff5bdc264c59303cc6a34d3695ac3ee0d09b74ccf9dc3a71fccafda396c`.
The receiver gets that state-only packet, the pinned Qwen model and steering
vector; it does not receive scalar trajectories, outcome labels, stopping
decisions or reference cycles. Reuse `benchmark_qwen_batch.QwenBackend` with the
same layer-twelve philosophy steering, fixed prefix, native float64
normalization and rotary operations, CPU math SDPA and one thread. Common-hold
controls remain alpha `0.07293697809562831` and log10 temperature
`-0.15395151820007266`.

The fixed schedule comprises 605 model calls: one scalar prefix preparation,
initial cached and full-prefix batch diagnostics, 600 carried cached batch
updates, then final cached and full-prefix batch diagnostics. Diagnostics
produce comparison arrays but never replace the carried state. Each real update
consumes the previous batch output. There is no early stopping, reference
observer or outcome-based choice of state, horizon or tolerance in the runner.

Initial and final diagnostic output coordinates and probabilities must agree
within maximum absolute gap `1e-12`, inclusive, with identical top tokens.
Finite diagnostic disagreement is retained and does not skip any carried
updates. Each call must preserve the input and scalar prefix cache, use an
exact repeated copy for its batch prefix, produce finite float64 arrays and
valid probability rows, and supply fresh native64 operation observations.
Invalid arrays, changed cache/input, missing prerequisites or resource failure
stop execution and retain the failed attempt. A returned state is preserved
even if a later integrity or resource check fails; distinguish returned from
qualified updates in every incomplete summary.

The independent local evaluator uses the already-running scalar full-600
fresh-readout experiment as its reference. Before seeing batch outcomes, the
finite-horizon engineering agreement gate is fixed to a maximum absolute
coordinate difference of `1e-10` at every corresponding step, zero top-token
mismatches, identical final qualified labels and first shadow-decision
label/step, and matching later-loss and later-switch flags together with exact
equality of the later-unresolved and later-different-label checkpoint lists.
These list comparisons include checkpoint timing, not only whether a loss or
switch occurred, and are fixed before batch trajectory outcomes. Both runners
must have complete independently audited raw evidence. The coordinate tolerance
is deliberately two orders above the one-step diagnostic tolerance; it is an
engineering criterion rather than a certified error bound. Scientific
disagreement must remain an auditable numerical null, not be disguised as an
integrity failure. An unresolved final label remains unresolved; the independent
receipt must distinguish agreement on unresolved status from a qualified symbol.

Record every update's forward duration and total process elapsed time. Forward
timers include normal batch-cache construction, transformer, softmax and
embedding feedback; they exclude input-tensor preparation, audit snapshots and
hashes, I/O and scalar comparison. Report full overhead and shared-host load
separately. The scalar run includes its own cache checks, all eight rows and a
shadow observer, while this run has two rows and no online observer. Whole-run
durations therefore are not directly interchangeable. A finite-horizon numerical
match alone does not establish general acceleration, global attractor
equivalence, fractal basin geometry or practical language-mediated communication.

Bounds: 605 calls, 900 internal seconds, external 960 seconds plus thirty
seconds of termination grace, eight GiB peak RSS, sixteen GiB virtual memory,
one thread and at most one other single-thread model process after measured
headroom. The raw cap is 64 MiB with a 256 MiB transfer reservation. State,
diagnostic and cache arrays are small enough without assuming compression;
the implementation also reserves space for the partial final trajectory.
No model acquisition, new rental, trajectory retries or adaptive reruns are
part of this design. A native-library delay may defer Python signals, so an
external watchdog remains necessary. Flush immutable evidence before processing
further termination signals.

The producer is `tools/validate_qwen_batch_trajectory.py`. Its fixed CLI is
`--cache-dir --vectors --packet --packet-sha256 --output-dir`. It saves:

- `inputs.json`, `metadata.json` and `initial-states.npz`, binding the exact
  input/source/model/config/vector hashes, prefix IDs and selected rows.
- `prefix-cache-initial.npz`, `prefix-cache-expanded.npz` and
  `prefix-cache-final.npz`, retaining initial/final scalar cache tensors and the
  actual repeated cache from the initial cached diagnostic.
- Four `diagnostic_INITIAL-OR-FINAL_CACHED-OR-FULL.npz` files (lowercase names
  with underscores), each containing input, output, input-after and complete
  probability arrays, plus the two diagnostic comparison JSON files.
- One immutable `call-NNN-LABEL.json` per attempted nonprefix call, with timing,
  top tokens, fresh dtype observations, input/output/probability array hashes,
  source/expanded cache hashes and lengths, and every gate outcome. Ordinary
  update probabilities are not retained in full: their hashes, normalization
  statistics and top tokens are recorded evidence, not independently replayed
  probability calculations. A failed returned update also retains its full
  output/probabilities in `failed-batch_step_NNN.npz`.
- `trajectory-block-012.npz` through `trajectory-block-600.npz`, closed and
  fsynced every twelve updates. Each stores twelve post-update states for both
  rows, top tokens, forward times, exact state indices and controls. A
  `trajectory_saved` event follows the closed file and includes its hash.
- `trajectory-final-NNN.npz`, containing both initial states and every returned
  update, including partial work; on full completion the state array has shape
  `(601, 2, 896)`. `events.jsonl` accounts for every attempted model call;
  `results.json` lists all 604 nonprefix slots, including unstarted slots,
  prefix status, block/final hashes and completed/incomplete/numerical status.

Array hashes include dtype, shape and C-order bytes. Cache hashes bind a sorted
mapping of named tensor hashes. The independent audit reconstructs all retained
array hashes, cache snapshots and block boundaries; intermediate cache checks
and ordinary probability calculations remain producer-recorded evidence unless
the model is separately replayed. Commit source and a run card before launch;
return and hash-check every raw artifact, log and incomplete attempt in
`/data2/shrimp-memory-data` before closing the transfer reservation.
