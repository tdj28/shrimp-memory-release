# Prefix-only geometric readout versus strict qualification

This single adapted retrospective analysis uses saved, audited public numerical
memory evidence. Earlier nearest-reference observations found mistakes after
early correct labels and used hindsight to report stable suffixes. This analysis
instead fixes one online-implementable decision rule before execution. It tests
whether the simpler rule makes a false commitment; it does not infer causes or
validate an actual stopping intervention.

There is no fitted classifier, threshold search or outcome-dependent revision.
The only labeled input is the pinned pair of six-phase, 896-dimensional float64
reference cycles, SHA-256
`d96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0`.
At post-update steps 12, 24, 36 and every subsequent multiple of twelve, compute
the minimum Euclidean distance from the current state to each labeled cycle.
Choose the strictly closer label; an exact equal-distance tie erases and resets
the streak. No tolerance or calibrated confidence is attached to this decision.
The first three consecutive equal non-erased checkpoint labels commit once.
Future states and final labels never enter or revise that commitment. The
pre-update initial state is excluded from the observer.

The comparator is the unchanged `StreamingReadout`: lag-six closure at 1e-8,
proper-divisor exclusion above 1e-4, unique all-phase reference matching at
1e-6, and three consecutive qualified checkpoints. The inherited gate is
recomputed on saved states. All 600-step comparator checkpoints and decisions
must agree with their bound saved audits. Every final qualification, including
the 120-step recovery controls, must agree with the independently audited full
gate. Failure of any source, inventory or arithmetic-agreement gate fails the
analysis; it never converts an unchecked row into a scientific null.

The fixed inventory contains 66 trajectories:

- Eight rows each from `fresh-readout`, `transverse-readout`,
  `transverse-intermediate`, `transverse-amplitude` and `transverse-direction2`:
  40 primary trajectories, each retaining all 600 updates.
- All 26 `p6-recovery` rows, including both unperturbed controls and all
  perturbed phases/signs: a separate 120-update horizon-control cohort.

The six compact receipts in `docs/receipts/` identify the exact immutable runs.
Primary filenames are `run/trial-NN-readback-final-600.npz`; recovery filenames
are `run/trial-NN-final-120.npz`. Rehash the receipt-bound complete audits,
all their input bindings, every consumed trajectory and the references. Reject
missing, duplicated, reordered, unverified or changed rows. Stream one
trajectory at a time and recheck all bindings before declaring completion.
Historical audit inventories may repeat an identical reference record: verify
every occurrence, then coalesce only identical normalized path/size/hash bindings.
Distinct reference paths or conflicting bindings remain errors.
No data are retrieved from a server and no model is loaded or called.

Keep the entire continuation and all 12-update checkpoints. For each policy,
record its immutable first decision, correct or wrong agreement with a qualified
final, no decision, and acceptance when the final remains unqualified. An
unqualified final is unknown, never automatically a wrong bit. Separately list
later opposite geometric labels, exact ties, opposite qualified labels and
unqualified checkpoints. These contradiction checks cover only the declared
12-update grid and original 120/600-step horizons. At each geometric decision,
record whether the instantaneous strict checkpoint and the strict streak
already qualified; geometric label visibility does not establish write readiness.

Report each cohort separately with all planned rows, qualified-final and
unknown-final denominators. Give correct/qualified, correct/accepted-qualified,
wrong, no-decision, and accepted-unknown counts explicitly. Do not pool
correlated amplitudes, signs, phases or trajectories as independent trials.
The direction-two and fresh-line cohorts are descriptive transfer challenges,
not statistically held-out data: their outcomes were already seen and the
entire study was adaptively selected. There is no random timepoint split.

All observers see full numerical states and labeled references. Results cannot
establish text-only or unlabeled auditing, general language understanding,
population reliability, safe early stopping, or hidden communication. Any
computed earlier decision is retrospective; additional and actually avoided
model-call counts are both zero. Correct labels do not certify the actual
carried state is ready for a subsequent write. Observer arithmetic was already
small compared with model inference, so this is a scientific readout diagnostic,
not a new speedup claim.

Use one BLAS/OpenMP thread, process niceness at least 10, at most 1 GiB peak RSS,
120 seconds internally and a 180-second external timeout. Fresh local admission
requires at least 8 GiB available memory and records load. Total output is at
most 8 MiB, with a 5 MiB working allowance and remaining space for final failure
records. Save each completed row in a separate exclusive JSON shard; final
`result.json` retains all 66 planned slots, including failed and unstarted
slots, and `manifest.json` binds every output. Preserve every failed attempt.
No GPU, model, SSH, remote transfer or provider budget is required.

Commit this source, tests and protocol and record a run card before executing
`tools/analyze_causal_geometry.py --data-root /data2/shrimp-memory-data
--output-dir NEW_DIRECTORY`. The source checks its own committed bytes and the
inherited gate/observer source, six receipt files, tests and environment lockfiles
before reading experiment outcomes. Final qualification stores neutral gate data,
without the inherited helper’s source-relative returned/switched field. Synthetic
tests cover causality, ties, later reversals, unknown finals, gate agreement,
real filename contracts, tampering, complete inventory and resource failure.
