# Public-controller endpoint perturbation pilot

This adapted exploratory experiment measures the response to five discrete
endpoint settings of an explicit public WRITE0 or WRITE1 action. It is a
prerequisite sensitivity measurement, not a language-interface solution or a
continuous tolerance bound. No covert text, semantic projection, carrier
selection, or hidden optimization is used.

Each process has one requested bit and profile, fixed before execution. The
original `axis-stencil` profile is the default. Its five groups run in
this order: nominal, alpha-minus, alpha-plus, log10T-minus, log10T-plus. Each
group starts two batched rows from the exact saved ad phase-zero and td
phase-zero states; the previous group's terminal states are not reused. The
nominal public action remains the unchanged strict command and is logged
separately from the realized control perturbation. Every group executes 120
endpoint updates and 240 updates at the unchanged common hold. The four
perturbed groups add respectively -1e-5 or +1e-5 to endpoint alpha, or -1e-4
or +1e-4 to endpoint log10 temperature. These amplitudes are engineering
choices; neither they nor success at these points establish text tolerances.

Each stage rebuilds the prefix and executes the existing cached/full batch
diagnostic pair before updates. The nominal group alone is compared against
the same audited scalar historical trajectories as the earlier batch-write
tests. The perturbed trajectories deliberately use different controls and
must never be labeled equivalent to those historical nominal trajectories.
Hold readout receives no requested bit. Its qualified label is compared with
the visible request afterward: requested label is `passed`, qualified
opposite label `failed`, no qualified label `unresolved`. Incomplete slots
remain `not_evaluated`. Execution completion, diagnostic validity and nominal
equivalence are separate fields; a numerical outcome from a group with a
failed diagnostic is not a valid robustness success.

The runner reuses the validated native-float64 batch backend, per-call
precision/cache/input gates, nonoverlapping twelve-update block archive and
reconstruction contract. Historical modules and constants are unchanged.
Each of the ten stages and 1,830 call slots has a globally unique identifier;
there are no repeated output-directory wrappers. The hold initial is the
actual terminal endpoint state. Every later group explicitly resets to the
same bound references. All planned rows and calls are recorded before
execution. Finite diagnostic or nominal-history disagreement does not stop
or select the subsequent groups. Structural, precision, execution or resource
failure aborts while preserving completed rows, partial blocks and all
unstarted slots. No retries, adaptation, trajectory extension or outcome-based
group removal occurs within this run.

The last-written immutable marker `group-NN.json` includes an `artifacts`
inventory of every closed file belonging to that completed group: both stage
initials, blocks, trajectory descriptions and cache snapshots, diagnostic
arrays and completed row records. All entries carry basename, byte size and
SHA-256. The marker excludes the append-only event stream and final/global
files and is never overwritten. A collector may safely return this inventory
and marker during execution; final retrieval still includes every remaining
file, including failed and partial groups.

Per process the schedule contains ten rows, ten prefixes, twenty diagnostic
calls and 1,800 batched updates: 1,830 calls and 3,600 row updates. Two separate
processes cover both requested bits, for twenty rows total. Prior 366-call
groups took roughly 225 seconds each; concurrent timing is not guaranteed.
The internal deadline is 1,800 seconds, external deadline 2,100 seconds plus
30 seconds termination grace, with one thread, 8 GiB RSS and 16 GiB external
virtual memory per process. The uncompressed working envelope is five times
28,627,392 bytes, plus one failed-return reservation of 2,473,984 bytes and a
small event margin. Raw output is capped at 160 MiB including an 8 MiB
finalization reserve; archive, transfer and source-upload limits are 224 MiB,
320 MiB and 8 MiB respectively per process. No compression assumption is
required for admission. Actual size, RSS, wall time and calls are checked
throughout. Remote originals remain until mandatory return and hash checks.

`tools/run_public_control_tolerance.py` accepts the same input path and hash
arguments as `tools/run_public_batch_writes.py`, plus required
`--requested-bit 0` or `--requested-bit 1` and optional
`--profile axis-stencil` or `--profile negative-axis-refinement` (defined in
the dated adaptation below). The profile is bound in design, result, source
arguments and every planned group, stage, row and call; schedule guards reject
cross-profile substitutions, even where nominal controls coincide. The
historical function defaults still select the original profile. Older outputs
without profile tags remain original-profile evidence and are not relabeled.
`--preflight-only` loads and
validates the actual reference and historical files, produces
`input-preflight.json`, and makes zero model calls. It neither loads weights
nor proves the model backend. Runtime retains independent input qualification,
model file hashes and effective-precision instrumentation. Preflight, source
commit, run card, measured server capacity, transfer reservation and automatic
data return must be established before numerical launch.

Interpret every row and retain failures. Even all twenty passes would concern
only two reference phases and five discrete settings per bit. It would not
bound an intervening parameter region, test joint or diagonal perturbations,
establish a radius of tolerable error or all-phase reliability,
language semantics or useful communication, or show any advantage specific to
the dynamical substrate over an ordinary register. Independent raw-data audit
precedes empirical claims.

After mandatory verified return, run the independent saved-output checker for
each bit with the actual launch commit:

```bash
uv run --no-sync python tools/audit_public_control_tolerance.py \
  --run /path/to/returned/run --commit ACTUAL_SOURCE_COMMIT \
  --requested-bit 0 --output /path/to/new-independent-review.json
```

Use `--requested-bit 1` for the second process and explicitly pass
`--profile negative-axis-refinement` when auditing that profile. The default
audit profile is `axis-stencil`; an inconsistent profile is rejected. The
checker imports neither the
producer nor its model backend. It reconstructs every saved trajectory,
recomputes readouts and diagnostic gates, checks resets and carried boundaries,
verifies source/input/return bindings, and retains failed and incomplete slots.
Its diagnostic-gated outcome summary is separate from the producer's raw label
counts. This is a saved-data audit, not a new model replay or mathematical proof.

The independent audit also verifies every completed collector shard: exact
archive members, marker inventory, extracted payload, and manifest receipt.
Incomplete transfer artifacts can be retained only with a recorded delivery
failure and a separately checked full final return; they are never used as
scientific evidence.

For the original axis-stencil profile, after both audits pass, the committed plotting tool generates a discrete
outcome matrix and provenance receipt:

```bash
uv run --no-sync python tools/plot_public_control_tolerance.py \
  --write0-audit /path/to/bit0-independent-review.json \
  --write1-audit /path/to/bit1-independent-review.json \
  --output-dir /path/to/new-figure-directory
```

It rechecks bound data and preserves all twenty planned outcomes without
interpolating a parameter region. Synthetic layout checks are not results and
must not be inserted into the manuscript as empirical figures.

## Adaptation recorded 2026-10-26T19:04:49Z

The independently audited axis-stencil pilot recovered 18 of 20 requested
labels. The two failures were qualified opposite labels with valid numerical
diagnostics: WRITE0 from td phase zero at endpoint delta log10 temperature
-1e-4 returned td (bit 1), and WRITE1 from ad phase zero at endpoint delta alpha
-1e-5 returned ad (bit 0). All four nominal cases passed; no case was unresolved
or diagnostic-invalid. The source evidence is
[the pilot receipt](receipts/public-control-tolerance.json), including both
full audits and their raw-data bindings. These are discrete finite-precision
observations, not estimates of a general error rate or language accuracy.

The adapted `negative-axis-refinement` profile measures five predetermined
fractions of each bit's failed offset, in this fixed order:

| Group ID | Group name | Fraction | WRITE0 endpoint delta log10 temperature | WRITE1 endpoint delta alpha |
| --- | --- | --- | --- | --- |
| 0 | `fraction-0` | 0 | 0 | 0 |
| 1 | `fraction-025` | 1/4 | -0.000025 | -0.0000025 |
| 2 | `fraction-050` | 1/2 | -0.00005 | -0.000005 |
| 3 | `fraction-075` | 3/4 | -0.000075 | -0.0000075 |
| 4 | `fraction-100` | 1 | -0.0001 | -0.00001 |

WRITE0 keeps endpoint alpha nominal; WRITE1 keeps endpoint log10 temperature
nominal. Fraction zero is a fresh nominal control. Fraction one repeats the
previously failed control setting as a reproducibility check, for both initial
states. Its outcome is measured, not required: unexpected success, failure,
unresolved readout or invalid diagnostics are all retained. Only fraction zero
is compared with the saved nominal scalar trajectories. No intermediate or
full-offset trajectory is compared with a nominal trajectory as though their
controls were identical. No result is required to be monotone in fraction.

Each group resets both batched rows to the exact same ad0 and td0 bytes, with
120 endpoint and 240 unchanged-hold updates. The public nominal command is
unchanged and logged separately from the realized engineering control.
Readout remains independent of the requested bit. All twenty planned rows,
including failures and incomplete slots, are retained. Five points preserve
the tested execution and return envelope while adding three interior samples;
they do not locate a threshold or certify the intervals between them.

Per-process caps, precision gates, immutable return markers, call accounting
and error retention are unchanged: 1,830 calls, one thread, 1,800 seconds
internal, 2,100 seconds external plus 30 seconds grace, 8 GiB RSS, 16 GiB
external virtual memory, 160 MiB raw output including finalization, 224 MiB
archive, 320 MiB transfer reservation and 8 MiB source upload. Paired processes
require measured headroom and verified automatic return before launch. Actual
input preflight, source commit and run cards precede execution; this amendment
alone is not a claim that the new run has occurred.

This is an adaptation selected after seeing the original failures. It is not
a fresh confirmatory test of a pre-existing headline claim. Even five successes
per initial state would establish only those sampled controls, with no safe
region, diagonal or joint tolerance, all-phase robustness, language recovery
or covert-channel claim. The original stencil figure remains evidence for its
original profile; refinement results require their own audited presentation.

### Refinement figure procedure

Use the dedicated refinement plotter only after both complete independent
audits pass with the explicit `negative-axis-refinement` profile. Commit the
plotter and its shared provenance helper before generating an empirical figure:

```bash
uv run --no-sync python tools/plot_public_control_refinement.py \
  --write0-audit /path/to/bit0-independent-review.json \
  --write1-audit /path/to/bit1-independent-review.json \
  --output-dir /path/to/new-refinement-figure-directory
```

The tool rehashes both reports and their bound raw evidence, requires all
twenty planned rows, verifies the fixed fractions and per-bit endpoint controls,
and writes `public-control-refinement.pdf`, `.png` and `.receipt.json` to a new
directory. Wrong, unresolved, diagnostic-invalid and unexecuted cases remain
visible alongside passes. The receipt binds generator and auditor source,
inputs, plotted values and output hashes. Inspect the rendered figure before
manuscript use. Synthetic layout checks are explicitly marked tests; they are
not empirical figures. No interpolation, monotonicity, threshold or continuous
safe-region inference is added by this presentation.
