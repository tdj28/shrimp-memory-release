# Transverse robustness and a frozen static decoder

The completed result is bound by [its receipt](receipts/transverse-readout.json).
All eight intended labels were retained, and the frozen nearest-neighbor
library also predicted all eight; the nearest-reference baseline predicted six.

This adapted exploratory experiment asks whether two selected basin encodings
remain usable off the previously studied line, and whether a cheap static
lookup predicts their readouts. The design below was recorded before this
experiment; its historical context and limitations remain unchanged. The earlier coarse, transition-thirds, nested-thirds and
fresh-line outcomes informed this design. The earlier 120-update recovery
failures remain unresolved under their original horizon. GBZ1 has no established
second usable state and is not the system tested here.

## Fixed inputs

Use the pinned period-six reference archive with SHA-256
`d96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0`.
Let `a=ad[0]`, `b=td[3]`, and `x(w)=(1.0-w)*a+w*b`, evaluated in that
float64 order. The four anchors, in order, are `x(7/27)`, `x(8/27)`, `a`, `b`.
Their previously qualified labels are respectively ad, td, ad, td; verify the
interior anchors against the hash-bound prior inputs and qualified outcomes.
The first two anchors are the interior stratum; the last two are matched
near-orbit multistability controls.

Draw `z=Generator(PCG64(2026102603)).normal(size=896)` once. Compute
`d0=b-a`, `d=d0/norm(d0)`, `z_perp=z-dot(z,d)*d`, then
`q=z_perp/norm(z_perp)`. Use
`epsilon=0.001*sqrt(mean(sum(refs*refs,axis=-1)))`, the original recovery
amplitude rule. For each anchor emit its minus then plus perturbation,
`anchor+sign*epsilon*q`, retaining this eight-row order without filtering,
redraws, replacement or outcome-dependent redesign.

The preparation gates, fixed before generating this packet, require finite
float64 inputs, nonzero line and transverse norms, `abs(dot(q,d))<=1e-12`,
`abs(norm(q)-1)<=1e-12`, relative displacement-norm error at most `1e-10`,
and absolute displacement projection on `d`, divided by epsilon, at most
`1e-10`. Require eight distinct resulting vectors and no exact equality to
any of the 32 historical packet rows or 12 reference phases. Failure is a
preparation failure, not a reason to choose another seed or amplitude.

## Historical data and static controls

The preparer requires the four completed audits bound by the committed receipts
`p6-phase-mixed-long.json`, `p6-transition-thirds.json`,
`p6-nested-thirds.json` and `fresh-readout.json`. It pins their audit hashes,
rehashes every bound file, checks their eight-row/full-600/common-hold context,
and checks packet starts against the saved trajectories. It recomputes the
terminal period-six/reference gates. A row whose outcome is unresolved is
retained in provenance but cannot label a library exemplar. A failed audit,
corrupt file or inconsistent claimed qualification stops preparation.

Build the library in this source order and original packet order, followed by
all ad phases zero through five and all td phases zero through five. Deduplicate
numerically identical float64 vectors, retaining every origin. Conflicting
labels for a duplicate fail preparation. Derive and record the actual unique
count; do not assume the anticipated count of forty-two. Preserve excluded
unresolved origins. Reference phases have the labels of their pinned cycles.

Freeze two predictions for every new input before model outcomes exist:

- Full-state one-nearest-neighbor over the deduplicated labelled library,
  using float64 Euclidean distance. Any exact nearest-exemplar distance tie
  erases the prediction, including a tie between examples with the same label.
- Nearest reference phase: minimize Euclidean distance over six phases for
  each label, then choose the smaller label distance. An exact label-distance
  tie erases the prediction. No distance rejection threshold is added.

Record all distances, tied indices and resulting labels. No classifier fitting,
hyperparameter search or refitting on the new outcomes is permitted. A cheap
classifier may learn earlier nonmonotone labels; the nearest-reference failure
alone is not evidence that dynamics are required for decoding.

## Files and execution

The supplied output directory must be new and outside the source repository.
`packet.npz` contains only the eight `states`, shape `(8,896)`, float64.
`geometry.npz` retains the Gaussian draw, line unit vector, transverse vector,
anchors and scalar amplitude. `library.npz` retains states and integer labels
(zero ad, one td). `design.json` contains all bindings, historical origins,
geometry checks, intended anchor labels, fixed predictions and source identity.
`library-origins.json` and `geometry-checks.json` also preserve those intermediate
records before final qualification.
`preparation.json` binds the completed preparation with zero model calls.
Every preparation failure after claiming a new valid output directory writes
an explicit failed receipt and retains all already generated artifacts. A
present packet alone is not permission to run: require a prepared receipt and
the independent preparation audit. Existing output directories are never reused.
The preparer takes explicit `--references`, `--coarse-audit`, `--thirds-audit`,
`--nested-audit`, `--fresh-audit` and `--output-dir` paths.
The receiver receives only `packet.npz` and separately supplied reference cycles;
geometry, library, intended labels and design stay with the local evaluator.
This is an information-flow contract, not operating-system isolation.

Use the unchanged scalar `run_p6_readback.py --steps 600 --shadow-references`
at alpha `0.07293697809562831`, log10 temperature `-0.15395151820007266`.
Retain every state, token, diagnostic and twelve-update checkpoint. All rows
run all 600 updates; the observer never terminates execution early. Bounds are
4817 model calls (one prefix, sixteen cache checks, 4800 updates), 2700 seconds
internal, 2760 seconds external plus thirty seconds termination grace, one CPU
thread, 8 GiB RSS, 16 GiB virtual memory and a 512 MiB transfer reservation.
Use a free standing-host slot, at most one other one-thread model process, and
fresh headroom/ledger checks. Preparation does not launch that job. Commit
tested source and record the run card before actual preparation and execution.

## Evaluation and limits

Independently audit every saved row, source binding and returned raw artifact.
Report intended-anchor return, qualified switch and unresolved outcome with the
four-row denominator for each stratum. Separately compare both static predictors
with qualified final labels, retaining prediction erasures and unresolved final
states. Compare the first three-consecutive-checkpoint decision with the full
600-update result, including later unresolved or different-label checkpoints.
Report all executed calls, elapsed/forward/observer time and hypothetical unused
updates; this run actually skips none. Static-prediction timing, if measured,
must disclose its hardware and setup scope.

The useful nulls include fragile interior encodings despite returning near-orbit
controls, and successful inexpensive static decoding. Opposite outcomes within
one transverse plus/minus pair would show that its shared line projection is
insufficient on those two states. No outcome here establishes a basin volume,
fractal boundary, general precision guarantee, population reliability, shrimp
specificity, independent register bank, secrecy or ordinary-language channel.
The controls test local ordinary attraction in the same map, not a different
non-shrimp system. Preserve every failure and do not extend unresolved rows.

## 2026-10-26T07:37:21Z: disclosed single-direction follow-up

After seeing the completed cohorts at relative amplitudes 0.001, 0.003 and
0.01 for seed 2026102603, test exactly one additional direction at amplitude
0.01 with seed 2026102604. This is an adapted exploratory follow-up to assess
whether the observed retention and static-prediction outcomes depend on the
one direction studied so far. It is not fresh confirmatory evidence, a seed
search or a new amplitude sweep. The two directions provide two finite probes,
not an estimate over a population of directions.

Prepare with `--direction-seed 2026102604 --relative-epsilon 0.01`. The first
PCG64 Gaussian draw, projection, normalization and perturbation arithmetic are
unchanged except for that seed. Preserve all four anchors, both signs, original
row order, geometry/novelty gates, reference cycles and common-hold controls.
The original seed remains the default and preserves the three existing packet
constructions bitwise; only those two declared seeds are accepted. The new
seed is accepted only at relative amplitude 0.01. The independent auditor
reconstructs from the declared seed and rejects an unsupported seed, a changed
seed declaration on unchanged packet bytes, or an incompatible amplitude.
No redraw or replacement is allowed after a failed preparation.

Keep the same historical library: four original eight-row cohorts followed by
twelve reference phases, yielding the existing forty-two unique labelled
examples after exact deduplication. Do not add outcomes from any of the three
transverse-amplitude cohorts or this new direction. Freeze both static
predictions before inference. Before launch, compare all eight new states with
all eight states from each of the three prior transverse packets (192 exact
pairwise checks), in addition to the historical and reference checks performed
by the preparer. Record all prior packet hashes and fail on any coincidence.

Use the unchanged scalar `run_p6_readback.py --steps 600 --shadow-references`.
Retain all eight rows through 600 updates, including wrong, unresolved and
interrupted outcomes, with 4,817 model calls and the original time, precision,
RSS, virtual-memory and one-thread bounds. Shadow decisions do not stop the
runner. Report intended-label retention, qualified switches, unresolved
finals, both frozen static-predictor results, first decisions, later sampled
losses/switches and runtime separately for the four interior probes and four
near-orbit controls. Compare the matching anchor/sign rows across the two
0.01-amplitude directions without selecting a favorable subset. Changed
outcomes would show direction dependence at those points; agreement would
extend the finite observations. Neither result establishes basin volume,
fractal geometry, a robust neighborhood or general static-decoder performance.

For this new seed only, reserve 384 MiB of transfer allowance; old-seed designs
retain their original 512 MiB reservation. The prior 0.003 cohort's immutable
`final-manifest.json` at
`/data2/shrimp-memory-data/audits/transverse-intermediate/20261026T064110Z-transverse-intermediate/`
binds 848 files totaling 99,624,556 bytes; all files were rehashed during this
planning check. Its six shard archives plus final archive and seven manifests
total 101,063,177 bytes, excluding source upload and transport traffic.

The unchanged writer's arrays fit a compression-independent allowance of
94,961,664 bytes: 400 readback blocks at 96 KiB each, eight final trajectories
at 4.25 MiB each, sixteen cache/full probability archives at 1.1875 MiB each,
and one packet at 64 KiB. This accounts for overlapping block endpoints,
complete final trajectories, diagnostic probabilities, scalar fields and
partial/failure arrays. No evidence is omitted to meet the estimate.

Budget 160 MiB per complete transfer pass: 91 MiB for arrays, 32 MiB for JSON
sidecars/results, 16 MiB for console/events, 8 MiB for source/input traffic in
both directions, 4 MiB for tar packaging/manifests, and 9 MiB for protocol and
control traffic. Two complete passes, including one full transfer retry, use
320 MiB, leaving 64 MiB contingency within the 384 MiB reservation. This is a
conservative planning allowance; JSON, logs, exception text and transport do
not have writer-enforced byte caps, so 256 MiB is not accepted as a hard bound.
Monitor actual traffic and artifact bytes, account for concurrent reservations,
and reassess before exceeding the reservation. Preserve all failed and partial
files and reserve their verified local return. No model weights are uploaded.
The hash checks, source bindings and arithmetic are saved in
[the transfer calculation](transverse-direction-transfer-budget.json).
Require the launcher to monitor total live output against 128 MiB for its owned
process and check the final manifest before copying. Retain the existing
255 MiB raw-return and 256 MiB return-archive caps. The retry margin does not
enable automated bulk retries; each transfer still requires current accounting
and sufficient reserved headroom.

The output monitor polls raw regular-file sizes every five seconds, excluding
return-shard packaging and the final archive/manifest. A 128 MiB observation
triggers termination of the exact owned model through a retained Linux pidfd;
this is a sampled guard, not a claim that overshoot is impossible. The wrapper
uses a supervisor that owns both the guard and the bounded timeout process.
It starts inference only after a valid readiness record, stops the owned
command if the guard exits unexpectedly, and waits for child cleanup before
writing completion markers. Signals use retained process handles. Ordinary
completion preserves the model command's status; supervision failures are
explicit orchestration failures, with all partial evidence retained.

The completion worker executes frozen operator helpers and independently
checks the announced archive size before copying it. It fetches and verifies
the manifest first, checks the raw payload allowance, then transfers and
verifies the archive and complete inventory. It closes the shared reservation
only after local return succeeds. Synthetic tests cover absent readiness,
monitor loss, process identity, failed setup, nonzero exits, and oversized
archives without running model inference.
