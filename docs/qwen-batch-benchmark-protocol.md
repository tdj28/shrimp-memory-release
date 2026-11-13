# Two-state Qwen batch benchmark

This exploratory one-step test asks whether a batch of two states costs less
than two separate forwards while preserving the saved numerical outputs to a
fixed tolerance. It does not execute a trajectory or establish equivalent
attractors, general float64 accuracy, communication reliability, or a speedup
for a complete workload. No benchmark outcomes are recorded in this protocol.

Use the existing pinned Qwen2.5-0.5B-Instruct snapshot, philosophy steering at
layer twelve, fixed prefix, CPU native float64 normalization and rotary code,
math SDPA and one thread. Input is the exact-readback packet with SHA-256
`8fc36ac62413a099fd6dd562f1165dfad92528de3e7011a3a072ec4f93e7b8b1`;
use indices zero and two, in that order, without modifying or evolving them.
The CLI also requires this expected packet hash and rejects a different source.
Controls are alpha `0.07293697809562831` and log10 temperature
`-0.15395151820007266`. The states were observed previously; this is a bounded
engineering comparison on two known inputs, not fresh dynamical evidence.

The fixed schedule is twenty-eight model calls:

1. One scalar prefix preparation.
2. Scalar cached and full-prefix forwards for each state: four calls.
3. Cached and full-prefix batch-of-two forwards: two calls.
4. Two scalar warmups followed by one batch warmup: three calls.
5. Six timing comparisons, each two scalar cached forwards and one batch cached
   forward: eighteen calls. Scalar pair first on even repeats, batch first on
   odd repeats; the scalar pair always retains input order zero then two.

The scalar implementation is the existing `FeedbackMap.forward`. The batched
implementation copies the scalar prefix-cache container and each layer, then
allocates new repeated key/value tensors along the batch dimension. Its source
cache remains scalar. Steering broadcasts identically to each batch row. Full
batch validation repeats the full fixed prefix and appends each state. Every
forward must retain finite float64 inputs, outputs and probabilities, unchanged
input and source-prefix tensors and lengths, and native64 normalization/rotary
observations. For cached batch calls, retain the actual repeated prefix tensors
referenced before the forward and verify the used cache advances one token.

Compare output coordinates and probabilities with a maximum absolute gap of
`1e-12`, inclusive, and require top-token agreement. Record exact-equality flags
separately. Apply these gates to scalar cached/full, batch cached/full,
scalar/batch, each returned call against its scalar diagnostic baseline, and
each timed pair. Preserve every comparison. Finite numerical disagreement does
not truncate the fixed schedule. Invalid arrays, mutated caches, missing inputs,
resource breaches and interrupts stop execution with all attempted artifacts
and explicit unstarted calls retained. No retries, new states or tolerance
changes are part of this design.

Timing compares the sum of the two scalar forwards with one batch forward.
It excludes input-tensor preparation, cache audit snapshots, hashes and file I/O;
it includes normal cache handling, transformer execution, softmax and embedding
feedback. Capture lightweight references to batch-prefix tensors inside the
timer, then copy audit snapshots afterward. Record every duration and all six
orders; compute per-pair scalar/batch ratios and their median, true minimum and
maximum, plus the ratio of summed durations, only when every numerical and
integrity gate passes. Otherwise timing eligibility is false and the ratio
summary is absent, while raw timings remain. Model loading, warmup, diagnostics,
data return and the numerical observer are outside the timed-forward comparison
and must not be silently counted as accelerated work.

Bounds are twenty-eight calls, 180 internal seconds, an external 240-second
timeout with thirty seconds of termination grace, eight GiB peak RSS, sixteen
GiB virtual memory and one thread. Keep raw run output below 255 MiB within a
512 MiB transfer reservation. Launch only after measured host headroom and at
most one other single-thread model process. Record that other process and host
conditions in the run card: alternating order reduces one order confound but
does not eliminate shared-host load, cache, warmup or burstable-CPU effects.
Python signal handling can be delayed inside native calls, so the external
watchdog remains required. Ignore additional termination signals while flushing
failed evidence; restore original handlers on exit.

`tools/benchmark_qwen_batch.py` writes `inputs.json`, `metadata.json`,
`prefix-cache.npz`, `events.jsonl`, immutable `call-NNN-LABEL.npz` and matching
JSON for each attempted nonprefix call, six `pair-NN.json` files on full
completion, and `results.json`. Every call records its input, output,
probabilities, before/after source-cache arrays and dtype observations; cached
batch artifacts additionally bind the repeated prefix arrays. The summary lists
all twenty-seven planned nonprefix calls, including unstarted calls, plus prefix
status and total call count. Source, packet, model, config and vector hashes and
the exact prefix IDs travel with the run. All raw artifacts, failed attempts and
logs return to `/data2/shrimp-memory-data`; source and run card precede launch.
An independent saved-array audit must qualify every row before reporting a
timing result. That audit does not independently recompute model inference.
