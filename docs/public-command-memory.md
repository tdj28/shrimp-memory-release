# Public command dispatch into the period-six numerical memory

This is one adapted exploratory integration attempt, following the language
contract consultation and its separate parser and retrospective-compatibility
prerequisites. It tests actual message-driven numerical execution. Earlier
language results, including the strict two-request null and both twelve-request
panels, remain unchanged. In the scored-response panel, grammar was imposed by
construction; its five correct valid commands did not establish reliable
interpretation, and two invalid commands were accepted. This experiment uses
the existing exact public parser, without another semantic decoder adaptation.

## Fixed message stream

From the qualified ad reference cycle, phase zero, send exactly these messages:

1. `Write bit 0 to register R0.`
2. `Write bit 2 to register R0.`
3. `Write bit 1 to register R0.`
4. `Write bit 0 to register R0. Then erase the latch.`
5. `Write bit 1 to register R0.`
6. `Do not write bit 0 to register R0.`
7. `Write bit 0 to register R0.`

The receiver accepts only the existing two complete strings. It does not trim,
normalize, repair, match a prefix or substitute a default. Unsupported bits,
trailing commands and negation must be rejected. The evaluator's expected
accepted values are separately authored; producer message inputs contain no
expected-bit field. Every numerical control and duration is supplied by the
parser's typed action, then bound again to the full input before dispatch. The
numerical backend does not receive a separate desired label or a gold state.

The four accepted requests prescribe 0, 1, 1, 0. They exercise retention and
switching only to the extent preceding writes actually reached their requested
states. They are one carried-state chain, not four independent reliability
trials or four simultaneously stored cells.

## Map, control and state contracts

Reuse the pinned Qwen2.5-0.5B-Instruct soft-feedback map, philosophy vector,
fixed prefix, one CPU thread and existing native float64 normalization and
rotary patches. Recheck weights, configuration, vector and qualified reference
hashes before loading. Reference qualification rechecks the source hold
trajectories and the two unperturbed recovery controls. The committed pulse
geometry must match the existing map implementation.

The exact command contract supplies the established 120-update endpoint pulse
and 240-update common hold. See
[the transparent command protocol](transparent-command-protocol.md) for public
coordinates. No parameter, duration or threshold is selected from this run's
outcomes. Each stage rebuilds its prefix cache and saves cached/full-prefix
one-step outputs at the same initial state. Require their maximum coordinate
gap at most 1e-10. Diagnostic outputs never advance the carried state.

Only successful, finite float64 update outputs advance the state. Save every
available attempted output on a failed numerical gate. Save the state before
and after each attempted message, and retain all seven message slots even if
loading or execution stops early. Rejected input requires zero dispatcher
calls, zero model calls and byte-for-byte identical state. Its before/after
arrays and raw-array SHA-256 hashes permit an independent check including
signed-zero differences. No rejected input initiates a model load, cache
preparation or numerical readout; the model may already be resident.

The producer records a neutral readout of each complete hold: all-phase
lag-six closure at tolerance 1e-8, exclusion of proper divisors 1, 2 and 3 by
maximum phase gap above 1e-4, and the distances from the last six states to
both qualified reference cycles over cyclic alignments. A unique match within
1e-6, conditional on the cycle gate, supplies a qualified label. The independent
saved-array evaluator compares that label and actual controls with the visible
command. Readout requires no model forwards. A wrong or unresolved label does
not cause an extension, reset or replacement; the actual finite state continues.

## Bounded execution and stopping

Four accepted actions each contain two stages. Each stage uses one prefix
preparation, two comparison forwards, and its prescribed number of updates:

- 1,440 numerical-update forwards;
- eight prefix forwards;
- sixteen cache/full-prefix comparison forwards;
- 1,464 forwards total, with no calls for the three rejected messages.

Derive and check this envelope from the parsed public actions before inference.
Enforce per-stage and global category caps, 840 seconds internally, 900 seconds
externally with 30 seconds termination grace, one thread and 8 GiB peak RSS.
Artifact writes have a 64 MiB working cap and 8 MiB finalization reserve, for
72 MiB total. The launch plan separately limits the returned archive to 96 MiB
and reserves 128 MiB transfer; launch requires measured resource headroom and
the shared transfer ledger. There is no new rental or paid review.

Execute the seven messages once. An execution, precision, cache, nonfinite,
resource or artifact-preservation failure aborts the bounded attempt; retain
all attempted and not-started rows and all available arrays. Wrong or unresolved
readouts are scientific outcomes and do not themselves abort the fixed stream.
No automatic retry, prompt revision, longer hold, phase expansion or larger
panel follows. Historical saved-data compatibility remains a retrospective
endpoint and is not relabelled as this fresh integration run.

## Files, audit and cost

`tools/run_public_command_memory.py` writes `result.json`, `message-00.json`
through `message-06.json`, `messages.json`, `planned-messages.json`, `design.json`,
`source.json`, `qualified-inputs.json`, `fixed-inputs.npz`, a separate
`digital-control.json`, model input/backend records and `events.jsonl`.
Before/after states use `message-NN-before.npy` and `message-NN-after.npy` for
attempted messages. Stage arrays retain the established P6 file contract:
`trial-NN-endpoint-block-KKK.npz`, `trial-NN-endpoint-final-KKK.npz`, and the
corresponding hold files, where NN is the message index and KKK the completed
update count. Blocks checkpoint every twelve updates; each contains the shared
boundary state, absolute first-state index, token IDs, update durations and
actual controls. Final stage files contain the initial state and all validated updates. Attempted
and completed update counts are separate; failed returned outputs remain in
separate bound artifacts and never advance the carried state.
Cached/full comparison files use `trial-NN-STAGE-cached.npz` and `-full.npz`.
`manifest.json` binds every run file except itself. A failed attempted output
may additionally have an `invalid-message-NN-STAGE-KKK.npz` artifact.

Independent evaluation must rehash all files, check the literal seven-message
stream, count every model-call event, audit full controls and durations, confirm
byte-exact continuity across messages and stages, recompute each numerical gate,
and separately report four planned writes and three expected rejections.
Missing or incomplete rows are explicit outcomes, not removed denominators.

The conventional digital latch receives the identical public stream and has
its own measured latency and dispatch counts. Numerical timing includes cold
initialization, each message's frontend handling and total elapsed execution;
stage files record update times and prefix events record prefix times. Audit,
packing and return costs belong to the separate completion record. No frontend
model inference is needed for this finite parser. There is no speed or practical
advantage claim over the conventional latch.

This is a finite-precision exploratory observation, not a proof, general memory
guarantee, unrestricted-language result, hidden channel, audit-evasion technique
or demonstration of a benefit specific to shrimp geometry. Even perfect delivery
would support only these commands, this initial state and this one sequence.

During model-free development, the first regression run reported 19 failed and
82 passed tests. All 19 failures were the same artifact-contract defect: the
in-memory serialized action retained a tuple of stages while a JSON reload
produced a list. The producer now records JSON-native stage lists, and the
final combined regression passed 102 tests. These were synthetic software tests,
not numerical-model attempts or scientific null outcomes. The test handoff
preserves the available original console excerpt and final console output.
