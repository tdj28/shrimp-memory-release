# Isolated public writes with a half-duration pulse

This adapted exploratory pilot asks whether halving the endpoint control pulse
preserves two selected switches while the common hold remains unchanged. Earlier
120-pulse/240-hold trials succeeded, and a separate carried-state test shortened
only its first hold. Neither result establishes that a shorter pulse works.
The choice of 60 is one half-dose challenge, not a fitted optimum or a safe bound.

## Fixed public profile and inputs

Execute exactly two isolated trials, in this order:

| Trial | Saved initial state | Literal public message | Endpoint updates | Hold updates | Historical control |
|---|---|---|---:|---:|---|
| 0 | ad phase 0 | `Write bit 1 to register R0.` | 60 | 240 | p6-writes trial 2 |
| 1 | td phase 0 | `Write bit 0 to register R0.` | 60 | 240 | p6-writes trial 1 |

The exact public parser still returns the canonical 120-pulse/240-hold action.
The separate immutable `public-isolated-half-pulse60-hold240-v1` action records
that canonical action and its explicitly declared 60-pulse/240-hold stages.
Both typed actions are checked again before dispatch. No historical module,
public contract, global constant or expected label is changed or injected into
the numerical receiver. Endpoint controls come from the parsed public request.

Both initial states are frozen before model loading. After trial 0, one fixed
reset uses the saved td phase-zero bytes for trial 1. It is not a continuation
of trial 0. Reset bookkeeping marks the mutation before saving its after-state,
so a failed artifact write cannot hide an executed reset. Wrong or unresolved
but finite readouts do not select, replace, extend or prevent the second trial.

The existing reference/recovery qualifier checks its prerequisites. Historical
controls are the independently audited run at
`audits/p6-writes/20261025T222347Z-writes/run`, source
`f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4`. The committed receipt binds its raw
results and metadata. Each selected endpoint and hold array is rehashed, its
controls and initial/boundary bytes checked, and its terminal qualification
recomputed. Model revision, weights, configuration, vector, prefix, reference
and recovery hashes must agree. The unchanged mathematical map, hold and
recovery source hashes are compared with historical metadata; both historical
and current orchestration sources are recorded rather than assumed identical.
The two historical first-60 prefixes are frozen locally before any model call.

After each new endpoint stage, all 61 states (including the initial state) and
60 argmax tokens are compared byte-for-byte with its historical prefix. A
disagreement is an implementation/comparability failure, preserved with its
coordinate gap, and aborts later trials. It is not a scientific wrong-state
outcome. This comparison happens before admitting the hold: a mismatch stops
after 63 calls, retaining the completed endpoint and not-started hold and next
trial. It never substitutes historical states or changes the pulse duration. The actual state after update
60 receives all 240 new hold updates. Historical timings are descriptive saved
controls, not contemporaneous replications or a causal runtime comparison.

## Numerical gates and resource bounds

Reuse the validated `run_public_command_memory.Backend`, stage archive,
cache checks and neutral terminal readout. Use one CPU thread, native float64
normalization/rotary instrumentation, the pinned offline Qwen model and unchanged
parameters. Each of four stages rebuilds the prefix once and performs cached
and full-prefix comparisons from the same state, with maximum coordinate gap
at most 1e-10. Diagnostic calls do not advance the numerical state. Every
carried output must be finite float64 with the established shapes. Preserve
invalid returned arrays and carry only the last validated state.

The terminal readout uses all-phase lag-six closure at 1e-8, excludes proper
divisors 1, 2 and 3 at separation above 1e-4, and requires a unique cyclic
reference match within 1e-6. A close reference alone cannot override failed
closure. An independent evaluator compares the neutral labels with the two
literal requests; neither requested correctness nor historical desired labels
feed the map or readout.

The fixed schedule contains 600 updates, four prefix forwards and eight
comparison forwards: exactly **612 calls**, 306 per trial. The corresponding
full schedules prescribe 732 calls. Their 120-call difference is a conditional
16.39% reduction for these two switches, not delivered success before auditing.
Enforce 600 seconds internally, 780 seconds externally plus 30 seconds grace,
8 GiB RSS and 16 GiB external virtual memory. No extra model call, extension,
pulse scan or retry is allowed.

The raw limit is 32 MiB total: 30 MiB working plus 2 MiB finalization reserve.
An explicit pre-model envelope accounts for duplicate trajectory/checkpoint
states, token/timing arrays, eight vocabulary diagnostic arrays, historical
prefixes, reference/state files, 80 container-header allowances and 4 MiB of
metadata/events. It totals 24,448,704 bytes, below the working limit. The late
partial-hold allowance is 2,001,664 bytes, below the finalization reserve.
These are admission estimates, not permission to exceed actual limits.
Before each update or comparison, check working headroom for one returned
native-shape diagnostic/invalid output plus its event. Bad outputs use working
space, leaving reserve for the validated partial trajectory and all final rows.
Actual serialized writes are checked individually. Abort on a cap failure and
retain every available artifact. The launcher must cap the return archive at
48 MiB and reserve 64 MiB total traffic, including source upload and retrieval,
under the fresh shared ledger before admission. All evidence returns locally.

## Files, failures and claim boundary

Entry point: `tools/run_public_half_pulse.py`. The CLI uses the established
seven path and three hash options, plus `--historical-run` for the historical
run directory. It writes the following actual file contract:

- `design.json`, `messages.json`, `planned-messages.json`, `source.json`,
  `qualified-inputs.json`, `historical-inputs.json`, `historical-prefixes.npz`,
  `fixed-inputs.npz`, `trial-initials.json`, backend/model records, events,
  `digital-control.json`, `result.json` and `manifest.json`;
- `message-00.json` and `message-01.json`, also present in `result.rows`, and
  `trial-00-initial.npy`/`trial-01-initial.npy` frozen before model loading;
- attempted `message-NN-before.npy`/`message-NN-after.npy`, the fixed reset's
  before/after arrays, and `terminal-state.npy` when a state is available;
- unchanged `trial-NN-STAGE-block-KKK.npz`, `-final-KKK.npz`, `-cached.npz`
  and `-full.npz` stage files. Blocks checkpoint every 12 updates; final files
  contain the stage initial state and every validated update. Attempted bad
  outputs use `invalid-message-NN-STAGE-KKK.npz`.

Every row retains its status, actual public action/profile, dispatch count,
model-call interval, state hashes, completed and attempted updates, stages,
neutral readout and historical-prefix comparison. On failures both row slots
remain, including not-started trials. The final manifest binds every run file
except itself. Tests use synthetic maps and literal independent expectations;
their results are software checks, not numerical research outcomes.

The first local test attempt preserved one actual-file-contract failure: the
historical run-level status is `passed`, while its row-level statuses are
`completed`. The prerequisite check was corrected to that existing format;
all 41 checks then passed. The original failing stdout is retained alongside
later tests. A separate local `uv` snap confinement failure occurred before
pytest started; tests used the already managed environment without installation
or environment changes. Neither failure started a model or is a scientific null.

A passing pair supports only these two isolated switches at these saved phases
under the declared shorter pulse. Either wrong or unresolved result blocks a
blanket duration reduction without overturning historical full-pulse successes.
Even two successes do not establish carried-state composition, all-phase
reliability, geometry-specific utility, efficient conventional memory, a language
translator, hidden communication or covert computation. This run remains an
adapted finite numerical observation. Source and run card must be committed
before one launch; no new rental or paid review is part of this protocol.
