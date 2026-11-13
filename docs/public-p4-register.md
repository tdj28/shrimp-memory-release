# A second register: two coexisting period-4 cycles in the same model

Written 12 November 2026 (UTC), before any model call. Approved by the human: "Proceed".

## Why

The short paper's register lives in one window (the period-6 kite) of one model, and a reader can ask whether it is a
one-off. The human pointed out that the earlier study found other windows nearby and that a second register "isn't
days of work". Survey (verified at source): the most ready candidate is the period-4 window of the same model, prompt
and steering direction. The aim is the smallest test that shows writes and holds there, so the paper can say the
mechanism is not specific to the kite. Logic and toggles there are out of scope (they would need their own sweep and
plan).

## What is known (earlier study, read-only; checked 12 November)

- Continued branch `neural-shrimp-data/explore/llm-soft-loop/pod64/p4/branch-T-up`: period 4, free parameter
  log10 T at alpha = 0.06484. It uses the same model, layer, precision, prompt, opener, window, vector file, concept and
  alpha scale as the kite branch (argument-by-argument comparison). The philosophy vector is byte-identical to the
  one in this study's instrument input `inputs/p8-reference/vectors.npz`.
- Two folds (multiplier crossing +1): the lower-temperature sheet L (attracting from log10 T = 0.0453) ends at
  0.066394; the upper sheet U (attracting until a flip at 0.06937) begins at 0.063981. Both attract over
  [0.063981, 0.066394], 2.41e-3 wide in log10 T (the kite's overlap is 8.0e-5 in alpha). The middle sheet is
  unstable (spectral radius 1.04 to 1.65).
- At the overlap's midpoint, log10 T = 0.065187: L has spectral radius about 0.45 (0.475 at 0.065409), with a second
  modulus near 0.045. U has 0.155 (at 0.065179), its leading multipliers a complex pair, so deviations from U turn as
  they shrink. The unstable orbit's radius is 1.65 (at 0.065161).
- At one phase, L and U are 0.0123 apart, and the unstable orbit is 0.0040 from L and 0.0090 from U (kite: 0.0154,
  0.0057 and 0.0126).
- Caveats from the earlier study: at this temperature, orbits are "no longer effectively one-dimensional" (second
  multipliers 0.06 to 0.09); T is about 1.16; the region "is not shown to be one shrimp-organized component"; no kite
  was located here. None of this stops a register, but the paper must say it.

## Design

- **Controls.** Hold H4 = (alpha 0.06484, log10 T 0.065187). Write 1 (to U) at log10 T = 0.066394 + delta, where L
  no longer exists. Write 0 (to L) at log10 T = 0.063981 - delta, where U no longer exists. delta = 2.4e-4 and
  6.0e-4 (0.1 and 0.25 of the overlap). alpha is fixed throughout.
- **Stage R, reference cycles** (one group, two lanes, 400 updates at H4, 403 calls). The lanes start from the saved
  branch states nearest H4 on each sheet (L index 29, U index 61). Each lane's last four states are its reference
  cycle.
- **Stage W, writes and holds** (five groups, both lanes starting on the two reference cycles at phase 0):
  - hold only, 240 updates at H4;
  - write 1 at each delta, 120 updates, then 240 at H4;
  - write 0 at each delta, 120 updates, then 240 at H4.
  About 1,710 calls; 120 updates is 30 periods.
- **Readout.** The last four hold states are aligned cyclically to each reference cycle. A lane reads the cycle it is
  within 1e-6 of; if it is within 1e-6 of neither, it is unresolved.
- **Total.** About 2,100 calls, one group per single-thread process on this machine, about 20 minutes of wall time.

## Predictions, fixed before any call

- **R1.** Both lanes settle on period-4 cycles at H4: lag-4 residual below 1e-12, lag-2 residual above 1e-3. The two
  cycles are distinct, more than 5e-3 apart at every phase.
- **W0.** Hold only: identity, each lane on its own cycle.
- **W1.** Write 1: both lanes read U (reset to 1), at both deltas.
- **W2.** Write 0: both lanes read L (reset to 0), at both deltas.
- **Reported, not predicted:** time to settle after each write; distances to the unstable sheet's saved state at the
  matching phase; whether the two cycles share their argmax tokens (compared, not printed).

If R1 fails, stage W does not run and the result is reported. If a write fails at the smaller delta, that is reported
as is; there is no retuning of delta without a new plan.

## Tooling (before any call)

- **New runner** `tools/run_public_p4_register.py`. Model calls go through the existing backend unchanged; inputs are
  pinned by hash, and the vector identity above is checked in preflight.
- **Period-4 readout and reference checks.** These are written fresh, with unit tests on synthetic period-4
  fixtures, because the audited instrument assumes period 6 in its readout gates and reference checks. The period-6
  instrument itself is not edited.
- **Preflight** with no model load. An independent audit script (zero calls) checks manifests, starts, controls and
  recomputed readouts. Score receipt: `docs/receipts/public-p4-register.json`.

## What it would add, and what it cannot show

If W0 to W2 hold, the paper gains one paragraph and a Table III row: a second register, in a period-4 window of the
same model and steering direction, written by excursions past each fold of an S-shaped branch and held for 60
periods. That would show the kite is not the only place the mechanism works. It cannot show robustness to phase or
perturbations, logic or toggles in that window, other models, or that the window is a shrimp in the strict sense.

## Execution, fixed before any call (12 November 2026)

- Exact values: H4 = (0.06484, 0.0651875), the overlap's midpoint. Write 1 at log10 T = 0.066634 and 0.066994. Write 0
  at 0.063741 and 0.063381.
- Runner `tools/run_public_p4_register.py`: the toggle check's instrument with hash-bound design and start files.
  The instrument's own input checks (pinned period-6 references) still run; those references are not used.
- Inputs by `tools/prepare_public_p4_register.py` (zero calls):
  - `stage-r` checks the branch settings against the kite branch's and that indices 29 and 61 are the attracting
    points nearest H4;
  - `stage-w` tests R1 on stage R's saved path and writes the references (phase j = the last state at an update
    congruent to j mod 4), stage W's design and its starts (the phase-0 points of both cycles).
- Readout `tools/p4_readout.py`, unit tests `tests/test_p4_readout.py` (5 passed). Score by
  `tools/analyze_public_p4_register.py` (zero calls).
- Run directory `audits/public-p4-register/20261112T034957Z-local`; stage R's preflight passed (403 planned calls).

## Results

**Stage R** (`audits/public-p4-register/20261112T034957Z-local/R`, 403 calls, 1,237 s, cache comparisons passed;
check by `tools/prepare_public_p4_register.py stage-w`, report `inputs-W/references-report.json`).

- **R1 holds.** Both lanes settled at H4 on period-4 cycles. The lag-4 residuals were 4.0e-15 and 4.1e-15. The lag-2
  residuals were 0.097 and 0.142, so the primitive period is 4.
- **The cycles are distinct.** At matched phases they are 0.0122, 0.0098, 0.0378 and 0.0589 apart; the first, at
  the branch's phase, matches the earlier study's 0.0123. At their best cyclic alignment they are at most 0.0589
  apart.
- **Unlike the kite, the two cycles do not share all their argmax tokens.** They agree at 3 of the 4 phases and
  differ at one, and they are not one sequence shifted in phase. Tokens not printed. So in this window the bit shows
  in the argmax text once per period.
- References: `inputs-W/references-p4.npz` (SHA-256 e6c1ffd6...). Stage W's design and starts are bound by hash in
  the same report.

**Stage W** (`.../W`, five single-thread shards, 1,707 calls, 907 to 1,373 s, cache comparisons passed). Score:
`docs/receipts/public-p4-register.json` (`tools/analyze_public_p4_register.py`, zero calls; audit passed: manifests,
starts, controls, stage lengths and carries).

- **W0 holds.** Hold only: each lane stays on its own cycle (L within 1.6e-15, U within 5.1e-15, from update 0).
- **W1 holds.** Write 1 at log10 T 0.066634 and 0.066994: both lanes read U, settled from hold update 25. At the end
  of the pulse both lanes are 0.0063 and 0.0070 from U's cycle at H4 and 0.016 from L's; the two lanes agree to
  1e-9.
- **W2 holds.** Write 0 at 0.063741 and 0.063381: both lanes read L, settled from hold update 39. At the end of the
  pulse they are 0.0019 and 0.0024 from L's cycle.
- **Phase.** Every lane ends at the phase its update count gives (cyclic shift 1 for the last four states, updates
  237 to 240 of the hold), so the writes keep the period-4 clock.

**Total:** 2,110 calls. Every prediction held. The paper gains one paragraph ("A second window", end of Sec. III), a
scope change in the discussion and a Table III row.
