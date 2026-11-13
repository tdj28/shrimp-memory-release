# Write time against the distance of the write endpoint beyond a fold

Plan written 11 November 2026 (01:44 UTC), before any saved state of the endpoint-perturbation pilots was opened
for this purpose. Zero model calls. Evidence class: adapted exploratory, post hoc.

## Why

The branch receipt (`docs/public-register-branch.md`) ended on an interpretation it could not test: that a write is
a slow passage through the place where the starting orbit ended at a fold, so that the time a write needs is set by
how far the write endpoint lies beyond that fold. No run varied that distance on purpose. The endpoint-perturbation
pilots of 26 October did vary it, for another reason: the tolerance grid, its refinement and the matched
continuation wrote each bit at fixed offsets of the endpoint and saved every state. On the two write lines those
offsets put the endpoint at eleven distances from the fold. The saved trajectories can therefore test the
interpretation without running the model.

## History, stated plainly

- The pass or fail outcome of every pilot row has been known since 26 October
  (`docs/receipts/public-control-tolerance.json`, `public-control-refinement.json`,
  `public-control-continuation.json`). Agreement with those outcomes is a consistency check, not a prediction.
- The nominal write of 1 was already seen to cross at about update 80 in the eight-note run (branch receipt). The
  nominal write-1 time is therefore not a blind prediction.
- A first hand estimate of that time used one coefficient for the whole first sheet (the straight-line fit of
  (1 - multiplier)^2 / 4 against alpha) and gave 62 to 66 updates. The companion's branch table then showed that the
  coefficient is not constant: its local value falls by nearly half toward the fold. The estimate was changed to use
  the coefficient at the fold, which gives about 86. That change was made knowing the observed value of about 80.
- Inspected in the pilot archives before this plan: the design files (control values, group names), file names and
  array shapes, and that each run's first start state equals the reference cycles at phase zero. No trajectory
  value of a pilot run was looked at.
- Blind at the time of writing: the passage times of the write of 1 at a quarter of the negative offset and at the
  positive offset; whether the lanes that failed are still creeping or have settled on an orbit; every write-0
  passage time; the fold positions estimated from the dynamics alone.

## What is read

- `predict` step: the companion study's saved continuation (`neural-shrimp-data/.../kite/kite-la-branch/branch.json`,
  read-only), `docs/receipts/public-e01-amended-geometry.json` (the fold solved on the hold-strength line),
  `docs/receipts/public-register-branch.json`, and the `design.json` of four pilot runs. It opens no saved state.
- `measure` step: the saved states of the same four runs
  (`audits/public-control-tolerance/20261026T182840Z-public-control-bit{0,1}/run`,
  `audits/public-control-refinement/20261026T191356Z-public-refinement-bit{0,1}/run`), of the two continuation runs
  (`audits/public-control-continuation/20261026T201011Z-public-continuation-bit{0,1}/run`), and the reference cycles
  of the audited XOR chain.

The script is `tools/analyze_public_write_passage.py`. Every input is bound by size and SHA-256 in the receipt.

## The model being tested

Along its one slow direction, the period-6 return map near a fold is taken as the fold normal form

    y -> y + a * delta + b * y^2      (one step = one period = six updates)

where delta is the distance of the control beyond the fold (negative short of it).

1. Short of the fold by D there are two period-6 orbits, at y = -sqrt(aD/b) (attracting) and +sqrt(aD/b)
   (unstable). The attracting one has multiplier mu with (1 - mu)^2 / 4 = a b D. The product a b is therefore
   readable from the companion's recorded multipliers. It needs no trajectory.
2. Beyond the fold by delta there is no orbit, and the state creeps past y = 0 at a per-period displacement
   a delta + b y^2. Treating the creep as continuous, the time from y0 to y1 is
   [arctan(y1 sqrt(b / (a delta))) - arctan(y0 sqrt(b / (a delta)))] / sqrt(a b delta) periods.
3. A write starts on the hold orbit, y0 = -sqrt(a D_H / b), where D_H is the distance from the hold to the fold.
   - **Full passage** (to far beyond the bottleneck): [pi/2 + arctan(sqrt(D_H / delta))] / sqrt(a b delta) periods.
   - **Reaching the hold's unstable orbit**, the mirror image +sqrt(a D_H / b):
     2 arctan(sqrt(D_H / delta)) / sqrt(a b delta) periods. A state that is beyond the unstable orbit when the pulse
     ends goes to the other cycle during the hold, so this is the predicted minimum write duration.
4. The smallest per-period displacement is a * delta: proportional to the distance beyond the fold, zero at the fold.
   Around that minimum the displacement grows as 1 + a b delta t^2, with t in periods.

For far endpoints both times scale as the inverse square root of the distance beyond the fold.

**Coefficient for the write of 1** (the fold that ends the cycle-0 sheet on the hold-temperature line): least
squares of alpha = alpha_fold + c2 m^2 + c3 m^3 over the recorded branch points with |m| <= 0.25, where
m = multiplier - 1. There is no linear term because the branch turns where the multiplier passes +1. Then
a b = 1 / (4 |c2|). Windows 0.15 and 0.35 are reported as a sensitivity check.

**Write of 0** (the fold that ends the cycle-1 sheet on the hold-strength line): no branch was recorded along the
temperature. The fold position is the earlier Newton solve of the fold equations (log10 T = -0.1536044832612347; a
numerical candidate whose continuation check failed). The coefficient is not known beforehand. Two rough values are
recorded with the predictions, but the analysis of the write of 0 is a straight-line fit of the measured passage
update against the model's regressor, with intercept and slope free.

## Fixed definitions

- **Coordinate.** c = <x - A0, A1 - A0> / |A1 - A0|^2 at the state's own phase, with A0 and A1 the reference cycles
  at the hold: 0 on cycle 0, 1 on cycle 1.
- **Passage update.** The first update from which c stays beyond the level three quarters of the way to the
  requested cycle for six consecutive updates. The same count at one quarter and at one half is reported beside it.
  Three quarters was fixed before any state was opened, for this reason: at the hold the unstable orbit projects at
  0.23 of the way from cycle 0 to cycle 1 (branch receipt). The orbit that ends in a write of 0 is cycle 1 merged
  with that unstable orbit, so its remnant may lie near the half-way level, and a half-way criterion could trigger
  inside the bottleneck. Three quarters of the way to the requested cycle is beyond the remnant for both writes.
- **Reaching the hold's unstable orbit.** The update, sampled at the branch's saved phase and interpolated linearly
  between two samples, at which c first passes 0.2316, the projection of the hold's unstable orbit.
- **Per-period displacement.** s(n) = |x(n + 6) - x(n)| during the 120 endpoint updates.
- **Bottleneck.** At each of the six phases, the smallest s before the passage, refined by a parabola through three
  samples when the minimum is interior.
- A lane **passes within the pulse** if its passage update is at most 120.
- Lanes at the same controls in two runs are both reported; fits use each control value once.

## Predictions, recorded before any state was opened

File: `docs/receipts/public-write-passage-prediction.json`, written 2026-11-11T01:44:40Z by the `predict` step, 16,469 bytes,
SHA-256 `2c4104f9f1ab584902b9e2c83524f6e07c7781f4840ceceb54ddb11b5336156c`.

Fold of the cycle-0 sheet from the branch: alpha_fold = alpha(point 16) + 6.68e-8, a b = 6,178 per unit alpha
(window 0.15: +6.21e-8 and 6,264; window 0.35: +8.08e-8 and 6,060). The hold is 5.4671e-5 short of it. For
comparison, one constant coefficient from the hold multiplier alone would give 10,698.

**Write of 1, endpoints on the hold-temperature line.**

| Endpoint | alpha beyond the fold | Predicted passage update | Predicted update reaching the unstable orbit | 120-update write | Known outcome |
|---|---|---|---|---|---|
| nominal + 1e-5 | +1.621e-5 | 50.1 | 40.7 | succeeds | passed |
| nominal (E1) | +6.21e-6 | 86.3 | 76.3 | succeeds | passed |
| nominal - 0.25e-5 | +3.71e-6 | 114.5 | 104.4 | succeeds | passed |
| nominal - 0.5e-5 | +1.21e-6 | 208.1 | 197.8 | fails, still creeping at update 120 | failed |
| nominal - 0.75e-5 | -1.29e-6 | none | never | fails, settled on a cycle-0 orbit | failed |
| nominal - 1e-5 | -3.79e-6 | none | never | fails, settled on a cycle-0 orbit | failed |

The predicted minimum duration of a nominal write of 1 is therefore about 76 updates (60 is known to fail, 120 to
pass). Nothing saved tests a duration between those two; that needs new model calls.

**Write of 0, endpoints on the hold-strength line.** Fold at log10 T = -0.1536044833; the hold is 3.4703e-4 short of
it. Rough coefficients per unit log10 T: 1,896 (hold multiplier, constant coefficient) and 1,348 (third-sheet fold
coefficient 3,242 carried along a straight chord of the fold curve). The same constant-coefficient shortcut is off
by a factor of 1.7 on the alpha side, so these are orientation only.

| Endpoint log10 T | Beyond the fold | Rough passage update (two coefficients) | Known outcome |
|---|---|---|---|
| -0.1534 | +2.045e-4 | 24, 28 | passed |
| -0.1535 (E0) | +1.045e-4 | 36, 42 | passed |
| -0.153525 | +7.95e-5 | 42, 49 | passed |
| -0.15355 | +5.45e-5 | 52, 61 | passed |
| -0.153575 | +2.95e-5 | 73, 86 | passed |
| -0.1536 | +4.48e-6 | 197, 234 | failed at 120; the continuation arm ran 240 without a write |

Off-line settings (write of 1 with the temperature offset, write of 0 with the strength offset) are measured and
reported with no prediction.

## What will count as agreement

1. **Write of 1.** Each measured passage update within 15 % of its prediction, and each measured update reaching the
   unstable orbit within 15 % of its prediction. The two blind cases are nominal + 1e-5 and nominal - 0.25e-5.
2. **Kind of failure.** The failing lane beyond the fold (nominal - 0.5e-5) should still be creeping at update 120:
   a per-period displacement that has not decayed geometrically toward zero. The two failing lanes short of the
   fold should be settling on an orbit: displacement shrinking by a steady factor per period.
3. **Fold from the dynamics alone.** On each line the smallest per-period displacement should fall on a straight
   line against the control, and the control at which that line reaches zero should agree with the independent fold
   position to within a small part of the spacing between settings (2.5e-6 in alpha, 2.5e-5 in log10 T).
4. **Write of 0.** The five passing settings should fall on a straight line against the regressor
   [pi/2 + arctan(sqrt(D / delta))] / sqrt(delta), with residuals of a few updates. The coefficient from the slope
   is reported against the two rough values. The line's extrapolation to -0.1536 is compared with the continuation
   arm: an extrapolation above 240 is consistent with the write that never happened, one clearly below 240 is not.
5. The coefficient a b read from the curvature of the bottleneck is reported against the branch value. It is a
   noisy three-sample estimate and is not a pass criterion.

Any miss is reported as it stands. No definition, window or threshold is changed after the states are opened.

## What this can and cannot show

It can show whether writes at eleven endpoint distances behave as one fold passage with a coefficient that was
measured elsewhere (the companion's multipliers), and whether the failures in the pilots are the same passage running
out of time or stopping short of the fold.

It cannot certify a fold, give a basin, or estimate reliability. The settings are a fixed grid from one model at one
hold and one start phase. The write-1 coefficient is from Arnoldi multipliers at a few recorded branch points. The
write-0 fold position is a numerical candidate. The normal form is one-dimensional and continuous in time; the map is
896-dimensional and its orbits alternate sides each period away from the fold. A duration experiment and settings
farther from the fold would be new model calls and are not part of this analysis.

## Results

Added 11 November 2026 after the `measure` step. Everything above this heading is the plan as it stood before any
saved state was opened. That version is bound in the receipt (12,424 bytes, SHA-256
`1e0a69609fa22b157e4d40e2c62d195db7845a3b4380bef006c7164744fbcd2c`) and archived as
`audits/public-write-passage/20261111T014440Z/plan-before-results.md`. Nothing above was edited afterwards.

- Receipt: `docs/receipts/public-write-passage.json` (2026-11-11T01:44:54Z, zero model calls): 40 lanes of four pilot
  runs and 8 lanes of two continuation runs.
- Post hoc receipt: `docs/receipts/public-write-passage-addendum.json` (01:54:21Z, zero model calls), written by
  `tools/analyze_public_write_passage_addendum.py` after the first receipt had been read. It is arithmetic on the two
  receipts and the companion's recorded multipliers. Its readings are marked "post hoc" below and count toward no
  criterion.

### Scorecard

| # | Criterion from the plan | Outcome |
|---|---|---|
| 1a | Write of 1: each passage update within 15 % of its prediction | Met at all three passing settings, two of them blind |
| 1b | Write of 1: each update reaching the hold's unstable orbit within 15 % | **Missed** at all three: measured 17, 20 and 27 % early |
| 2 | Kind of failure: creeping beyond the fold, settling short of it | Met |
| 3 | Fold from the dynamics alone, straight line through every setting | **Missed as defined** on the write-0 line and at four of six phases on the write-1 line; met at two phases on the write-1 line |
| 4 | Write of 0: five passing settings on a straight line; extrapolation against the continuation | Met |
| 5 | Coefficient from the bottleneck's curvature (reported, not a criterion) | Within 1 to 9 % of the branch value at the three write-1 settings nearest the fold |

The 20 lanes that started on the requested cycle all ended on it.

### 1. Write of 1 (lane starting on cycle 0, endpoints on the hold-temperature line)

| Endpoint | alpha beyond the fold | Passage update at 1/4, 1/2, 3/4 | Predicted | Measured / predicted | Reached the unstable orbit: measured, predicted | Stored |
|---|---|---|---|---|---|---|
| nominal + 1e-5 | +1.621e-5 | 37, 43, **49** | 50.1 | 0.98 | 29.5, 40.7 | yes |
| nominal (E1) | +6.21e-6 | 73, 79, **79** | 86.3 | 0.92 | 60.7, 76.3 | yes |
| nominal - 0.25e-5 | +3.71e-6 | 97, 103, **103** | 114.5 | 0.90 | 86.4, 104.4 | yes |
| nominal - 0.5e-5 | +1.21e-6 | none within 120 | 208.1 | | none, 197.8 | no |
| nominal - 0.75e-5 | -1.29e-6 | none | none | | none, never | no |
| nominal - 1e-5 | -3.79e-6 | none | none | | none, never | no |

The passage is read at a sampled level, so it moves in steps of about six updates. All three passages came earlier
than predicted, by 2, 8 and 10 %. The crossing of the hold's unstable orbit came 17, 20 and 27 % early. The narrowest
point of the bottleneck shows the same thing: at nominal - 0.5e-5 the smallest displacement falls at updates 57 to 67
(by phase), where the symmetric normal form puts it at update 99. The form's entry leg is too long; because the whole
passage is nearly right, its exit leg is too short by about as much.

A straight line through the three passages against the model's regressor, with intercept and slope both free, gives
an intercept of +6.9 updates and a coefficient of 8,782 against the branch's 6,178. Three points and two free numbers
test little; the line is reported because the script computed it.

Off the recorded line (no prediction): the write of 1 with the temperature lowered or raised by 1e-4 passed at
updates 73 and 91.

### 2. Kind of failure

- **nominal - 0.5e-5 (1.2e-6 beyond the fold).** The per-period displacement fell to an interior minimum of 1.06e-5
  and was rising again when the pulse ended: 1.37e-5, 1.49e-5, 1.65e-5, 1.85e-5 at the last four periods (ratios
  1.09, 1.11, 1.12). The lane was past the narrowest point and still creeping, 0.034 of the way to cycle 1.
- **nominal - 0.75e-5 and nominal - 1e-5 (short of the fold).** Displacement shrinking by a steady factor per period:
  0.8082, 0.8087, 0.8091, and 0.6651 three times. Both lanes had settled on an orbit beside cycle 0 (0.025 and 0.022
  of the way). The matched continuation held the second for 120 more updates: the displacement went on down to
  6.6e-13.
- **Write of 0 at log10 T = -0.1536 (4.5e-6 beyond its fold).** Displacement still falling at update 120, with
  ratios 0.974, 0.981, 0.989 rising toward 1: the lane was still entering the bottleneck. The matched continuation
  held the same controls for 120 more updates from the identical state (difference 0.0). Over the joined 240 updates
  the smallest displacement came at updates 109 to 120, by phase, and it was rising at the end (ratios 1.14, 1.15,
  1.16). The coordinate read 0.440, 0.403, 0.367, 0.294 at updates 60, 120, 180, 240 and never reached the hold's
  unstable orbit at 0.2316, so the hold returned the state to cycle 1. This write was an unfinished passage, not a
  settled state.
- All four restored-nominal arms of the continuation completed the write: passages at updates 44 and 43 (write of 0)
  and 79 and 84 (write of 1) of the restored stage.

Post hoc (addendum): the lane at nominal - 1e-5 settles at 0.6651 per period. The companion's first-sheet branch at
that strength has multiplier 0.6657 (linear between its two recorded neighbours) or 0.6649 (cubic through four). The
lane at nominal - 0.75e-5 settles at 0.8091; the branch gives 0.8167 or 0.8045 there, from neighbours at 0.766 and
0.866. The fold coefficient alone, 1 - 2 sqrt(a b D), gives 0.694 and 0.821.

### 3. Fold from the dynamics alone: missed as defined

The plan asked for one straight line through the smallest displacement at every setting that has an interior minimum.

- **Write-1 line.** At phases 4 and 5 three settings qualify. The line is straight to 0.03 % and 0.08 % of its
  largest value and reaches zero 2.8e-9 and 1.9e-8 from the branch's fold (0.1 % and 0.8 % of the 2.5e-6 spacing):
  met. At phases 0 to 3 a fourth setting, nominal + 1e-5, qualifies. There the line is not straight (residuals 10 to
  34 %) and its zero is off by 8.9e-7 to 1.06e-5, up to four spacings: missed.
- **Write-0 line.** Zero off by 1.2e-5 to 9.8e-5 (half a spacing to four spacings), residuals 2 to 24 %: missed at
  every phase.

Post hoc (addendum; chosen after the fixed lines had missed, so not a test). The smallest displacement is not a
straight line in the control over these ranges; it bends, and at the far settings the minimum falls inside the first
two periods, where it is a start-up transient and not a bottleneck. Using only the settings nearest the fold, at all
six phases:

| Line | Settings used | Zero of the line minus the independent fold position |
|---|---|---|
| Write 1 | two nearest (1.2e-6 and 3.7e-6 beyond) | -4.3e-9 to -1.8e-8 |
| Write 1 | three nearest | -1.5e-9 to -3.4e-8 at five phases; -1.27e-6 at phase 3, where one minimum is inside the first two periods |
| Write 0 | two nearest of the 120-update pilots (2.9e-5 and 5.4e-5 beyond) | +1.0e-8 to -4.1e-6 |
| Write 0 | joined 240-update continuation at -0.1536 (4.5e-6 beyond) and the next setting | -2.0e-9 to -3.0e-7 |
| Write 0 | the same with a third setting | +6.2e-10 to -1.09e-6 |

The independent positions are the fold fitted to the companion's multipliers (write 1) and the Newton solve of the
fold equations (write 0, log10 T = -0.1536044833). Near the fold the dynamics puts each fold within 2e-8 in alpha
and 3e-7 in log10 T of those positions.

### 4. Write of 0 (lane starting on cycle 1, endpoints on the hold-strength line)

| Endpoint log10 T | Beyond the fold | Passage update at 1/4, 1/2, 3/4 | Fitted line | Measured - fitted | Rough guesses | Stored |
|---|---|---|---|---|---|---|
| -0.1534 | +2.045e-4 | 7, 26, **32** | 31.8 | +0.2 | 24, 28 | yes |
| -0.1535 (E0) | +1.045e-4 | 14, 38, **50** | 49.1 | +0.9 | 36, 42 | yes |
| -0.153525 | +7.95e-5 | 20, 50, **56** | 58.1 | -2.1 | 42, 49 | yes |
| -0.15355 | +5.45e-5 | 32, 62, **74** | 72.9 | +1.1 | 52, 61 | yes |
| -0.153575 | +2.95e-5 | 56, 92, **104** | 104.1 | -0.1 | 73, 86 | yes |
| -0.1536 | +4.48e-6 | none in 240 | 289 (extrapolated) | | 197, 234 | no |

The five passing settings lie on a straight line against the model's regressor: intercept -3.9 updates, slope 0.2050,
largest residual 2.1 updates. The coefficient from the slope is 857 per unit log10 T. The two rough values recorded
beforehand, 1,896 and 1,348, were too high by factors of 2.2 and 1.6, as the plan warned they might be. A bare
inverse square root of the distance fits as well (largest residual 2.3, coefficient 886): these five settings cannot
tell the two forms apart.

The line extrapolates to update 289 at -0.1536. That is above 240, so it is consistent with the continuation arm,
which held those controls for 240 updates without a write.

The quarter and half levels come much earlier than the three-quarter level here (14 and 38 against 50 at E0). The
continuation shows why: the bottleneck of the write of 0 sits near 0.40 of the way from cycle 0 to cycle 1, so the
lower levels are crossed on the way into it. This is the reason the plan gave for reading the passage at three
quarters.

Off the recorded line (no prediction): the write of 0 with the strength lowered or raised by 1e-5 passed at updates
44 and 62.

### 5. Coefficient from the bottleneck's curvature (not a criterion)

| Line | Setting, distance beyond the fold | a b from curvature, median of six phases | Reference |
|---|---|---|---|
| Write 1 | nominal - 0.5e-5, 1.2e-6 | 6,378 (6,274 to 6,558) | 6,178 from the companion's multipliers |
| Write 1 | nominal - 0.25e-5, 3.7e-6 | 6,702 (6,037 to 6,764) | 6,178 |
| Write 1 | nominal, 6.2e-6 | 6,224 (five phases 6,116 to 6,449; one at 290,916 with its minimum at update 9) | 6,178 |
| Write 1 | nominal + 1e-5, 1.6e-5 | no usable value: every minimum is inside the first nine updates | 6,178 |
| Write 0 | -0.153575, 2.9e-5 | 735 | 857 from the passage line |
| Write 0 | -0.15355, 5.4e-5 | 708 | 857 |
| Write 0 | -0.153525, 7.9e-5 | 824 | 857 |
| Write 0 | -0.1535 (E0), 1.0e-4 | 660 | 857 |
| Write 0 | -0.1534, 2.0e-4 | no usable value | 857 |
| Write 0, post hoc | -0.1536 joined over 240 updates, 4.5e-6 | 790 (772 to 794) | 857 |

For the write of 1, the trajectory's own curvature in the bottleneck gives back the coefficient that the companion's
multipliers give, to 1 to 9 %, at the three settings nearest the fold. These two numbers come from different
objects: one from the eigenvalues of recorded periodic orbits short of the fold, the other from the shape of a
transient beyond it.

### Reading, as an interpretation these numbers are consistent with

1. The write of 1 is the companion's cycle-0 sheet ending in a fold just short of the write endpoint. The time the
   write takes follows from the fold's coefficient with no fitted number, to 10 % at three distances, and the
   bottleneck's curvature returns that coefficient to 1 to 9 %.
2. Each failure in the endpoint pilots is one of two things. Either the endpoint is short of the fold, the starting
   orbit still exists, and the state settles on it at the rate the branch records. Or the endpoint is just beyond
   the fold and the passage needs more than the pulse gave it: about 208 updates predicted for the write of 1 at
   nominal - 0.5e-5, about 290 extrapolated for the write of 0 at -0.1536, where 240 were observed not to be enough.
3. The write of 0 follows the same law with its own coefficient, obtained three ways from the dynamics (857 from the
   passage line, 660 to 824 from the curvature at four settings, 790 from the joined continuation).

### What missed, and what is not shown

- The symmetric normal form predicts the crossing of the hold's unstable orbit 17 to 27 % too late. The predicted
  minimum write duration of about 76 updates is therefore not supported, and the minimum is not measured.
  (Corrected 11 November 2026, 03:45 UTC. This item first went on: "the measured crossing at about update 61
  suggests a minimum near there. No duration between 60 and 120 was run". Both parts were wrong as written. The
  second is true only of the write of 1 at the nominal E1, where 60 updates failed and 120 succeeded
  (`public-half-pulse.json`, `public-repair-half-pulse.json`). For the write of 0 at the nominal E0 the study had already run
  shortened pulses: 60 and 72 updates wrote a 0 in every case tried, and 36 updates wrote a 0 from phases 0, 1
  and 2 of cycle 1 and left a qualified 1 from phases 3, 4 and 5 (`public-nominal-handoffs.json`, `public-short-write0-chain.json`, `public-short-write0-phase1.json`, `public-short-write0-phase-panel.json`, `public-half-pulse.json`, `public-repair-half-pulse.json`, `public-intermediate-write0-phase-panel.json`).
  The first part leaned on a one-coordinate picture that those runs contradict. In the full write of 0 from
  phase zero of cycle 1 the coordinate stays beyond the halfway level only from update 38 and reaches the
  projection of the hold's unstable orbit at update 43 (this receipt, nominal row), yet a 36-update pulse from
  that start wrote a 0. The position along the segment between the cycles when the controls return does not by
  itself decide the outcome, so a crossing time is not an estimate of the shortest pulse that writes.)
- One straight line for the smallest displacement does not hold over the full range of settings. The nearest-setting
  version is post hoc.
- The nominal write-1 time was known approximately when the method was fixed; only two of the three passage
  comparisons were blind.
- The write-0 line has two fitted numbers and five points.
- One model, one hold, one start phase per lane, fixed grids of settings no farther than 1.6e-5 in alpha and 2.0e-4
  in log10 T beyond the folds. Not a certified fold, not a basin, not a reliability estimate.
