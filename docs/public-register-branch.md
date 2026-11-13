# The public register on the companion study's recorded branch

Written 11 November 2026 (UTC). Zero model calls. Evidence class: adapted exploratory, post hoc.

## Why

The deep review of the short paper (`docs/consults/2026-11-10-short-paper-review/`) asked for two things the paper
did not report: the Floquet multipliers of the two period-6 cycles at the hold H, and evidence that the writes are
tied to folds of the branch (a multiplier reaching +1) instead of an appeal to "fold-mediated switching". Both can
be answered from files that already exist, without running the model.

## History, stated plainly

This was not planned before the data were seen. A first look in a scratch directory (10 November, during the
eight-note run) found that the hold strength and the E1 write strength are, bit for bit, two of the strengths at
which the companion study (arXiv:2610.31692) saved its continuation of the period-6 orbit along the hold temperature.
`tools/analyze_public_register_branch.py` was written after that first look and reproduces every number from the
saved files. One statistic was changed after the script's trial run: the approach to the orbit at E1 was first
summarised by the median one-period ratio of the last five periods. The one-period ratios there alternate (about 0.90
and 0.96), so that median describes neither; it was replaced by the geometric mean over an even number of periods.
No model output was produced or inspected for this analysis.

## What is read

- **Companion continuation (read-only):** `neural-shrimp-data/explore/llm-soft-loop/pod64/kite/kite-la-branch/`
  (`branch.json`, `states.npy`). 124 Newton-converged period-6 points along log10 T = -0.15395151820007266, free
  parameter alpha, tolerance 1e-10, each with Arnoldi estimates of its leading multipliers and the companion's own
  list of where a real multiplier crosses -1 (flip) or +1 (fold).
- **This study:** the audited XOR chain's reference cycles and its 240-update holds at H after each of its four
  toggle pulses (`audits/public-xor-chain/20261110T013000Z-local/run`), and the E1 write stages of the eight-note
  threshold run (`audits/public-note-byte/20261110T234809Z/threshold/run-2`), after that run's audit.

Every input file is bound by size and SHA-256 in the receipt.

## What the script reports

1. **The line.** The four recorded crossings; the three sheets between them (first: attracting, carries cycle 0;
   middle: one multiplier above +1; third: attracting, carries cycle 1); the interval of alpha on which both outer
   sheets are recorded as attracting; where H sits in that interval.
2. **At H.** The branch point whose alpha equals the hold alpha exactly, its distance to our reference cycle 0 and
   its multiplier. The third-sheet and middle-sheet orbits at H, each interpolated between the two recorded
   neighbours that bracket H: linear, and a cubic through the four nearest points as an interpolation-error check.
3. **E1.** The branch point whose alpha equals the E1 alpha exactly, its multiplier, and how far E1 lies beyond the
   fold that ends the first sheet and short of the flip that ends the third.
4. **Saved dynamics at H.** For each toggle of the XOR chain and each lane, the states at the branch's saved phase
   during the following hold: the per-period ratio of the distance to the final cycle while that distance lies in a
   fixed window, and the distance from the interpolated unstable orbit during the first five periods.
5. **Saved dynamics at E1.** For each 120-update stage held at exactly the E1 controls and each lane: the distance
   to the recorded E1 orbit once per period, the one-period ratios, their two-period geometric means, the first
   sampled update at which the distance is below half its starting value, and the first at which the deviation has
   changed side.

## Fixed choices

- Linear window for settling ratios: deviation norm between 1e-9 and 1e-5 (small against the 5.7e-3 distance from a
  cycle to the unstable orbit, large against the 1e-11 closure of the reference cycles). Fixed in the script before
  its trial run; not tuned.
- The phase at which the branch states were saved is identified as the phase of our reference cycle 0 nearest to
  the exact-alpha branch point.
- Nothing is fitted. Nothing is selected by outcome: all four toggles and every E1 stage are used.

## What this can and cannot show

It can show that the register's two cycles are the companion's recorded branch at H, give their multipliers and the
unstable orbit's, say where the write endpoint E1 lies relative to the recorded fold, and check those multipliers
against trajectories this study had already saved.

It cannot show a basin, a certified bifurcation, a stable-manifold geometry, or anything about E0, which lies at a
different temperature (its own fold candidate is in `docs/receipts/public-e01-amended-geometry.json`). The
multipliers are Arnoldi estimates from the companion study. The interpolated values at H carry an interpolation
error, reported as the difference between the linear and cubic estimates. An agreement between a multiplier and a
measured ratio is a consistency check on a handful of trajectories, not a reliability estimate. The account of the
write of 1 as a slow passage beyond a fold is an interpretation that these numbers are consistent with; no
experiment here varies the distance past the fold.

## Results

From `docs/receipts/public-register-branch.json` (final run 2026-11-11T01:07:06Z, zero model calls). Every number
below is in that receipt; rounded here.

**The line.** 124 points, largest Newton residual 9.97e-11. The companion recorded four crossings:

| Between points | Type | alpha (two neighbours) | Leading multiplier (two neighbours) |
|---|---|---|---|
| 0, 1 | flip (-1) | 0.0729069929, 0.0729115362 | -1.0000000014, -0.93274 |
| 16, 17 | fold (+1) | 0.0729915824, 0.0729915447 | +0.96015, +1.04938 |
| 60, 61 | fold (+1) | 0.0727928303, 0.0727927639 | +1.03158, +0.98833 |
| 104, 105 | flip (-1) | 0.0730069082, 0.0730161158 | -0.97411, -1.01380 |

Points 0 to 16 are the first sheet (attracting), 17 to 60 the middle sheet (one multiplier between +1.03 and +1.80,
every other recorded modulus at most 0.0021), 61 to 104 the third sheet (attracting). Both outer sheets are recorded
attracting for alpha from 0.0729069929 to 0.0729915824: from the first sheet's flip to its fold, not between the two
folds. H lies 2.999e-5 above the lower edge and 5.460e-5 below the upper edge.

**At H.**

| Orbit | How obtained | Distance to this study's reference | Leading multiplier | Next modulus |
|---|---|---|---|---|
| Cycle 0 (first sheet) | branch point 4, alpha equal to the hold bit for bit | 7.4e-12 | -0.529516 | 0.0027 |
| Cycle 1 (third sheet) | between points 95 and 96 | 1.05e-9 (cubic), 5.3e-7 (linear) | -0.622101 (cubic), -0.621961 (linear) | 0.0066, 0.0062 at the neighbours |
| Unstable orbit (middle sheet) | between points 30 and 31 | none saved | +1.764479 (cubic), +1.763879 (linear) | 0.0010 at the neighbours |

The saved phase is phase 5 of the reference cycles. At that phase the cycles are 0.015380 apart; the unstable orbit
is 0.005654 from cycle 0 and 0.012608 from cycle 1 (its linear and cubic interpolations differ by 1.3e-6).

**E1.** Branch point 103 has alpha equal to the E1 write strength bit for bit (Newton residual 4.1e-12). It lies on
the third sheet with leading multiplier -0.933709 (next modulus 0.0044). E1 is 6.27e-6 in alpha beyond the last
recorded point of the first sheet (multiplier +0.960), which is 89.7 % of the way from H to E1, and 9.1e-6 to 1.8e-5
short of the flip that ends the third sheet. At E1 the recorded branch therefore has one attracting period-6 orbit,
weakly attracting. (Wording corrected 11 November: an earlier version said "beyond the fold". The fold itself lies
between that point and the next. A fit of the recorded multipliers, in
`docs/receipts/public-write-passage-prediction.json`, puts it 6.7e-8 farther on, so E1 is 6.21e-6 beyond the fold.)

**Saved dynamics at H** (XOR chain, the 240-update hold after each of the four toggle pulses; window 1e-9 to 1e-5).

| Quantity | From saved trajectories | From the branch |
|---|---|---|
| Per-period contraction onto cycle 0 | median 0.529519 of 56 ratios (0.52787 to 0.53261) | 0.529516 |
| Per-period contraction onto cycle 1 | median 0.622101 of 72 ratios (0.62135 to 0.62291) | 0.622101 |
| Sign of successive deviations | cosine -1.0000 for both | both multipliers negative |
| Per-period growth away from the unstable orbit, first period | 1.767 (lane ending on cycle 0), 1.797 (lane ending on cycle 1) | 1.7645 |

After the toggle pulse the lane that ends on cycle 0 is 0.000415 from the unstable orbit (2.7 % of the cycle
separation) and the lane that ends on cycle 1 is 0.001060 from it (6.9 %). The later growth ratios fall away from
1.76 as the distance leaves the linear range (1.749, 1.731, 1.670, 1.458 and 1.844, 1.901, 1.738, 1.130). All four
toggles give the same two trajectories with the lanes exchanged.

**Saved dynamics at E1** (eight-note threshold run, the four 120-update stages held at E1: stages 0, 4, 6, 14).

- Five lanes started on cycle 0. Their distance to the recorded E1 orbit, sampled once per period, stays between
  0.0153 and 0.0140 for the first 59 updates, falls to 0.0083 at update 77 and 0.00091 at update 83, and from
  update 89 the deviation alternates in sign. The geometric mean of the last four one-period ratios is 0.9262. The
  lag-six residual at the stage's last update is 5.3e-4; the readout's closure gate asks for 1e-8. The five
  trajectories agree to a relative 2e-9.
- Three lanes started on cycle 1. Their distance falls from 0.00100 to 0.00026 with alternating sign throughout;
  the one-period ratios alternate between about 0.90 and 0.96, and the geometric mean of the last four is 0.9315,
  against 0.9337 from the branch. The lag-six residual at the last update is 2.6e-4.

**Reading, as an interpretation these numbers are consistent with.** At H the register is the companion's
S-shaped branch: two attracting period-6 orbits and one unstable orbit between them, each with a single multiplier
that matters. The write of 1 moves alpha to a point just beyond the fold where the cycle-0 sheet ends. A state that
was on cycle 0 then drifts slowly through the place where its orbit used to be (about 80 updates here), crosses to
the one surviving orbit and approaches it at about 0.93 per period. Three earlier observations fit that picture:
the E1 endpoint showed no qualified period after 480 updates (the approach is too slow for a 1e-8 closure in that
time); a 60-update write requesting 1 failed with a clean readout while one requesting 0 succeeded (60 updates end
before the passage, which here happens near update 80); and the eight notes used as continuous controls did not
write (`docs/receipts/public-note-byte.json`: the notes meant as ones moved alpha 74 to 78 % of the way to E1, short
of the fold at 90 %, though they also moved the temperature, so that comparison is only a rough guide).

**Not shown.** This analysis uses no run that varies the write duration or the distance past the fold. The fold is
bracketed by two recorded points, not located by a defining system. (Added 11 November: the endpoint pilots of 26
October did vary the distance past the fold. Their passage times are analysed in `docs/public-write-passage.md`,
receipt `docs/receipts/public-write-passage.json`.) Nothing here concerns E0 beyond what the earlier fold-candidate receipts state. The agreement of
rates is over a handful of trajectories from one model at one hold.

**Correction, 11 November 2026, 03:45 UTC.** The note added above first ended with the sentence "The write duration is still not varied." That sentence was wrong about the study and has been removed. Shortened writes were run in October and are in the full report. At the nominal E1, 60 updates: every write requesting 1 ended in a qualified 0 (`public-half-pulse.json`, `public-repair-half-pulse.json`). At the nominal E0, 36, 60 and 72 updates: 60 and 72 wrote a 0 in every case tried; 36 wrote a 0 from phases 0, 1 and 2 of cycle 1 and left a qualified 1 from phases 3, 4 and 5 (`public-nominal-handoffs.json`, `public-short-write0-chain.json`, `public-short-write0-phase1.json`, `public-short-write0-phase-panel.json`, `public-half-pulse.json`, `public-repair-half-pulse.json`, `public-intermediate-write0-phase-panel.json`). What remains true is narrower: this analysis uses none of those runs, no duration between 60 and 120 updates was run at E1, and the shortest pulse that writes is not located for either bit.
