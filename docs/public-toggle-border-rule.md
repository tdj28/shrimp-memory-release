# Are the nine toggles thin borders between a reset to 0 and a reset to 1? A test on saved outcomes

Written 11 November 2026 (UTC), before any adjacency below was computed. Zero model calls. Evidence class:
exploratory, post hoc (the rule was formed after every outcome existed).

## Why

The human asked whether the short paper's explanations are cheap, and wants the paper to be great (target: arXiv
only). The paper explains one toggle (30 degrees, radius 3.5, 12 updates) and gives a heuristic that slow excursions
cannot toggle. It says nothing general about where among pulses toggles occur. This plan tests a general rule, stated
in conversation before any adjacency was computed, against the saved sweep and the saved toggle checks.

## History, stated plainly

- Every outcome used here existed before the rule was formed. Earlier sessions (10 November) printed the sweep's
  outcome table and each candidate's check verdict when the audits ran.
- When the rule was stated (11 November), this session's context held only: the sweep's class counts (79 hold, 30
  reset to 0, 26 reset to 1, 9 toggle); the 30-degree toggle's check neighbourhood (20 degrees reset to 1 at all three
  radii, 40 degrees reset to 0 at all three, 30 degrees at radii 3.15 and 3.85 toggle), its phases and dwells; the
  sweep's 48- and 120-update outcomes at 30 degrees, radius 3.5 (reset to 1), and at 0 and 60 degrees, radius 3.5.
- After the rule was stated and before this plan, the two candidate lists were read (names of the nine toggles only)
  and the field names of the audit reports. No other outcome was looked at.
- The 30-degree neighbourhood helped suggest the rule, so it is reported but not counted as a test.

## The rule

Write each outcome as the pair (fate of the lane that started on cycle 0, fate of the lane that started on cycle 1):
hold (0, 1), reset to 0 (0, 0), reset to 1 (1, 1), toggle (1, 0).

- **Rule A (continuity).** Along a continuous path of pulse controls, at fixed duration and start phase, a lane's
  fate changes only where its end-of-pulse state crosses the boundary between the two basins at H. Generically one
  lane crosses at a time. So neighbouring outcomes differ in one lane: the outcomes form a cycle hold - reset to 0 -
  toggle - reset to 1 - hold, and hold next to toggle, or reset to 0 next to reset to 1, needs two crossings at once.
- **Rule B (where toggles are).** Where a pulse leaves the two lanes close together, a path from a reset to 1 to a
  reset to 0 must pass a window in which one lane has crossed and the other has not. That window is a hold if the
  pulse kept the lanes' order and a toggle if it reversed it. A slow, order-keeping excursion therefore cannot
  toggle (the precise form of the paper's heuristic), and every toggle is a thin border between a reset-to-0 region and
  a reset-to-1 region, with a width set by the lanes' separation at the end of the pulse.

Caveats stated with the rule: long pulses at settings where the loop does not settle may give outcome patterns finer
than the 10-degree check grid; the rule does not say why a pulse reverses the order.

## Tests, fixed now

**P1 (primary; the eight toggles other than 30 degrees).** Each toggle's check has eight neighbours from phase 0:
angle minus 10, same, plus 10 degrees, times radius 0.9, 1.0, 1.1 times the toggle's, without the centre. For each
toggle:

- B1: at least one neighbour resets to 0 and at least one resets to 1;
- B2: no neighbour is a hold;
- B3: no pair of neighbours opposite each other across the centre resets the same way at both ends. The four pairs
  are (angle - 10, 0.9 r) with (angle + 10, 1.1 r); (angle - 10, r) with (angle + 10, r); (angle - 10, 1.1 r) with
  (angle + 10, 0.9 r); (angle, 0.9 r) with (angle, 1.1 r). Allowed: opposite resets, toggle with toggle, toggle with
  a reset.

A toggle passes if B1, B2 and B3 hold. The count of passes out of eight is reported, split by pulse length (12
updates against 48 or 120). A neighbour whose readout is unresolved is listed, and the tests are applied to the
resolved neighbours with the toggle flagged. What contradicts the rule: a hold among the neighbours (the toggle is not
a thin border at this grid scale); all resolved neighbours resetting the same way (an island); same resets at both
ends of an opposite pair. Context only, not a null model: if the eight neighbours were drawn independently with the
sweep's class frequencies, B2 alone would hold with probability (65/144)^8, about 0.0017.

**P2 (descriptive; the coarse sweep grid).** For each toggle, the outcomes of its sweep neighbours: angle plus and
minus 30 degrees at the same radius and length; the next radius in and out at the same angle and length; the same
pulse at the other two lengths. Over the whole grid, the counts of each pair of outcomes for angle neighbours (same
radius and length) and radius neighbours (same angle and length). Rule A says a reset to 0 next to a reset to 1, and a
hold next to a toggle, hide a window between grid points. These are listed; the grid cannot test them.

**P3 (the premise of Rule B).** For each of the nine toggles (the sweep's run), the lanes' separation at the end of the
pulse, the same relative to their separation at the start, and each lane's change over its last six pulse updates
(near zero if the pulse has settled on a period-6 cycle). Ordinal claim, fixed now: any toggle that fails P1 ends its
pulse with a larger lane separation than every toggle that passes. If all pass or all fail, the claim is not tested.

The dwell neighbours (two thirds and four thirds of the length) and the phase variants are recorded, not tested:
they are discrete changes, not paths of controls.

## What is read

All read-only, bound by size and SHA-256 in the receipt: the sweep audit
`audits/public-toggle-sweep/20261109T230942Z/audit-pod.json` (the pod host has all 147 groups); the check audits
`audits/public-toggle-check/audit-batch1.json` and `audit-batch2.json`; for P3, the nine toggles' saved pulse
trajectories in `audits/public-toggle-sweep/20261109T230942Z/returned-pod/shard-*`, and the reference cycles of the
audited XOR chain. Script `tools/analyze_public_toggle_border.py`; receipt `docs/receipts/public-toggle-border-rule.json`.

## What this can and cannot show

It can show whether the saved outcomes around each toggle have the arrangement the rule predicts, at the grid scale
the check used. It cannot show the borders themselves, which windows hide between coarse grid points, or that the
lanes' order is reversed at the other eight toggles; those need trajectories (P3 gives only separations) or model
calls.

## Results

From `docs/receipts/public-toggle-border-rule.json` (zero model calls). One item was added after the first run and is
marked as such in the script and the receipt: P3b, the 120-update pulse at each toggle's controls.

**P1: the rule fails. 0 of 8 toggles pass** (0 of 3 at 12 updates, 0 of 1 at 48, 0 of 4 at 120). The 30-degree
toggle, which suggested the rule, has the predicted arrangement (reset to 1 at all three radii at 20 degrees, reset to
0 at all three at 40, toggles at 30 degrees). Of the eight others:

- six have a hold among their eight neighbours (B2 fails): 210/3.5/120, 240/2.0/120, 240/3.5/12 (four holds),
  270/3.5/120, 300/3.5/48 (three), 300/3.5/120 (two, plus one unresolved neighbour);
- 150/3.5/12 has five resets to 0 and three toggles around it and no reset to 1 (B1 fails), and the opposite pair
  at 150 degrees, radii 3.15 and 3.85, resets to 0 at both ends (B3 fails);
- 240/2.0/12 has no hold and both resets, but one opposite pair resets to 1 at both ends (B3 fails).

**P2 (descriptive).** On the sweep's 30-degree grid, 14 of 48 unlike angle-neighbour pairs (0.29) and 9 of 43 unlike
radius-neighbour pairs (0.21) are double changes (a reset to 0 next to a reset to 1, or a hold next to a toggle). If
classes were placed independently with the sweep's frequencies the fraction would be 0.23. At this spacing the
outcomes show no sign of the order Rule A describes; the grid is too coarse to say more. Nine reset-to-0 /
reset-to-1 angle neighbours are listed in the receipt as places where Rule A puts a hidden window.

**P3 (the premise).** End-of-pulse separation of the lanes, relative to the start (0.0053 for every toggle, all start
at phase 0): 30 degrees 0.24; 150/3.5/12 0.35; 240/2.0/120 0.21; 240/3.5/12 0.89; 240/2.0/12 3.0; 300/3.5/120
2.5; 270/3.5/120 4.4; 300/3.5/48 4.9; 210/3.5/120 5.1. The ordinal claim is untested, because no counted toggle passes.

**P3b (added after the first run, descriptive).** Only at the 30-degree controls does the 120-update pulse settle both
lanes onto one period-6 cycle (lanes 2.6e-14 apart; last six updates change by 2e-14). At the other eight toggles'
controls it does not: after 120 updates the lanes are 3e-4 to 2.7e-2 apart and each changes by 5e-4 to 2.4e-2 over
its last six updates. Two of those toggles also change outcome under the 1e-15 perturbations of the check (270/3.5/120
one of three, 300/3.5/120 two of three).

**Reading.** The rule holds where its premise holds and fails elsewhere: only the 30-degree pulse ends near an
attracting cycle of its own with the lanes still apart, and only there is the toggle a thin border between a reset to
1 and a reset to 0. The other eight are at controls where the loop has not settled in 120 updates, and the outcomes
of neighbouring pulses mix all four classes at the 10-degree scale. They are not candidates for a gate, and the paper
should not suggest that toggles in general sit on thin reset borders. The heuristic that slow excursions cannot
toggle is not touched by this: none of the eight is a slow excursion.
