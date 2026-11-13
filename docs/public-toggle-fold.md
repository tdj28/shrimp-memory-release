# Why the 30-degree pulse reverses the order of the two copies: a fold in the pulse's return map

Written 11 November 2026 (UTC), before any quantity below was computed. Part A: zero model calls. Part B: model
calls, not run without the human's go-ahead. Evidence class: exploratory, post hoc.

## Why

The human: the flip mechanism "must be impeccably explained, no hand waving". The account in
`docs/public-toggle-mechanism.md` shows where the two copies (lanes) are when the dials return to H and that H's basin
boundary passes between them, in exchanged order. It does not say why the order is exchanged. It says only that the
first period at the pulse controls is nonlinear, and "why the lane from cycle 1 is thrown out is not known".

## What was seen before this plan

Everything in the results of `docs/public-toggle-mechanism.md` and `docs/public-toggle-timing.md`, in particular the
receipt's trace of the phase-0 pulse (13 updates: each lane's distance to the pulse cycle C_p and its coefficient
along the slow direction v_p at that label) and the label-0 coefficients every period of the 120-update pulse. Reading
those numbers for this plan:

- the fraction of each deviation along v_p is 0.87 to 1.00 at labels 0, 3, 4 and 5, but 0.02 to 0.26 at labels 1 and
  2 in the first period, and 0.17 to 0.81 in the second;
- the label-0 coefficients six updates apart are +0.00186 to +0.00080 and -0.00736 to +0.00207 (second period), then
  +0.00080 to +0.00021 to +0.00003 and +0.00207 to +0.00104 to +0.00032. By mental arithmetic the positive pairs are
  close to s' = 0.131 s + 164 s^2. No other quantity below was computed.

## Hypotheses, stated before computing (Part A)

The register's dynamics are close to one-dimensional: at H every multiplier but one has modulus below 0.007, and at
the pulse controls deviations from C_p shrink along one line per label. Write s for the coefficient along v_p at label
0. The idea to be tested: the six updates of one period act on s as a one-hump map whose turning point lies very close
to C_p (C_p is nearly superstable: its multiplier 0.131 is small because one update nearly flattens the slow line).
Two copies on opposite sides of the turning point come out on the same side, the one that was farther from the
turning point farther out. That is an order reversal.

- **F1 (one update folds the slow line).** In the linear regime (deviations from C_p between 1e-9 and 1e-5 in the
  120-update pulse) the one-update factors along the slow lines, mu_k = (coefficient at label k+1) / (coefficient at
  label k), are consistent across periods and lanes and multiply to the per-period ratio, 0.131. The smallest is at
  the update from label 0 to 1 or from 1 to 2, and it is the only one below 0.3. Contradiction: the smallest is
  elsewhere, or two or more factors are below 0.3.
- **F2 (the fold is quadratic).** At that update (label k_f to k_f + 1), take every visit to label k_f whose deviation
  is between 1e-4 and 1e-2 and at least 98 % along v_p. The deviation one update later, minus its linear part
  mu c v_p, divided by c^2 (c the coefficient before the update), is nearly one fixed vector: pairwise cosines at least
  0.9 and norms within a factor 2. Contradiction: any cosine below 0.9 or any norm ratio above 2.
- **F3 (one-hump return map at label 0).** Pool every pair of label-0 visits six updates apart, with both visits at
  least 98 % along v_p, from the 120-update pulse (phase 0) and the 12-update pulses from phases 1 to 5 (their label-0
  positions have never been computed). (a) For s > 0, every image is positive and smaller than s, and images increase
  with s. (b) For s below -0.001, every image is positive, and images decrease as s increases. A pair violates the
  order of its side if its image is out of order by more than 5 % of the larger image. Contradiction: any violation,
  or an image of the wrong sign.
- **F4 (consequence, not a test).** If F1 to F3 hold, the toggle is this fold. In the second period from phase 0 the
  copies lie on opposite sides of the turning point (from cycle 1 at -0.00736, from cycle 0 at +0.00186), and the fold
  sends the farther one farther out: +0.00207 against +0.00080, on opposite sides of H's boundary
  (between +0.00104 and +0.00207).
- **F5 (the first period, descriptive).** The start points are points of H's cycles, not of the pulse's slow curve.
  For every lane, the fraction along v_p at its first label-0 visit and that visit's image are reported; for a visit
  below 98 %, also the difference between its image and the image an on-curve point at the same s would have,
  interpolated between the nearest pooled on-curve pairs on the same side. The reading to be checked: near the
  turning point the slow coordinate's first-order effect vanishes, so a copy that starts off the slow curve lands where
  its offset sends it. Part B tests this.

What is read: the toggle check, batch 1 (groups 1 to 5: phases 1 to 5) and the sweep's 120-update pulse at 30
degrees, radius 3.5 (group 26; its first 12 updates equal the phase-0 check bit for bit), the reference cycles of the
audited XOR chain. C_p and v_p are defined exactly as in `tools/analyze_public_toggle_mechanism.py`, and the script
first reproduces the receipt's trace. Script `tools/analyze_public_toggle_fold.py`; receipt
`docs/receipts/public-toggle-fold.json`. Nothing is fitted for the tests; the quadratic coefficient above is not used.

## Part B (needs model calls and the human's go-ahead; not run)

From phase 0 at the pulse controls, for each lane, start at the point between the lane's start and its projection onto
C_p's slow line at label 0, x(lambda) = proj + lambda (start - proj), for lambda = 0, 0.25, 0.5, 0.75 and 1, and run six
updates: 2 x 5 x 6 = 60 calls. Predictions: lambda = 1 reproduces the saved paths; for the lane from cycle 1 the
landing at update 6 moves monotonically from near C_p at lambda = 0 (within 0.0005, on the side of positive s) to
-0.00736 at lambda = 1. If so, the offset of H's cycle from the pulse's slow curve, not the position along it, throws
that copy to the far side. If not, the first period needs another explanation.

### Part B execution, fixed 12 November 2026 before any call (human's go-ahead: "Yes, run Part B")

- As in every earlier run, both lanes run together in one group, so the ten starts are five groups (lambda = 0, 0.25,
  0.5, 0.75, 1). Each group is one stage of six updates at the 30-degree, radius-3.5 controls, plus the instrument's
  three diagnostic calls: 45 calls in all, fewer than the 60 stated above. No hold at H, so fates are not measured.
- Starts are prepared with zero model calls by `tools/prepare_public_toggle_offset.py`. That script takes C_p and v_p
  from the pod's 120-update pulse exactly as `tools/analyze_public_toggle_mechanism.py` does; proj is C_p's label-0
  point plus the lane's coefficient along v_p times v_p. The lambda = 1 start is the reference state itself, not
  recomputed, and the script checks it equals the saved phase-0 start bit for bit. The runner
  (`tools/run_public_toggle_offset.py`, the check's instrument with a start file instead of start phases) refuses a
  start file whose SHA-256 differs from the one given on its command line. One group per single-thread process.
- Scored by `tools/analyze_public_toggle_offset.py` (zero model calls; receipt
  `docs/receipts/public-toggle-offset.json`):
  - B0 (path): at lambda = 1 every state of both lanes is within 1e-12 of the pod's phase-0 path. If B0 fails, the
    test is void.
  - B1 (the prediction above, unchanged): the landing of the lane from cycle 1 at update 6 decreases monotonically
    from lambda = 0 to 1, and at lambda = 0 lies within 0.0005 of C_p on the side of positive s. Noted before running:
    Part A's curvature estimate puts the on-line image of s = -0.00099 within about 3e-5 of C_p, so at lambda = 0 its
    sign may come out either way; the criterion is scored as written.
  - Reported, not scored: each lane's coefficients and fractions along v_p at every update; for both lanes, the
    landing at update 6 against a straight line in lambda.
- Every outcome is reported, including a void or failed run; there is no rerun.

## What this can and cannot show

Part A can show whether the pulse's period map acts on the slow coordinate as a single fold close to C_p, and whether
pulses started from other phases obey the same map. It cannot show the map where no saved trajectory goes, or the
fate at H of states on the far side of C_p, and it does not test the first period; Part B does that.

## Results

From `docs/receipts/public-toggle-fold.json` (zero model calls). The script first reproduced the mechanism receipt's
phase-0 trace exactly (same path bit for bit). Two items were added after the first run and are marked in the script
and the receipt: F5b and F6. Both are descriptive.

**F1 holds.** One-update factors along the slow lines in the linear regime (8 or 9 ratios each, spread below 1 %):
label 0 to 1, 0.193; 1 to 2, 0.440; 2 to 3, 0.759; 3 to 4, 2.647; 4 to 5, 1.684; 5 to 0, 0.458. Product 0.1315, the
per-period ratio. The update from label 0 to 1 contracts the slow line most and is the only factor below 0.3. (The
cycle is not close to superstable in the sense of one factor near zero: the smallest is 0.19.)

**F2 fails as stated.** At that update, the 20 visits with |c| from 2.1e-4 to 7.4e-3 and 98 % or more along v_p have
second-order parts along one direction (pairwise cosines 0.997 or more). But their size per c^2 is 76 to 85 for c < 0
and grows from 82 to 193 as c goes from +2e-4 to +2.6e-3: ratio 2.56, beyond the factor 2 fixed in the plan. The bend
is one-dimensional, but over this range it is not a pure quadratic.

**F3 holds.** 19 pairs of label-0 visits six updates apart lie 98 % or more on the slow line at both ends: 15 on the
near side (s > 0), 4 on the far side (s < -0.001), none in between. No order violation, no wrong sign. Near side: the
images are positive, smaller than s and increasing (+0.00207 to +0.00104, +0.00186 to +0.00080, +0.00104 to +0.00032,
+0.00080 to +0.00021, down to the linear regime). Far side: -0.00736 to +0.00207, -0.00617 to +0.00186, -0.00428 to
+0.00116, -0.00326 to +0.00071. These images are positive and fall as s rises toward C_p. Three of the four far-side
pairs come from phases 1 to 3, whose label-0 positions had never been computed; they fall in order with the phase-0
pair. So the period map along the slow line is a one-hump map. Its turning point lies between -0.0033 and C_p: the
image is +0.0007 at -0.0033, and just left of C_p it is negative, because the slope there is +0.131. No saved state
lies in that interval.

**F5 and F5b: the first period.** Every lane arrives at its first label-0 visit off the pulse's slow curve (fraction
along v_p 0.79 to 0.93), except the lane from cycle 1 from phases 1 to 3, which reaches label 0 only after three to
five updates (0.99 to 1.00). In the three runs that reach label 0 within two updates (phases 0, 5 and 4), the lane from
cycle 1 arrives close to C_p (s = -0.0010, -0.0010, -0.0009), with off-line parts 0.00072, 0.00046 and 0.00037. Three
updates later, past the fold update, it is on the far side: -0.00242, -0.00149, -0.00118, that is -3.38, -3.24 and
-3.15 times its off-line part. That close to C_p, a state on the line itself can only go to the bottom of the hump.
Estimated from the curvature on either side (a quadratic through the slope 0.131 and the nearest pairs; an estimate,
not a test), that bottom lies a few times 1e-5 below C_p, about 0.0004 to its left. So in these runs the off-line
part, not the position along the line, sends the copy to the far side. This is consistent with three runs and is not a test; Part B
tests it.

**F4: the account of the toggle.** From phase 0 the first update at the pulse controls is the fold update. It acts on
two points of H's cycles, which lie off the pulse's slow curve. The copy from cycle 0 starts on the near side
(+0.00364) and comes in along it (+0.00186 after one period). The copy from cycle 1 starts near the turning point,
where its offset dominates, and is thrown to the far side: -0.00242 at label 3, then the expanding updates (3 to 4 and
4 to 5) carry it out to -0.0150, and it ends the period at -0.00736. In the second period the copies are on opposite
sides of the turning point. The fold sends both to the near side, and the one from farther out lands farther out:
+0.00207 against +0.00080. Their order along the line is now reversed. Back at H they return to opposite cycles: the
exchange.

**Why only from phase 0 (from the traces, descriptive).**

- From phases 1 to 3 the pulse reaches the fold update only after three to five updates. By then the copy from cycle
  1 is on the slow curve, on the far side (-0.0062, -0.0043, -0.0033). The fold brings it in inside the copy from
  cycle 0 (at label 3: +0.00095 against +0.00168, +0.00055 against +0.00146, +0.00032 against +0.00097). The order is
  kept and there is no exchange.
- From phases 5 and 4 the copy from cycle 1 is thrown out as from phase 0, but its offset is smaller, so it lands less
  far out (-0.0041 and -0.0031 at the next label-0 visit, against -0.0074). The second pass either reverses the copies
  by a little (from phase 5: +0.0026 against +0.0021 at label 5) or brings them together (from phase 4: 1.5e-5 apart at
  the end). Both end on cycle 1's side.
- Only from phase 0 does the fold update meet the copy from cycle 1 while it still carries its full offset from H's
  cycle. That throws it far enough (-0.0074) for the second pass to put it clearly outside the other copy (+0.0021
  against +0.0008), with H's boundary between them.

**F6 (added after the first run, descriptive): the slow line does not order the fates of neighbouring pulses.** The
check's eight neighbours end at label 0 at 30-degree slow-line coordinates that do not separate by fate:

- at 20 degrees the lane from cycle 1 ends at +0.0018 to +0.0021 and returns to cycle 1, while at 30 degrees a lane at
  +0.0021 returns to cycle 0;
- at 40 degrees, lanes at +0.0010 to +0.0024 all return to cycle 0, while at 30 degrees +0.0010 returns to cycle 1.

Across pulse controls the fate at H also depends on the part of the state off that line (fractions 0.66 to 1.00). So
the toggle's narrowness in angle cannot be read off this line, and the paper's "which is why the toggle is narrow"
has no support. Likewise, the boundary bracket (+0.00104, +0.00207) holds for this pulse's own states, which lie 99.5 %
or more on the line, and not for arbitrary states near it.

**Status.** Shown from saved trajectories:

- the slow line is folded by one update per period (0 to 1);
- the period map along the line is a one-hump map, and pulses from three further phases obey it;
- the toggle is the second pass of the two copies through that fold, from opposite sides;
- the phase dependence follows from when the pulse first meets the fold update.

Not shown: that the offset from the slow curve causes the first-period throw (Part B), where exactly the turning
point and H's boundary lie, and why the toggle is narrow in angle.

## Part B results (12 November 2026, 45 model calls)

Run `audits/public-toggle-offset/20261112T020802Z-local` (five single-thread shards on this machine, 9 calls each, cache
comparisons passed; starts bound by SHA-256 820314c8...). Score: `docs/receipts/public-toggle-offset.json`
(`tools/analyze_public_toggle_offset.py`, zero model calls; audit passed: manifests, starts and controls).

- **B0 holds.** At lambda = 1 both lanes stay within 4.3e-15 of the pod's phase-0 path over all six updates.
- **B1 fails as written, on the sign at lambda = 0; its monotone part holds.** The lane from cycle 1, started at the
  same position on the line (s = -0.00099) with its offset scaled by lambda, lands after one period at
  -0.00016, -0.00145, -0.00308, -0.00506 and -0.00736 for lambda = 0, 0.25, 0.5, 0.75 and 1: strictly decreasing, and
  at lambda = 0 within 0.0005 of C_p, but at negative s, not positive as written. Before the run the plan noted that
  this sign could go either way.
- **Reading.** The offset throws the copy out. Without it the copy lands 1.6e-4 from C_p; with it, 7.4e-3. That is
  98 % of the throw. Three updates in (label 3), the landing is close to linear in the offset: -0.00008 plus about -3.0
  to -3.3 times the off-line norm. At update 6 it bends away from a straight line in lambda by at most 6.8e-4.
- **Correction to Part A.** Part A's curvature estimate, that an on-line state at s = -0.001 stays within a few times
  1e-5 of C_p, is wrong: the measured on-line image is -1.6e-4, and every state of that path lies at least 99.9 % along
  the slow lines. The bottom of the hump is therefore at least 1.6e-4 below C_p. Its position is still known only to
  lie between -0.0033 and 0. The tangent line is also not the slow curve far from C_p. The lane from cycle 0, put on
  the line at s = +0.00364, lands at +0.00011, below the on-curve images at smaller s. Its landing is not monotone in
  lambda (+0.00011, +0.00171, +0.00232, +0.00237, +0.00186). Its offset matters too, but not in proportion.
- **Status of the account.** The toggle is the second pass of the two copies through the hump of the pulse's period
  map, from opposite sides. The copy from cycle 1 gets to the far side because the bending update, the first update
  of a pulse started at phase 0, acts on its offset from the pulse's slow curve: tested here. Still not shown: where
  exactly the turning point and H's boundary lie, and why the toggle is narrow in angle.

## Part C: the zero-offset counterfactual to a readout (fixed 12 November 2026, before any call)

Human's go-ahead: "Yes, run it" (about 260 calls). Codex's Task 7 item 2 independently recommends the same test.

- **What runs.** One group, both lanes together, started from Part B's lambda = 0 states: each lane's start projected
  onto C_p's slow line at label 0, at the same position along it, with no offset. These are index 0 of the bound start
  file `audits/public-toggle-offset/20261112T020802Z-local/starts/starts.npz`. The pulse at 30 degrees, radius 3.5,
  runs for 12 updates, then 240 updates at H. Instrument, readout, qualification gates and pair mapping are those of
  the toggle check, unchanged. 258 calls.
- **Predictions.**
  - Z0: the first six updates reproduce Part B's lambda = 0 path within 1e-12 (same start, controls and machine).
  - Z1: no exchange. Both lanes read cycle 1 (reset to 1).

  Reason: by Part B, after one period both lanes are within 2e-4 of C_p (+0.00011 and -0.00016). A second period
  near C_p contracts both toward it, and C_p's point and every state tested between it and +0.00104 went to cycle 1.
- **What would contradict the account.** An exchange, or a hold, with Z0 holding.
- **Reported.** The readout and pair mapping; each lane's coefficient along v_p at updates 6 and 12; the closest
  approach to the unstable orbit at H; the settle update. Every outcome is reported, and there is no rerun.

### Part C results (12 November 2026, 258 model calls)

Run `audits/public-toggle-offset-readout/20261112T233414Z-local` (one single-thread shard, 255 s, cache comparisons
passed). Score: `docs/receipts/public-toggle-offset-readout.json` (`tools/analyze_public_toggle_offset_readout.py`,
zero model calls; audit passed).

- **Z0 holds.** The first six updates equal Part B's lambda = 0 path bit for bit (largest difference 0).
- **Z1 holds.** No exchange: both lanes read cycle 1 (reset to 1). The runner's readout and the recomputation from
  saved states agree.
- Along the slow line the lanes are at +0.000106 and -0.000161 after one period, and at +0.000018 and -0.000017
  after two. Every state is at least 99.9 % on the line. At H both lanes come no closer than 0.0014 to the unstable
  orbit, as C_p's own point does, and settle from hold update 108.
- **Reading.** With its offset (the original pulse) the lane from cycle 1 is thrown out and the cycles are exchanged.
  Placed on the line at the same position, without the offset, it is not thrown out, the lanes merge near C_p, and
  both reach cycle 1. For this pulse from phase 0, the offset of H's cycle from the pulse's slow line is what the
  exchange depends on.
