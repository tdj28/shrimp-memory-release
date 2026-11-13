# Why the 30 degree pulse exchanges the two cycles, and only from phase zero

Written 11 November 2026 (UTC), before any quantity below was computed. Zero model calls. Evidence class: adapted
exploratory, post hoc.

## Why

The human asked for "a precise deterministic explanation including why it has to be done at just the right part of
the cycle", and said the short paper cannot be published without one. The paper explains why holds and resets cannot
exchange the cycles and why a slow excursion cannot either (the cusp heuristic), and it reports that after a toggle the
lanes leave the unstable orbit at the rate of its multiplier. It does not explain why the pulse at 30 degrees, radius
3.5, 12 updates exchanges the cycles from phase 0 and resets them from every other phase.

## History, stated plainly

Seen before this plan was written:

- the outcome table of the toggle check for this pulse (`audits/public-toggle-check`, batch 1): phase 0 toggle; phases
  1 to 5 reset to 1; angle neighbours at 20 degrees reset to 1 at radii 3.15, 3.5 and 3.85, at 30 degrees toggle at
  3.15 and 3.85, at 40 degrees reset to 0 at all three radii; dwells of 8 and 16 updates reset to 1; the three 1e-15
  perturbations toggle;
- the sweep's outcomes at the same control: 48 and 120 updates reset to 1; at radius 3.5, 0 degrees resets to 1 and
  60 degrees resets to 0 for all three durations;
- the register-branch analysis (`docs/public-register-branch.md`): after this pulse the lanes sit 0.000415 and
  0.001060 from the unstable orbit at H and leave it at about 1.77 per period.

No trajectory quantity below had been computed. The hypothesis was formed from those outcomes.

## Hypothesis, stated before computing

- **H1 (where a pulse ends).** At the pulse controls the loop is attracted to a period-6 cycle of its own, C_p. A
  pulse that starts at phase k of a reference cycle and lasts D updates leaves both lanes near the point of C_p at label
  (k + D) mod 6, closer together than they started, by an amount that shrinks as D grows.
- **H2 (why only phase 0).** The point of C_p at label 0 lies on cycle 1's side of the boundary between the two basins
  at H, close to it; the other five lie well inside cycle 1's basin. So pulses ending at labels 1 to 5 (phases 1 to 5,
  and the 8- and 16-update dwells, which end at labels 2 and 4) reset to 1; long pulses ending at label 0 (48 and 120
  updates) also reset to 1, because by then the lanes have merged onto C_p; after 12 updates the lanes have not merged
  and straddle the boundary.
- **H3 (why exchange and not identity).** Along the direction in which the pulse dynamics contract most slowly, the
  lane that started on cycle 1 is farther from C_p than the lane that started on cycle 0. After 12 updates it is still
  beyond the boundary, on cycle 0's side, while the lane from cycle 0 has crossed to cycle 1's side.
- **H4 (why the toggle is narrow in angle).** Rotating the pulse from 20 to 40 degrees moves the landing across the
  boundary: on cycle 1's side at 20 degrees, on cycle 0's at 40.

What would contradict them: the 12-update landings from the six phases are not near one period-6 cycle (H1); the
lanes from phases 1 to 5 come as close to the unstable orbit as the phase-0 lanes do (H2); along the slow direction
the lane from cycle 1 is not the outermost of the points that go to cycle 0 and to cycle 1 (H3).

## What is read

All read-only, bound by size and SHA-256 in the receipt.

- The companion continuation (`/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/kite/kite-la-branch/`), for the
  unstable orbit at H at the branch's saved phase, interpolated exactly as in `tools/analyze_public_register_branch.py`
  (cubic through the four nearest points; the linear estimate as an error check).
- The reference cycles (`audits/public-xor-chain/20261110T013000Z-local/run/reference-cycles.npz`).
- The toggle check, batch 1 (`audits/public-toggle-check/20261110T003500Z-pod-returned/shard-*`): groups 0 to 18 (the
  six phases, eight neighbours, two dwells, three perturbations). Each group is a pulse stage and a 240-update hold,
  every state saved.
- The toggle sweep (`audits/public-toggle-sweep/20261109T230942Z/returned-pod/shard-*`): the pulses at 30 degrees,
  radius 3.5, for 12, 48 and 120 updates (groups 24 to 26), and at radius 3.5 for 0 and 60 degrees, 120 updates
  (groups 14 and 38).

## What is computed (fixed now)

1. **The pulse cycle C_p.** From the 120-update pulse at 30 degrees, radius 3.5: its last six states as C_p at labels
   (t mod 6), t counted from the start of the pulse (which starts at phase 0); the lag-six residual; the two lanes'
   agreement. The per-period ratio of deviations from C_p at the same label, while the deviation lies between 1e-11
   and 1e-5, and the cosine between successive per-period deviations (near +1 or -1: a real leading multiplier; well
   inside: rotation). The unit vector of the latest deviation in that window at each label is that label's slow
   direction v_p, oriented so that lane 0's deviation has a positive coefficient.
2. **Where each pulse ends.** For phases 0 to 5, the two dwells, the three perturbations and the 48-update pulse: at
   the end of the pulse, each lane's distance to C_p at label (k + D) mod 6 and at the nearest label; the separation
   between the lanes; each lane's deviation coefficient along v_p at that label, and the fraction of the deviation's
   norm that this coefficient carries. For phase 0, also every update of the pulse.
3. **What happens at H.** For every group listed above and the eight neighbours: the cyclic alignment of each lane's
   last period to the reference cycles; at every hold update whose aligned phase is the branch's saved phase, the
   distance to the unstable orbit; its minimum; and the first hold update after which the lane stays within 1e-6 of
   its final reference cycle.

Nothing is fitted. Nothing is selected by outcome: every group named above is used.

## What this can and cannot show

It can show where the lanes are when the dials return to H, relative to the pulse's own cycle and to the unstable
orbit at H, and whether the outcomes order themselves along one direction as H1 to H4 say. It cannot show the basin
boundary itself (that needs the left eigenvector of the unstable orbit or model runs from chosen states), a
certified cycle at the pulse controls, or that the account holds for other pulses. A test that predicts the outcomes
of new pulse lengths before running them would need model calls and a plan of its own.

## Results

From `docs/receipts/public-toggle-mechanism.json` (final run 2026-11-11T21:32:08Z, zero model calls, 610 input files
bound). Rounded here. One item was added after the first run, and is marked as such in the script: the coefficients
along the slow line every period of the 120-update pulse, which shares the 12-update pulse's first 12 updates bit
for bit.

**The pulse cycle.** At the pulse controls the loop has an attracting period-6 cycle C_p: the last period of the
120-update pulse closes to 5.5e-14 or better at every label, and the two lanes agree to 6e-15. Deviations from it
shrink by 0.131 per period (median of 71 ratios, 0.1314 to 0.1335), and successive per-period deviations are parallel
(cosines 0.9997 to 1.0000). The leading multiplier is real and positive, so near C_p the approach runs along one line
per label, without overshoot.

**Where a pulse ends (H1: holds).** In every pulse at these controls (six phases, two dwells, three perturbations,
and the sweep's 12-, 48- and 120-update pulses), the point of C_p nearest the end of the pulse is the one at label
(k + D) mod 6. The lanes end 0.00008 to 0.0031 from it after 8 to 16 updates, and about 1e-7 from it after 48.

**Phase 0, 12 updates.** At the end of the pulse both lanes lie on C_p's slow line at label 0 (99.95 % and 99.5 % of
each deviation): the lane from cycle 0 at +0.00080, the lane from cycle 1 at +0.00207. The 48- and 120-update pulses
follow the same two paths bit for bit for their first 12 updates, then carry both lanes onto C_p (+0.00003 and
+0.00032 at update 24, below 1e-6 by update 42). Returned to H, C_p's own point (the long pulses) goes to cycle 1, the
point at +0.00080 goes to cycle 1, and the point at +0.00207 goes to cycle 0. Along that line the boundary between the
two basins at H therefore lies between +0.00080 and +0.00207. After 12 updates the lanes are on opposite sides of it,
in exchanged order: the toggle. Had the dials returned one period later (update 18) the lanes would have been at
+0.00021 and +0.00104; two periods later (update 24), at +0.00003 and +0.00032, both on cycle 1's side.

**Why exchange and not identity (H3: the outcome holds, the stated reason does not).** At the return the lane from
cycle 1 is the outer one, beyond the boundary, as H3 said. But it did not start farther out: at update 0 it is 0.0012
from C_p's label-0 point and the lane from cycle 0 is 0.0042. During the first period at the pulse controls the lane
from cycle 1 is thrown out (0.015 from C_p at update 5; -0.0074 along the slow line at update 6, on the far side of
C_p) and then swings back past C_p to +0.0021 by update 12, while the lane from cycle 0 approaches steadily (+0.0036,
+0.0019, +0.0008). A change of sign along a line whose multiplier is positive means that first period is nonlinear.
These data do not show why the lane from cycle 1 is thrown out.

**Why only phase 0 (H2: holds in part).** A 12-update pulse from phase k ends at C_p's label k. Back at H the lanes
from phases 2 to 5, and from the 16-update dwell (label 4), come no closer than 0.0023 to the unstable orbit, and all
go to cycle 1. Phase 1 (label 1) also lands near the boundary: one lane comes within 0.00059 of the unstable orbit,
but both lanes stay on cycle 1's side. The 8-update dwell (label 2) lands 0.0015 and 0.0039 from it. Only at label 0
does the boundary pass between the two lanes. H2 said the other five labels lie well inside cycle 1's basin. That is
true for labels 2 to 5, not for label 1.

**Why the toggle is narrow in angle (H4: holds).** At 20 degrees (radii 3.15, 3.5, 3.85) both lanes go to cycle 1 and
come no closer than 0.0018 to the unstable orbit; at 40 degrees both go to cycle 0, no closer than 0.0011; at 30
degrees, radii 3.15 and 3.85, the lanes straddle the boundary (0.00014 and 0.00121; 0.00030 and 0.00065) and
exchange. The long pulses at radius 3.5 put C_p's label-0 point deep in cycle 1's basin at 0 degrees (0.0121 from the
unstable orbit), on cycle 1's side near the boundary at 30 degrees (0.0014), and in cycle 0's basin at 60 degrees
(0.0050).

**Account, as an interpretation these numbers are consistent with.** The toggle is a straddle of the basin boundary
at one phase. Wherever a 12-update pulse starts, it leaves both copies near one point of the pulse controls' own
cycle, the point at the starting phase, still spread along one slow line. Back at H, the boundary between the two
basins runs through that spread only at label 0. There the first period at the pulse controls has thrown the copy
from cycle 1 to the outside, beyond the boundary, so the copies come back exchanged. It needs phase 0 because only
that point of the pulse cycle has the boundary passing through the copies' spread. It needs 12 updates because one or
two more periods merge the copies onto that point, which is on cycle 1's side: a reset. The pulse is a whole number of
periods, so the XOR schedule's 252 updates bring the loop back to phase 0.

**Not shown.** Where the boundary crosses the line, beyond the bracket (+0.00080, +0.00207); why the lane from cycle 1
is thrown out in its first period; the fate of states on the far side of C_p (negative coefficients), so the outcome
of a 6-update pulse cannot be predicted from these data; whether the boundary crosses the line only once; anything
about the other eight toggles. A prospective test needs model calls and its own plan: from phase 0 at these controls,
an 18-update pulse should exchange the cycles if the boundary lies below +0.00104 and reset to 1 if it lies above, and
a 24-update pulse should reset to 1 if the boundary crosses the line once.

**Follow-up, 11 November 2026, 22:15 UTC.** The prospective test proposed above was run
(`docs/public-toggle-timing.md`, receipt `docs/receipts/public-toggle-timing.json`). Both predictions held: 24 updates
reset to 1, and after 18 updates the lane from cycle 0 reached cycle 1. The lane from cycle 1, at +0.00104 after 18
updates, also reached cycle 1, which narrows the boundary to between +0.00104 and +0.00207 on the slow line.
