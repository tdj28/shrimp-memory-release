# Does the toggle account predict new pulse lengths? 18 and 24 updates at 30 degrees

Written 11 November 2026 (UTC), before any model call of this test. Evidence class: prospective check of a post hoc
account.

## Why

`docs/public-toggle-mechanism.md` explains the 30 degree, radius 3.5, 12-update toggle from saved trajectories: the
pulse ends near the pulse controls' own period-6 cycle C_p, at the point with label (start phase + duration) mod 6,
and from phase 0 the two lanes lie on that point's slow line, the lane from cycle 0 at +0.00080 and the lane from
cycle 1 at +0.00207. C_p's own point and +0.00080 return to cycle 1 at H, +0.00207 to cycle 0, so the basin boundary at
H crosses the line between them. That account was built after the outcomes were known. This test asks it to say, in
advance, what two pulse lengths never run before will do. The test was proposed with the account; the human approved
it on 11 November ("Test, then paper").

## What runs

Two groups on this machine, one single-thread process each, with the instrument, inputs, readout and pair mapping of
the toggle check (`tools/run_public_toggle_check.py`, the same as the sweep's), through
`tools/run_public_toggle_timing.py`:

- `t030-r3.5-d018-phase0`: both lanes start on the reference cycles at phase 0; the pulse at 30 degrees, radius 3.5,
  for 18 updates; then 240 updates at H.
- `t030-r3.5-d024-phase0`: the same, for 24 updates.

About 530 model calls in all. Both pulse lengths are whole periods, so both end at label 0.

## Predictions, fixed now

The pod's 120-update pulse at these controls (`docs/receipts/public-toggle-mechanism.json`) starts from the same
states, so its first 18 and 24 updates are these pulses' paths. Along the slow line at label 0 it puts the lanes at
+0.000209 (from cycle 0) and +0.001040 (from cycle 1) after 18 updates, and at +0.000033 and +0.000319 after 24.

- **P0 (paths).** Each lane's state at the end of each pulse is within 1e-12 of the pod's state at that update. (The
  audited XOR chain's two computers agreed to 1.7e-14.) If P0 fails, the predictions below do not apply and the test
  is void.
- **P1 (24 updates): reset to 1.** Both lanes read cycle 1. Both landings lie between C_p's point, read as cycle 1
  after the 48- and 120-update pulses, and +0.00080, read as cycle 1 after the 12-update pulse.
- **P2 (18 updates, lane from cycle 0): cycle 1.** It lands at +0.00021, between the same two points.
- **Measurement, not a prediction (18 updates, lane from cycle 1).** It lands at +0.00104, inside the bracket. Cycle 0
  places the boundary between +0.00080 and +0.00104, and the pulse then toggles; cycle 1 places it between +0.00104
  and +0.00207, and the pulse resets to 1.

What would refute the account: P1 or P2 failing while P0 holds. That would mean the boundary crosses the slow line more
than once between C_p's point and +0.00080, or that where a lane goes depends on more than its place on that line.

## Reported

For each group: the pair mapping from the runner's readout; an independent recomputation from the saved states (each
lane's final cycle by nearest reference point); each end-of-pulse state's distance to the pod's state at the same
update; the coefficients along the slow line; and, at H, the closest approach to the unstable orbit and the update
after which each lane stays within 1e-6 of its final cycle, computed as in the mechanism analysis. Every outcome is
reported, including a failed or void run; there is no rerun.

## Not shown

The boundary's position beyond the narrowed bracket; why the first period throws the lane from cycle 1 out; whether
the account holds for the other eight toggles or other controls.

## Execution record

- First launch, 2026-11-11 about 22:05 UTC: both shards stopped while loading the model, before any model call (0
  calls, status `execution_failed`). The launcher pointed at a Python environment with transformers 4.55, which
  rejects the instrument's `dtype` argument. Kept as `attempt-1-wrong-environment/` in the run directory. No outcome
  existed, so this is not a rerun of a result.
- Second launch: the study's pinned CPU environment (`/archive/shrimp-memory-data/envs/llm-cpu`: Python 3.12.14,
  torch 2.14.1+cpu, transformers 5.19.0, as in `docs/hardware-ledger.md`). Nothing else changed.

## Results

Run directory `audits/public-toggle-timing/20261111T215842Z-local` (shards 00 and 01, second launch; 264 and 270 model calls,
763 and 777 s, cache comparisons passed). Score: `docs/receipts/public-toggle-timing.json`
(`tools/analyze_public_toggle_timing.py`, zero model calls). The reference cycles used are byte for byte those of the
audited XOR chain.

- **P0 (paths): holds.** Both pulses start exactly at the pod's start and stay within 1.2e-13 of the pod's path over
  every pulse update; at the end of the pulse the lanes differ from the pod's by 5e-15 to 4e-14.
- **P1 (24 updates): holds.** Both lanes read cycle 1 (reset to 1), at +0.000033 and +0.000319 on the slow line, as
  predicted. Closest approach to the unstable orbit at H: 0.00139 and 0.00131.
- **P2 (18 updates, lane from cycle 0): holds.** It reads cycle 1, at +0.000209 (closest approach 0.00135).
- **Measurement (18 updates, lane from cycle 1).** At +0.001040 it reads cycle 1, so the 18-update pulse resets to 1
  and the boundary crosses the slow line between +0.00104 and +0.00207. Closest approach 0.00088, the nearest of any
  lane that went to cycle 1 from label 0.
- The runner's readouts and the recomputation from saved states agree for all four lanes.

Along the slow line at label 0, the closest approach to the unstable orbit falls steadily as the position grows
(0: 0.00140; +0.00003: 0.00139; +0.00021: 0.00135; +0.00032: 0.00131; +0.00080: 0.00106; +0.00104: 0.00088), and the
lane at +0.00207 lands on the other side (0.00041, cycle 0). That is consistent with one crossing of the boundary
between +0.00104 and +0.00207. It is not proof that there is only one crossing.

**Reading.** Of the returns to H after whole periods that have been run from phase 0 at these controls (12, 18, 24,
48 and 120 updates), only 12 exchanges the cycles. By update 18 the lane from cycle 1 has come back across the
boundary to cycle 1's side. Six updates has not been run: there the lane from cycle 1 is on the far side of C_p,
where no state's fate is known.
