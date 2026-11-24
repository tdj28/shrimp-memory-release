# Basins of the two cycles at H: a phase-zero slice, a zoom and basin stability

Written 23 November 2026 (UTC), before any model call of this test. Evidence class: fixed exploratory measurement,
with a few checks predicted from earlier receipts. The human chose the scope and one RunPod pod on 23 November
("slice + zoom + basin stability, on one RunPod pod"). Cost estimate and hard ceiling: see private ledger.

## Why

The papers say that no basin of attraction was measured (short paper, Secs. III and VII; it is also listed as open for
generation by language models in Farid, arXiv:2611.21983). The register's account of writes and of the toggle rests on
which side of the boundary between the basins of A_0 and A_1 at H a state lies, and that boundary is known only as a
bracket on one line: between +1.04e-3 and +2.07e-3 along the pulse cycle's slow line at label 0
(`docs/public-toggle-timing.md`). This test maps the basins on a plane containing that line, zooms in on the boundary,
and estimates how much of a ball around each cycle returns to it.

## Fixed inputs

- Map, prompt, steering vector, instrument, backend guards and readout: those of the toggle check
  (`tools/run_public_toggle_check.py`, the sweep's), unchanged. Every start is held at H for 240 updates and read by
  `pilot.neutral_readout` against the audited reference cycles (`public-xor-chain/20261110T013000Z-local/run/
  reference-cycles.npz`); the label is A_0, A_1 or unresolved. Each group carries two starts as its two lanes.
- The plane. Origin C = C_p at label 0, the pulse cycle's phase-zero point; e_1 = v_p at label 0, its slow direction;
  e_2 = the part of A_0[0] - A_1[0] orthogonal to e_1, normalised. C_p and v_p are those of
  `tools/analyze_public_toggle_fold.py` (`cycle_and_directions` on the pod's 120-update pulse, sweep group 26). A start
  is x(s, t) = C + s e_1 + t e_2. Measured before this plan (zero model calls): A_0[0] lies at (s, t) = (+3.64, +2.03)
  x 1e-3 and A_1[0] at (-0.99, -0.54) x 1e-3, each 4.7e-4 off the plane; the phase-0 toggle's two end states lie
  within 6e-6 and 4e-5 of it.
- **M (main slice).** s and t each on 65 equally spaced values from -5e-3 to +5e-3 (spacing 1.5625e-4; s = 0 and
  t = 0 are grid values). 4225 starts, row-major in t then s, paired consecutively into lanes; the odd last start is
  paired with a copy of the grid's centre start (s = t = 0) as a same-batch determinism check.
- **Z (zoom), rule fixed now.** On the t = 0 row of M, take the first label change from A_1 to A_0 at s > 0, and let
  s* be the midpoint of the two pixels. Z is a 65 x 65 grid of width 6.25e-4 (one sixteenth of M) centred on (s*, 0),
  paired as in M. If the t = 0 row of M has no such change, Z is centred on the M pixel pair with a label change nearest
  to (+1.5e-3, 0) along any row, ties broken by smaller |t| then smaller s; if M has no label change at all, Z is not
  run.
- **S (basin stability).** For each cycle c and each kick size eps in {0.005, 0.01, 0.02} of |A_c[0]|, 500 starts
  A_c[0] + eps |A_c[0]| r, with r uniform on the unit sphere in 896 dimensions (normalised standard normals from
  `numpy.random.default_rng(20261123)`, drawn in the order eps, then start index, then cycle). Lane 0 carries the A_0
  start and lane 1 the A_1 start of the same eps and index: 1500 groups.
- **X (second host).** 64 groups of M chosen by `default_rng(20261124).choice(n_groups, 64, replace=False)`, run again
  on the local machine with the same code and inputs.

Calls: 243 per group (240 updates, one prefix, two comparisons). M 2113 groups, Z 2113, S 1500: 1,391,418 calls on the
pod; X 15,552 calls locally. On the toggle-sweep pod (10 single-thread workers, 0.61 s per call) that is about 24 hours.
Each worker gets an internal wall cap of 30 hours and a raw-output cap of 16 GiB; outputs (about 70 GB) return to
`/archive/shrimp-memory-data/audits/public-basin-map/<UTC>/`, are hash-verified, and the pod's termination is verified.

**Amendment, 23 November 2026, 16:45 UTC, before any call of M, S or Z.** The local link is on Wi-Fi with the
receive side at 3.6 Mbit/s (the wired port has had no carrier since 7 November), too slow to return about 80 GB, and
the first pod's datacenter has no network storage; that pod was terminated unused. The human proposed keeping the raw
evidence on a RunPod network volume and returning the key results first. M, S and Z therefore run on one CPU pod
(cpu5g, 32 vCPU, AMD EPYC 9575F, 128 GB) in EU-RO-1 with 22 single-thread workers, through
`public-basin-map/pod-run.sh`: each shard runs on the pod's local disk and is copied, hash-checked file by file, to
the network volume before its local copy is removed. The receipts, labels and every shard's JSON records come home
first; the volume's full contents follow when the link allows, and the volume is deleted only after they are verified
here. Code, inputs, starts, rules and predictions are unchanged. Cost: see private ledger.

**Amendment, 23 November 2026, 17:25 UTC, after the first pod had started M and before any other host ran.** To
finish today, the human approved more pods. Two more CPU pods (cpu3g, 32 vCPU, AMD EPYC 9654 and 9655) on the same
network volume run `public-basin-map/helper-run.sh`: M, then S, then Z, each in reverse shard order, while the first
pod works forward. Every shard is claimed atomically on the volume before it runs and records the host that ran it, so
no shard runs twice and every pixel's host is known. Hosts differ in the last bits of float64 (B4 measures how often
that changes a label); a pixel's host is never chosen by its label. Z is still built once, from M's labels, by the rule
above.

**Amendment, 23 November 2026, 19:10 UTC, while M runs (115 of 265 M shards read; no S or Z call yet).** (1) A fourth
CPU pod (cpu5g, 16 vCPU, AMD Threadripper 7960X) on the same volume runs `helper-run.sh` with 11 workers. (2) The first
pod's driver skips shards a helper has claimed, so it ends its list while the helpers' last shards still run; its
analysis then stops at missing starts and it would not run Z. `public-basin-map/pod-z-after.sh` waits for that driver
to exit and for Z's starts, then works through Z forward under the same claims. (3) The analysis tool now ignores
`.partial` copies in flight and writes B4's lane index as an integer (it crashed writing the receipt when X was given);
labels and checks are otherwise unchanged. (4) B1 as written misplaces the old bracket's upper point. The pulse
landing at +2.07e-3 that read A_0 lies at t = +0.20e-3 on this plane, not on t = 0 (its lower points: +0.80e-3 at
t = +0.02e-3, +1.04e-3 at +0.04e-3; computed from the plane and the sweep's 120-update pulse, zero model calls). M's
partial t = 0 row, already read when this was noticed, reads A_1 from s = -0.94e-3 to +5e-3. B1 is reported as written,
and therefore as failing if the full row agrees; the pixel nearest each old landing is reported beside it, marked as
added after M's partial labels were seen. Code of the runner, inputs, starts, rules and other predictions are unchanged.
Cost: see private ledger.

**Amendment, 23 November 2026, 23:20 UTC, after M and S were read and while Z runs.** B3 as built cannot test what
it states: the odd last start of M (the grid corner) is paired with the copy of the centre start, so the two lanes of
M's last group carry different starts and their terminal states differ for that reason alone. The centre start is
also lane 0 of group 1056. B3 is therefore evaluated as written in intent, by `tools/check_public_basin_b3.py`: lane
1 of group 2112 against lane 0 of group 1056, all 240 updates, bitwise, from stage files checked against their shards'
manifests. The same-group flag is kept in the receipt, marked as a design error.

## Checks predicted from earlier receipts

- **B1 (the bracket).** On the t = 0 row of M, every pixel with -1.0e-3 <= s <= +1.0e-3 reads A_1 and every pixel with
  +2.1e-3 <= s <= +3.6e-3 reads A_0. (C_p's own point and the states at +0.80e-3 and +1.04e-3 read A_1; +2.07e-3 read
  A_0; those states were within 4e-5 of the plane, not on it.)
- **B2 (the cycles).** The M pixel nearest the projection of A_0[0] reads A_0, and the one nearest A_1[0] reads A_1.
- **B3 (determinism).** The two lanes of M's last group, which carry the same start, end bitwise equal.
- **B4 (hosts).** In X, at least 124 of the 128 lanes read the same label on both hosts. Earlier work found the two hosts
  agree in mapping but not in terminal bytes (`docs/public-toggle-sweep.md`); disagreements near the boundary are
  possible and would be reported, not removed.

If B3 fails, the run is void. If B1 or B2 fails, the maps are reported with the failure, and the paper's account of
the toggle is revised before any figure is used.

## Measured, not predicted

- Fractions of A_0, A_1 and unresolved in M and Z; the number of label changes along each row and column; whether the
  boundary in Z is one curve, several, or not resolved at that scale. A 65 x 65 grid at two scales cannot measure a
  fractal dimension, and none will be claimed.
- For S, the fraction returning to its own cycle at each eps, with Wilson 95% intervals, and where the others went.
  This is basin stability in the sense of Menck et al. (2013) for these kick sizes, on one phase, at H only.

## What would surprise us

Unresolved pixels far from the boundary (slow return at H away from the unstable orbit); a third attractor; A_0 or
A_1 pixels as isolated islands deep inside the other basin in M; basin stability below one half at eps = 0.005.

## Not claimed

Other phases, other controls, the stable manifold away from this plane, any language-interface or reliability claim.
