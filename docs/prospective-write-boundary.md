# Prospective pulse-release test
The six predictions in boundary_plan use the existing estimate of the write-1 fold and its branch coefficient. Each independent row starts on reference A0, phase zero. Compare the fold-crossing time with the pulse duration, then score the measured bit after the standard 240-update hold. The duration-only comparator predicts 1 at durations of at least 90 updates (midpoint of the historical nominal 60/120 bracket). Omissions and erasures score as misses for both classifiers; operationally interrupted rows remain incomplete. Report each row, both totals, and their disagreements. No substitutions or refitting. Six designed rows do not estimate population accuracy. The two predictions use the same controls and timeout. One scalar CPU process, 1944 calls, 1200 seconds internal limit, 20 GiB peak limit. A failed prediction makes the historical runner exit nonzero even with all six rows completed. Retain all artifacts and verify the final trajectories independently.

## Provenance

The fold (alpha = 0.07299164917758323) and the branch coefficient (ab = 6177.93 per unit alpha) are the first-sheet
values in `docs/receipts/public-write-passage-prediction.json`. The plan above, word for word, and the runner's
`boundary` protocol were committed to the private study repository (commit 8275b9a, 17 November 2026, 19:14:08 UTC)
twenty seconds before the run started (19:14:28 UTC). This repository received them a few minutes later, so its own
history does not show the freeze. The scorer, `tools/audit_write_boundary.py`, was written after the run. The run
recorded the SHA-256 of the four source files it executed; the receipt (`docs/receipts/prospective-write-boundary.json`)
copies them and notes that each equals the file here.

## Measured outcome

All six holds resolved. At offsets of −2, +4 and +12 millionths in steering strength from the fitted fold, the measured bits after 54 and 102 pulse updates were respectively (0,0), (0,1) and (1,1). The frozen geometry rule called all six. The 90-update duration comparator called four. The two rules disagreed on two rows and both favored the geometry rule on scoring. The largest lag-six closure residual was below 2.86e-11. One single-thread CPU process completed the 1,944 calls in 522 seconds rounded up. No row was replaced. The receipt binds both full trajectories per row. A second calculation from the raw holds reproduced each label, and all 192 distinct raw files cited in the row records passed size and hash checks (both done by `tools/audit_write_boundary.py`). This tests the final fate on the specified panel. It does not locate the exact switching update or estimate reliability over other phases and controls.
