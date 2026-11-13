# Next-write readiness after repair

Adapted exploratory comparison, motivated by the completed hold-state repair
and retrospective early-readout analysis. Their outcomes have been seen; this
is not a confirmatory test of general reliability.

For each next-write target bit, use the opposite repaired bit from the completed
hold-repair experiment. Preserve both original source phases. Select nominal
repair groups1and3 (repair started after36and240 failed-hold updates). From each
repaired final hold, select literal post-update states36and240. This makes four
matched batches of two states per target, sixteen rows across the pair.

| Group | Failed-hold age before repair | Repaired-hold age before next write |
|---|---:|---:|
| 0 | 36 | 36 |
| 1 | 36 | 240 |
| 2 | 240 | 36 |
| 3 | 240 | 240 |

Each branch applies120nominal endpoint updates for the opposite bit followed
by240common-hold updates. No observer label selects or skips a branch. All
qualified wrong, unresolved, invalid and incomplete results must remain.
The earlier reference-phase labels identify origin history, not the current
transient phase. Inputs are copied exactly; no additional settling or reference
replacement is allowed. Four initializations per process, no within-branch reset.

History contains120perturbed endpoint updates,36or240failed-hold updates,
120repair endpoint updates, and36or240repaired-hold updates. New360updates yield
complete path lengths672,876,876,1080. Historical model work is separately
accounted and is not free initialization. A successful result would show the
selected early states accepted this next write, not arbitrary composability.

Intended execution bounds match the previous four-group batch design:
1464calls per process, two states per batch, one numeric thread, nativefloat64,
1800sinternal2100sexternal plus30sgrace,8GiBRSS16GiBVM,256MiBraw including8MiB
finalization reserve,352MiBarchive,448MiBtransfer reservation,8MiBsource upload.
Fresh capacity and transfer admission, isolated committed source checkouts,
actual-input preflight, and verified local return workers must precede launch.
No numerical launch is authorized by this document alone without those gates.

Input preparation is committed in tools/prepare_public_repair_handoff.py.
It rehashes the complete source audits and reconstructs source trajectories
before saving four literal batches. It performs no model or observer replay.
The numerical producer is tools/run_public_repair_handoff.py. Both actual-file
preflights and27execution tests passed before launch. The independent final
auditor for this new profile remains required before any outcome claim.
