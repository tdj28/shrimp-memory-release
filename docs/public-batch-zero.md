# Reverse-control batched public write

This is one separately recorded adapted experiment after the fixed write1
batch recovered both requested rows and met the full saved-trajectory
comparison. The previous run and its source at commit
`42f07ee935aad9d39674ffa57901b35b271ad1bf` remain immutable. That result tested the write1 endpoint, whose
steering differs from the hold. This run tests the reverse public operation
and its temperature-only control change; it is not a pulse or phase scan.

Run `tools/run_public_batch_writes.py --requested-bit 0` with the same eight
path and three hash arguments documented in `public-batch-writes.md`.
Omitting `--requested-bit` retains the fixed write1 default. The only accepted
values are integer 0 and 1. An immutable validated plan supplies the exact
public strings, canonical actions, stage controls, historical row identities,
design, budget and source provenance. No historical module globals are changed.
The new run selects zero once, before model loading; no result can switch plans.

Both visible messages are exactly `Write bit 0 to register R0.`. Row zero
starts at saved ad phase zero, giving a same-bit control; row one starts at
saved td phase zero, giving the reverse switch. Both receive the unchanged
canonical 120-update endpoint and 240-update hold. Alpha is
0.07293697809562831 in both stages. The endpoint log10 temperature is -0.15350;
the hold log10 temperature is -0.15395151820007266. The stage adapter must
rebuild the prefix even though alpha does not change. The original pinned
steering vector remains explicit. CPU native float64, math SDPA, one thread,
reference cycles, model/configuration/vector hashes and prerequisite checks
remain unchanged.

Use the already audited scalar isolated trials 0 and 1 from source
`f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4`. Bind their actual metadata,
results, 121/241-state trajectories and 120/240-token records before loading.
Their initial bytes, stage continuity and input identities must qualify under
the new zero plan. Compare every one of the 361 distinct carried states and
360 top tokens per row. The existing gates remain maximum coordinate error
at most 1e-10, zero token mismatches and matching qualified terminal labels;
exact byte equality is separate. Terminal closure, proper-divisor exclusion
and unique all-phase reference matching remain neutral, with no desired bit
passed to the readout. Request correctness is an independent evaluation.

All call, storage and failure contracts in `public-batch-writes.md` apply:
366 admitted slots on completion (two prefixes, four diagnostic forwards and
360 batch updates), 720 row updates, no reset, row drop, early stopping,
extension or retry. Finite cached/full or scalar-trajectory disagreement does
not truncate the schedule. Structural, cache-mutation, precision, execution
and resource errors abort while retaining planned slots and partial blocks.
The 1e-12 cached/full coordinate and probability gates are unchanged.

The raw envelope remains 28,627,392 bytes including metadata/container
allowances, with 2,473,984 bytes plus an event allowance admitted before each
call for a failed native-shape return. Limits remain 600 internal seconds,
780 external seconds plus 30 seconds grace, one thread, 8 GiB RSS, 16 GiB
virtual memory, 30 MiB working raw plus 2 MiB finalization reserve, 48 MiB
archive, 64 MiB transfer reservation and at most 8 MiB source upload. Store
each trajectory state once in stage initials and twelve-update blocks, with
final manifests; retain scalar and actual expanded cache snapshots once per
stage. No local weight-cache upload is part of the run.

Model-free regression tests retain the default write1 behavior and exercise
the explicit zero plan, temperature-only prefix rebuilding, wrong plan/control
rejection, actual old filenames, reconstruction and failure preservation. A
zero-forward preflight must qualify the actual zero inputs. Preserve every
test attempt and every numerical outcome. Commit source and this adaptation,
record a run card, then execute the one zero attempt and return its evidence.

A positive result extends finite batch equivalence to the two known starts
for the reverse operation. It does not establish arbitrary starting phases,
independently addressed registers, general reliability, a matched runtime
speedup, natural-language semantics or hidden communication. Historical timing
and the earlier write1 duration remain descriptive, rather than matched
performance controls for this run.
