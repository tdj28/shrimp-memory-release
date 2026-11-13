# Six phase-dependent write round trips

This exploratory follow-up tests whether the isolated write pulses compose
without resetting the state. It follows four successful phase-zero isolated
writes and a mixed perturbation test; those outcomes informed this design.
Neither earlier failures nor their fixed cutoffs change.

Start once at each of the six phases of the saved `ad` reference cycle.
For each start, request `td`, then request `ad`. Carry the actual terminal
state between the two writes, even when the first readout is wrong or
unresolved. Do not replace it with a reference, correct its phase, extend a
stage, or select another trial. This is six chains and twelve writes, not
all phases of both source cycles or an endurance test.

Each write uses the existing endpoint controls for 120 steps, followed by
240 steps at the common hold. Rebuild the prefix cache at each control change
and save a cached/full comparison without advancing the input. Use the pinned
offline Qwen snapshot, CPU native float64 normalization and rotary operations,
math attention and one thread, as in the preceding run.

Read each write with the unchanged gates: maximum all-phase lag-six residual
at most 1e-8; exclude proper divisors 1, 2 and 3 at a maximum phase gap above
1e-4; uniquely match a reference within 1e-6 after cyclic alignment. Report
correct, wrong or unresolved for every completed pulse. Count a round trip as
successful only if both writes are correct. Execution errors stop the attempt;
not-started and incomplete rows remain in its inventory.

The maximum is 4392 model calls, including 24 prefix preparations and 48
comparison calls. The preceding 1464-call run took about 656 seconds, predicting
about 33 minutes at unchanged throughput. Bound this attempt at 2400 seconds
internally and 2460 seconds externally, with a 30-second termination grace,
20 GiB peak RSS and 24 GiB virtual memory. Use the standing Lightsail server
and the shared experiment lock; no rental or paid review is involved.

Expected raw return is about 123 MB from the preceding per-write measurement.
Reserve 512 MiB in the shared transfer ledger for source, outputs, overhead
and retry margin. Return closed write shards incrementally; verify a complete
remote inventory locally before releasing the reservation. Keep all remote
evidence. Fresh load, memory, disk and traffic checks precede launch.

The saved-array audit checks all stage comparisons, checkpoint blocks,
starting phases and exact inter-write continuity independently of the producer
classifier. A passing result would support only these six two-write sequences.
Longer sequences, other initial cycles, disturbance robustness, basin geometry,
language interfaces, logic and concealment remain separate questions.
