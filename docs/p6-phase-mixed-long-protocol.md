# Fixed longer-horizon replay of all phase-mixed starts

Adapted exploration after the 120-update line probe: one of eight points
qualified the ad cycle, at weight 7/9, despite nearest-reference prediction td;
the other seven failed recurrence. Keep those outcomes and the original stopping
rule intact. This new experiment starts afresh from all eight original states
and runs exactly 600 updates each. It is not a continuation from their saved
endpoints, and it must not replace the short-horizon result in summaries.

Use the unchanged eight-point ad[0] to td[3] line and pinned reference archive.
No point selection, target labels or parameter adjustment. Test whether the
longer, separately fixed horizon resolves slow transients; do not extend it
again merely to obtain a preferred result. Audit every point and retain wrong
geometric predictions and unresolved recurrence. Compare the qualified counts
at both horizons, and static nearest-reference agreement only among qualified
states; do not present the latter as unconditional accuracy.

Use the native float64 CPU receiver with --steps 600: 4817 model calls,
2700 seconds internal, 2760 seconds external plus 30 seconds termination grace,
8 GiB RSS, 16 GiB virtual memory, one thread. Expected wall time is about
36 minutes from measured map throughput. New isolated checkout, free first slot,
fresh memory/load and shared transfer checks; 512 MiB reservation for full raw
return. Preserve every 12-update block, final trajectory and cache comparison.
Verify all returned files before ledger closeout. Use a return supervisor that
opens its own durable log and records errors because earlier inherited logs
were redirected and closeout required manual recovery.

Any basin conclusion is limited to these eight points and finite horizons.
A nearest-reference mismatch alone does not demonstrate secrecy, a language
channel, fractality, or a reliable encoding scheme.
