# Separate-process numerical readback pilot

Exploratory follow-up to the successful eight-write sequence. Export each
qualified terminal hold vector, permute the eight records with NumPy PCG64 seed
20261026, and send only an array of eight float64 states to a fresh model process.
Keep the evaluation key (labels and original order) locally. This is data-flow
isolation, not a hostile-process or operating-system security boundary.

The receiver uses the pinned model, prefix, steering vector and common P6 hold
controls. It receives no write schedule or reference cycles. Each record gets
120 updates, preceded by a cached/full-prefix comparison. It saves trajectories
without predicting bits. No endpoint intervention or reset to a reference is
allowed. Native float64 normalization and rotary patches are retained; earlier
precision probes establish limited local evidence, not a global precision proof.

A local independent evaluator checks every hash, initial vector, block and
cache comparison, then applies the existing period-six recurrence and reference
match gates. Wrong and unresolved outcomes stay explicit. Reassemble the source
order only after decoding. Also decode each received vector directly by nearest
reference phase, requiring a unique distance at most 1e-6. Static decoding does
not establish recurrence. If both methods work, do not attribute information
recovery uniquely to dynamical evolution.

Bounds: one CPU thread, 977 model calls, 600 seconds internal, 660 seconds
external plus 30 seconds termination grace, 8 GiB RSS and 16 GiB virtual memory.
Use the free second process slot only after fresh memory and traffic checks.
Reserve 512 MiB for source upload, complete data return and retries; verify all
returned hashes before closing the reservation. No new rental.

Success would establish reproducible readback of explicit numerical states by
a separately loaded instance. It would not establish an ordinary-language
channel, hidden communication, basin fractality, logic gates, or improved
capacity or concealment from dynamics. This pilot reuses selected successful
sender states and is not a reliability estimate on fresh messages.
