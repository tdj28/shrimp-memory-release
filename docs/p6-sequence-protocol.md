# Fixed byte sequence in one numerical register

Exploratory, after four isolated writes passed and while the six-phase round-trip
batch is still finishing. Map ad to zero and td to one. Starting once from ad
reference phase zero, request exactly `10110001`: eight pulses, each with the
existing 120 endpoint updates followed by 240 common-hold updates. Carry the
actual final state into every next pulse, including after wrong or unresolved
readouts. Never reset, substitute a reference, extend a horizon or replace a
failure. Preserve all intermediate states and cache/full checks.

The independent saved-array audit verifies the fixed order, all seven links,
all stage links and every existing closure/divisor/reference gate. A failed
closure remains unresolved; no nearest-reference forced bit is allowed. This
is a serial register test, not transmission between two instances, a language
channel, a secrecy test or evidence for general sequence reliability.

Use the standing Lightsail server, native float64 and one thread, with 2928
model calls maximum, 1600 seconds internal and 1660 seconds external plus 30
seconds termination grace. Expected runtime is about 22 minutes from current
measurements. Retain the existing 20 GiB RSS / 24 GiB virtual-memory caps; measured
peak has been about 5.5 GB. Check available memory before launch, use a separate
source checkout and a free slot lock, with at most two model processes overall.
Reserve 512 MiB in the shared ledger, enforce the conservative small-run traffic
ceiling, incrementally return all eight closed writes and verify the complete
inventory before closing the reservation. No new rental or weight transfer.
