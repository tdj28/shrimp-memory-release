# Eight phase-mixed initial states at the parent hold

Exploratory state-space probe: use x(w)=(1-w) ad[0]+w td[3] for weights
1/9 through 8/9, in increasing order, from the pinned reference archive.
The different phases make this a line between specific orbit points, not an
ordering of parameter controls. Do not assign target bits beforehand or select
points based on their subsequent outcomes. This is not a random basin sample
or a test of fractality.

The unchanged receiver evolves each of the eight states for exactly 120 common
hold updates. Its packet contains only states. Audit every trajectory and cache
comparison, then label an endpoint only if it passes the existing period-six
closure/divisor gates and uniquely matches a reference. Preserve all unresolved
states without horizon extensions. Compare qualified labels with both static
reference proximity and nearest-phase decoding without a cutoff. Report the
number of qualified points separately from agreement conditional on qualification.

One CPU thread; 977 calls; 600 seconds internal/660 external plus 30 seconds
grace; 8 GiB RSS/16 GiB virtual memory. Use an isolated source checkout in a free
slot after fresh headroom and shared traffic checks. Reserve 512 MiB and verify
full data return before closing it. The existing native float64 implementation
and per-state cache comparisons are retained; earlier finite-difference probes
are local evidence, not a precision proof for every new line point.

This probe asks whether selected points between differently phased reference
states converge within the fixed horizon, and whether simple static geometry
predicts their labels. It does not establish a writable cell bank, a language
interface, secrecy, or that a decoder needs the dynamics.
