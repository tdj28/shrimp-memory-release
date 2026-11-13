# Tenfold transverse displacement

Status: the 0.01-amplitude experiment is complete and independently audited
([receipt](receipts/transverse-amplitude.json)). The first transverse
experiment retained all eight labels at relative amplitude 0.001, and the frozen
42-example nearest-neighbor library predicted all eight. Those outcomes were
seen before choosing this follow-up. The question is whether retention and
static prediction survive a tenfold larger displacement, with near-orbit
controls to distinguish interior fragility from ordinary attraction.

Use exactly the same reference cycles, four anchors, fixed orthogonal direction
(seed 2026102603), signs, order, 42-example historical library and two decoders
as [the original protocol](transverse-readout-protocol.md). Change only the
relative amplitude to 0.01 with `prepare_transverse_readout.py
--relative-epsilon 0.01`. Do not add the eight recent outcomes to the library.
Verify the resulting packet differs from every earlier small-amplitude input,
as well as the historical states and reference phases already checked by the
preparer. Record the prior packet hash and all pairwise equality checks before
inference. Retain failures without replacement or a smaller fallback amplitude.

The receiver still gets only the eight states and the reference cycles.
Use the unchanged full-600 scalar runner and shadow observer. Bounds remain
4817 model calls, 2700 internal seconds, 2760 external seconds plus 30 seconds
grace, one CPU thread, 8 GiB RSS, 16 GiB virtual memory and a 512 MiB transfer
reservation. Check live headroom and the locked transfer ledger before launch;
use an isolated committed source checkout and verified local artifact return.

Report all four interior and all four near-orbit outcomes, intended-label
retention, qualified switches, unresolved finals, both frozen decoder scores,
first decisions and later sampled losses or switches. Report executed calls
and total time. This run skips no updates. Compare the two amplitudes as paired
exploratory observations, retaining the first result unchanged. There is no
amplitude search, seed search, rerun or selection of favorable inputs.

Retention at both amplitudes would extend the observed local tolerance at these
finite points. A failure confined to interior probes would identify fragility
relative to the controls. Static success remains evidence against requiring
simulation on the tested inputs; static failure would only defeat that fixed
library. Neither result demonstrates a robust neighborhood, fractality,
security, general channel reliability or a language-to-dynamics interface.

## Completed tenfold result

At amplitude 0.01, six of eight intended labels were retained: two of four
interior probes and all four near-orbit controls. The negative displacement
from the 7/27 anchor switched ad to td; the positive displacement from 8/27
switched td to ad. All finals qualified. Each fixed geometric decoder matched
six of eight finals; the shadow readout matched all eight, first deciding at
216,156,228,156,168,156,204,204 steps. No later sampled loss or label change
was observed. The run used all 4,817 calls and 2,146.50 seconds; no calls were
skipped. All 848 inventory files returned and the reservation closed.

## Disclosed third-amplitude adaptation

After seeing both amplitudes, add one bounded probe at relative amplitude
0.003. The purpose is to distinguish whether the two observed switches already
occur at this intermediate displacement. This changes the earlier plan's
statement that there would be no amplitude search; the extension is explicitly
adapted and does not replace either completed result. Preserve the same
anchors, signs, direction seed, full horizon, limits and original 42-example
library. Do not train the library on either new batch.

Compare all eight proposed initial states against all eight inputs from each
prior amplitude (128 equality checks), as well as the preparer's historical
checks. Fail on any exact coincidence without replacement. Both signs and all
controls remain, even though only two earlier rows switched. No further
amplitude is automatically scheduled. Three discrete amplitudes cannot
establish monotonicity, a unique basin boundary or fractality.
