# Shorter writes from repaired states

Adapted exploratory test after all sixteen nominal 120-update next writes
from saved repaired states succeeded. Reuse exactly the same four literal
input batches per target, with both historical phase origins. Halve only the
new endpoint pulse to 60 updates; retain the full 240-update hold and unchanged
terminal readout. Preserve every wrong, unresolved and incomplete outcome.

The matched historical comparator is the completed public-repair-handoff
experiment. It used identical input bytes and nominal control coordinates.
The new experiment tests pulse duration, not a contemporaneous runtime speedup.
No initial state is selected using its new outcome, and no reference reset or
extra settling is allowed. Both requested bits are tested.

Each process has four groups of two states, eight rows, eight stages,
1,200 batch updates and 24 prefix/comparison calls: 1,224 model calls total.
Historical paths remain separately accounted; full path lengths are
612, 816, 816 and 1,020 updates. New work per row is 300 updates.

The working limits remain conservative: one effective-float64 CPU thread,
1,800 seconds internally, 2,100 seconds externally plus 30 seconds grace,
8 GiB RSS, 16 GiB address space, 256 MiB raw evidence including an 8 MiB
finalization reserve, 352 MiB archive and 448 MiB transfer reservation.
At most two model processes run on the standing server. Require fresh capacity
and transfer checks, isolated committed sources, actual-input preflight and
ready verified-return workers. No new rental or model-weight transfer is needed.

Input packets and their hashes remain those of the longer-pulse experiment.
Their recorded prospective 360-update continuation describes that earlier
experiment; this profile explicitly declares the new 300-update continuation.
The historical input-state extraction and its provenance do not change.
