# Direct and gradual approach to a merging region

This exploratory comparison follows the unresolved periodic-root search at
N, fraction 0.78125 along H to WRITE0. At P, fraction 0.75, two distinct
period-six histories were qualified. Their original terminal states, not
newly fitted roots or states selected from this experiment, start each arm.

Both histories receive the same controls in a batch. The three fixed arms are:

| Arm | Approach | Return |
|---|---|---|
| Direct short | N for 240 updates | P for 240 updates |
| Direct long | N for 960 updates | P for 240 updates |
| Staircase | Fractions 0.7578125, 0.765625, 0.7734375, 0.78125, each for 240 updates | P for 240 updates |

The long direct and staircase arms have equal elapsed update counts; the short
direct and staircase arms have equal dwell at N. This separates two simple
route/dwell explanations without pretending that all exposures are identical.
Exact state is carried within each arm. Both original P states are restored
only between arms. There are 2,880 batch updates, nine prefix preparations,
18 cached/full comparisons and at most 2,907 model calls. Stop at 3,600 seconds,
8 GiB RSS or 512 MiB raw output. Use one CPU thread and effective float64.

At every stage, retain recurrence residuals and the phase-aligned distance
between the two terminal six-cycles. A primitive-period-six gate requires
closure at most 1e-8 and proper-divisor gaps greater than 1e-4. Call the pair
distinct only when both qualify and their minimum phase-aligned maximum L2
distance exceeds 1e-4. Distances at most 1e-6 are reported as the same within
tolerance; the intervening band remains intermediate, not a merger. At the
final P stage additionally compare each lane
with each original P cycle at distance 1e-6. Off-P reference labels are only
descriptive; they are not branch identities or bit values.

Preserve every state, token and duration, per-call probability hashes, full
cached/full diagnostic probabilities, and raw failed calls under the existing
compact mechanism-runner contract. A cached/full disagreement stops updates;
operational failures are not numerical nulls. All finite wrong, unresolved,
merged and distinct outcomes are retained without retries or extensions.

If a gradual approach retains qualified separation at N while a direct jump
does not, these finite trajectories demonstrate route dependence. If both
merge, that does not prove the second branch ceases to exist. Stability,
continuous branch identity, folds and the causal mechanism of the nominal
writer remain separate questions. This run does not encode private payloads
or communicate between agents.
