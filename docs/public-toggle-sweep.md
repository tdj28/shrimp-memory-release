# Can the Qwen two-cycle register compute XOR? Toggle sweep, then two-loop gates

Plan written 9 November 2026 at the human's request ("let's test whether the Qwen two-cycle loop can compute XOR, and
update the film and blog only if a result passes audit"; approved the same day: "do it approved go go go go").

## Why this question

The audited logic (public-mechanism, public-mechanism-followup) uses two moves on a stored bit: hold, and reset to a
fixed value. Compositions of holds and resets are holds and resets, so XOR with an incoming bit needs a move that maps
cycle 0 to cycle 1 and cycle 1 to cycle 0: a toggle. The 29 October toggle pilot (docs/public-toggle-pilot.md) tried
eight pulses (four diagonal corners, 12 and 36 updates) and found none: six identity, two reset0. This plan widens that
search, then tries a two-loop construction if no single-loop toggle exists.

Prior expectation, stated before running: a slow excursion that returns to H across the cusp fold can only set or
hold, because the fold surface of a cusp has no monodromy. A toggle, if any, needs a fast or chaotic excursion whose
landing depends on the starting cycle, and such outcomes tend to be phase- or rounding-sensitive. A negative sweep is
the expected result and is worth recording.

## Stage 1: the toggle sweep (one loop)

Instrument: exactly the pilot's (tools/run_public_toggle_pilot.py): pinned CPU Qwen2.5-0.5B-Instruct map, effective
float64 normalization and rotary operations, math attention, one thread, batch size two (the two lanes start from the
literal reference cycles `references[0,0]` and `references[1,0]`), cached/full comparisons before updates, the neutral
cycle-matching readout after a 240-update hold at H, and the pilot's pair classification (identity, toggle, reset0,
reset1, unresolved). No adaptive choice: the candidate list below is fixed before any model call.

Candidates (`tools/run_public_toggle_sweep.py`, profile `public-toggle-sweep`):

- Units: `da = E1.alpha - H.alpha`, `dt = E0.log10_temperature - H.log10_temperature` (the audited write offsets).
- Targets: 12 directions (0, 30, ..., 330 degrees) x radii 0.5, 1, 2, 3.5, at
  `alpha = H.alpha + r cos(theta) da`, `log10_temperature = H.log10_temperature + r sin(theta) dt`. Radius 1 at 0 and
  90 degrees reproduces E1 and E0; radius 2-3.5 reaches past the fold and flip curves and the located cusp.
- Dwells: 12, 48 and 120 updates, each followed by 240 updates at H.
- Baselines first: H, E0 and E1 for 120 updates (expected identity, reset0, reset1).
- 147 groups, 45,162 model calls (about 7.4 CPU-hours at the pilot's 0.59 s per call).

Execution: groups are independent (each starts from the same literal references), so the fixed list is split into
`--shards N` interleaved shards (`group_id % N`), one single-thread process each. Each shard writes its own closed
evidence directory. Before the sweep, a parity gate reruns the three baselines and must reproduce the pilot's archived
baseline terminal states byte for byte; any difference stops the run.

Reporting: every candidate's mapping, drawn on the real kite map (fold and flip curves, cusp, H). A toggle candidate is
only a candidate. To count, it must also pass Stage 1b.

Stage 1b (only for toggle candidates): the same pulse from all six phases of each reference cycle; a 3 x 3
neighbourhood in radius and angle and the two neighbouring dwells; a rounding check (initial state and controls
perturbed by 1e-15 relative in three fixed random directions); an independent rerun; the repo audit.

### Stage 1b rules, fixed 10 November 2026 00:40 UTC (two candidates seen, before any check ran)

For every Stage 1 toggle candidate (theta, r, d), `tools/run_public_toggle_check.py` runs, with the same instrument:

- phases: both lanes start at reference phase k = 0..5 (6 groups; phase 0 repeats Stage 1);
- neighbourhood: theta +-10 degrees x r x {0.9, 1, 1.1} (8 neighbours), and dwells d +- d/3 at the centre (2 groups);
- rounding: the centre at phase 0 with each lane's start multiplied elementwise by (1 + 1e-15 u), u = +-1 from seeds 1, 2, 3
  (3 groups).

A candidate counts as a robust toggle only if it toggles at all 6 phases, under all 3 perturbations, and in an
independent rerun on a second host, and if at least 4 of its 8 neighbours also toggle. Everything else is reported as
measured (a narrow or phase-dependent toggle is a result, not a pass).

### Stage 1c, only if a candidate passes Stage 1b (fixed 10 Nov 2026 00:50 UTC, before any Stage 1b result)

A robust toggle T gives XOR in one loop: the incoming bit B selects T (B = 1) or the same number of updates at H
(B = 0), each followed by 240 updates at H; the stored bit becomes A XOR B. Demonstration, in the style of the audited
carried byte: the fixed incoming sequence B = 1, 0, 1, 1, 0, 0, 1, 0, applied from both reference starts (A = 0 and
A = 1, phase 0) with the state carried literally from stage to stage. Expected stored bits after each stage:
A = 0 -> 1, 1, 0, 1, 1, 1, 0, 0; A = 1 -> 0, 0, 1, 0, 0, 0, 1, 1. It passes only if all 16 readouts match, on two
hosts. This covers all four (A, B) cases several times each, including toggling twice in a row.

### Post-hoc deviation, recorded 10 Nov 2026 01:30 UTC (after partial Stage 1b results)

Stage 1b so far: the 30 and 150 degree candidates toggle only from phase 0 (other phases reset or hold), so they fail the
pre-registered six-phase rule; the 30 degree toggle survived all three 1e-15 perturbations. Both toggled lanes land at
the same alignment (best shift 1) as the untouched hold baseline, i.e. phase-locked: pulse 12 + hold 240 = 252 updates,
a multiple of six. A fixed schedule may therefore always meet the pulse at phase 0. This was not foreseen when the rules
were written, so the Stage 1c chain is run for the 30 degree candidate as an explicitly post-hoc test, on two hosts, and
reported as such. It does not change the Stage 1b verdict (no candidate is phase-independent).

Parity note (10 Nov 00:35 UTC): on the pod, the three baselines reproduced the pilot's mappings, but not its terminal
state bytes; the largest elementwise difference was 1.4e-15, 7.6e-16 and 5.1e-15 after 366 updates. The byte-for-byte
gate as written failed across hardware; the run continued with mapping parity and the measured differences recorded.

## Stage 2 (only if Stage 1 finds no robust toggle): two loops

1. NOT gate. Loop A drives loop B's dials through one fixed linear rule, applied at every step with no thresholds in
   code: `p_B = H + g (r - r_mid)(E0 - E1)/2`, where r is A's state projected on the direction separating the two
   reference cycles, averaged over one period, and r_mid is the midpoint of the two references' r (both fixed from the
   references before any run). 120 coupled updates, then B decoupled and held at H for 240. All four (A, B) starts;
   g, dwell and phase varied on a fixed grid. The logic lives in the wiring (the sign of g and a linear readout
   computed by the program) plus each loop's bistability; the write-up says so.
2. COPY gate. Ordinary synchronizing coupling `(1-c) F(x_A) + c F(x_N)` for one write, then hold at H: does A land on
   N's cycle? (The earlier study showed synchronization, not that it preserves the bit.)
3. XOR, only if NOT and COPY pass: A holds the bit; N keeps NOT A through the inverter; the incoming bit B selects
   hold A, or copy N into A. N is decoupled and parked at H during the copy (otherwise inverter plus copy-back forms a
   ring oscillator), then recoupled. All four (A, B), each from both starting cycles, state carried literally.
4. Mirrored-state coupling only if the inverter fails, said explicitly.

## Standards

Independent rerun and repo audit; receipt in docs/receipts; rounding sensitivity; model calls, time and peak memory
counted; finite demonstrations, not reliability estimates; the endpoints are instructed programs. Withheld as before:
runner code, prompt prefixes, steering vectors, sentence bank.

## Compute

The local machine is shared and currently saturated; Lightsail did not answer (9 Nov). The sweep runs on one RunPod pod
chosen for its CPU count, under the 1 November "within reason" authorization and the human's 9 November approval, with
a hard ceiling recorded in the private cost ledger. All outputs return to /archive/shrimp-memory-data/audits/
public-toggle-sweep/<UTC>/; the pod is terminated and its termination verified.

## Publication rule

If and only if a toggle or XOR passes audit: update the paper (paper/main.tex), the blog's "precise limit" paragraph
with a receipt-backed figure, and the film's c21. Otherwise change nothing in the paper, blog or film, and record the
negative result here, in docs/receipts and in the explainer's RUN.md.

## Results (10 November 2026)

- Stage 1: 144 probes: 79 identity, 30 reset0, 26 reset1, 9 toggle, 0 unresolved; baselines as expected. Audit passed
  on both hosts; 39 groups on two hosts agree in mapping, none in terminal bytes.
- Stage 1b: no robust toggle under the rule above (phase patterns and neighbours in the receipt).
- Stage 1c (post-hoc, see the deviation note): XOR chain 8/8 stages, 16/16 readouts, on the pod and on the local
  machine; stage-end states of the two hosts within 1.7e-14.
- Stage 2 was not run: a single loop already supplies XOR on the phase-keeping schedule.
- Receipt: docs/receipts/public-toggle-xor.json.

## Stage 3: a four-bit adder in one register (fixed 10 Nov 2026 03:40 UTC, before any run)

Human request (10 Nov): a demo that uses AND, OR, NOT and XOR together, "more than just a toy". Moves: T = the
Stage 1c toggle (30 deg, radius 3.5, 12 updates), W = 12 idle updates at H, E0/E1 = the audited writes (120 updates);
every move is followed by 240 updates at H and a readout. The register holds the carry c. For each bit position,
least significant first, with incoming digits x and y:

1. x = 1: T, else W                       register = c XOR x
2. y = 1: T, else W                       register = c XOR x XOR y = the sum bit (read)
3. x = y: write E_x, else T               register = majority(x, y, c) = the next carry

Step 3 is the carry rule of a full adder: when the digits agree the carry is that digit (x AND y, which equals x OR y
there); when they differ the carry is c = NOT(sum bit). The controller chooses every move from x and y only and never
reads the register. Both lanes share every move: lane 0 starts on cycle 0 (carry in 0), lane 1 on cycle 1 (carry in 1).

Fixed case: X = 11 (1011), Y = 6 (0110), LSB first. Moves: T W T | T T E1 | W T T | T W T. Expected register after
each move (computed from the rules above): lane 0: 1 1 0 | 1 0 1 | 1 0 1 | 0 0 1; lane 1: 0 0 1 | 0 1 1 | 1 0 1 | 0 0 1; sum bits
1 0 0 0 and carry out 1 = 17 (lane 0), 0 1 0 0 and 1 = 18 (lane 1). Pass: all 12 readouts on each lane match, on
two hosts. Same instrument, inputs and readout as Stages 1-1c; state carried literally.
- Stage 3 (pre-registered): the four-bit adder matched all 12 moves on both lanes, on the pod and on the local
  machine: 11 + 6 = 17 (carry in 0) and 18 (carry in 1). 3,204 calls each; stage-end states within 1.3e-14.
