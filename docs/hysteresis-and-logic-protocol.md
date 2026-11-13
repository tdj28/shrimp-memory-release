# Hysteresis mechanism and stateful logic

This is a new, public numerical investigation. The existing writes establish
controllable terminal states, not their causal bifurcation mechanism. Outcomes
must determine whether the proposed mechanism survives these tests.

## Hysteresis: staged evidence

First carry both qualified period-six reference states (phase zero) along
identical outward-and-return paths. For each previously used write endpoint E,
start at common hold H and visit H + 0.5(E-H), E, H + 0.5(E-H), H. Each point
receives 240 updates, with no reset within a loop. Reset to the two reference
starts only before testing the other endpoint direction. Preserve every step,
fresh cache comparison, effective-precision observation and intermediate
readout. The initial pilot has two groups, eight stages and 1,944 calls.

Reference matching away from H is descriptive only: an attractor can move
without disappearing. Analyze recurrence, primitive period, cyclic separation
between the two carried histories and convergence of each trajectory. Do not
classify a failed reference match as loss of a branch. Unresolved trajectories
remain unresolved, not chaotic by default.

A path-dependent loop alone is insufficient. Repeat the declared path with
480 updates per point if resource and evidence checks permit, keeping the
240-update result. If the loop disappears with dwell, transient lag remains a
candidate explanation. If stable branch separation and repeatable switching
persist, refine the affected parameter interval and test local orbit stability
on each branch. A numerical fold/disappearance claim requires continuation and
stability evidence; merely choosing different basins is a different mechanism.
The two write directions and common-H endpoints supply matched controls. A
later off-window control is required before claiming shrimp-specific logic.

No outcome of this coarse first pilot alone establishes that hysteresis causes
the existing writes. It locates the next discriminating mechanism test.

## Logic: truth tables and carried-state composition

The first input A is encoded only in the initial dynamical state, with both
reference states run in one batch. For a stateful AND operation, input B alone
selects either the fixed WRITE0 endpoint (B=0) or common hold (B=1), for120
updates, followed by240 common-hold updates. The controller never reads A or
an intermediate decoded label when choosing controls. This is a conditional
reset/hold gate, not a claim of autonomous combinational hardware.

For each (B,C) in (0,0),(0,1),(1,0),(1,1), initialize A=0 and A=1, apply the
first gate and then a second gate with C. Carry the exact terminal state into
the second gate; do not restore a reference or insert extra settling. Save both
intermediate and final readouts. This covers the complete AND truth table and
all eight A,B,C combinations for two-gate composition. The four groups have
sixteen stages and 2,928 calls. Phase zero is the initial exploratory scope;
other phases and disturbed states are follow-up tests, not implied success.

An OR profile uses hold when B=0 and WRITE1 when B=1, with the same independently
encoded inputs and composition panel. It is implemented for a subsequent run,
not implicitly launched. XOR requires a common toggle intervention that maps
both stored states to their opposites. Reset operations or external computation
of A XOR B do not qualify. A complete successful truth table precedes a claim
of a gate, and carried-state chains precede a composability claim.

## Limits and evidence

The initial pair consists only of the240-dwell hysteresis pilot and AND chain,
one effective-float64 CPU thread each. Each process has a3,600-second internal
limit,3,900-second external limit plus30seconds grace,8GiB RSS and16GiB virtual
memory. Raw evidence is capped at512MiB including16MiB finalization reserve;
archive768MiB and transfer reservation1GiB per process. Source upload8MiB.
Fresh capacity and transfer admission, isolated committed checkouts, model-free
actual-input preflight and ready incremental/final-return workers precede launch.
All raw data returns to the local sibling data root. No new rental is needed.
Failed/incomplete attempts and every truth-table entry remain in the record.
No human-auditing evasion or hidden communication is tested.

## Saved-data analysis declared during the initial run

The independent checker tests periods1 through64 using the last two full
candidate cycles. Closure requires maximum phase L2 residual at most1e-8;
each proper divisor must have maximum cyclic gap greater than1e-4. It reports
all candidate tests, including unresolved tails. This finite-tail test is not
an orbit-stability test. Aligned cycle gaps minimize the maximum phase L2 gap
over cyclic shifts, and are compared only when qualified periods agree.
At each control point, report both carried histories; at the repeated middle
control, compare outward and return histories separately for each initial bit.
Reference labels away from H remain descriptive. For logic, strict reference
readout is recomputed at both common-hold outputs for every A,B,C row, retaining
wrong and unresolved outputs. This analysis was specified after launch but
before inspecting numerical stage observations; it is exploratory.

## Follow-up pair prepared 2026-10-28T20:36:04Z

The completed240-update loop retained separated period-six histories at the
WRITE0 midpoint and unresolved off-hold tails in the WRITE1 direction. The
completed AND chain recovered both outputs for all eight A,B,C combinations.
The next exploratory pair therefore doubles each loop dwell to480 updates
(3864calls) and tests the complementary OR chain at the existing120/240 schedule
(2928calls). These are the already specified paths and controls; the longer
dwell tests whether transient lag explains the observed separation. Neither
profile alone establishes a bifurcation mechanism or general gate reliability.

Both jobs retain the same3600second internal and3900second external limits,
plus30second termination grace,8GiB RSS,16GiB virtual memory,512MiB raw,768MiB
archive and1GiB return reservation each. The immutable group transfer cap is
320MiB. At most two one-thread model processes may run. The byte-qualification
pair must first complete, return all evidence, close both reservations and
pass its independent audits. A bounded, single-use continuation may then launch
this pair only from its frozen clean commit after fresh capacity and traffic
admission. Failed preconditions stop the continuation without retries. There
is no automatic experiment beyond these two jobs.

Every OR truth-table row and every loop stage receives independent saved-data
checks. Report wrong, unresolved and incomplete states as observed. Compare the
480-dwell loop with the retained240-dwell evidence before proposing interval
refinement or branch-stability tests. No new rental, paid review or language
concealment experiment is part of this follow-up.

## Spatial follow-up declared 2026-10-28T21:40:11Z

At480 updates, both directions retained distinct outward midpoint period-six
states and merged on returning from the endpoint. The WRITE1 endpoint itself
remained recurrence-unresolved through period64; its two midpoint visits
qualified. Thus the longer dwell does not remove the observed history effect,
but does not establish loss of an attractor or local stability.

The next exploratory pair changes only the repeated interior control fraction
from0.5 to0.25 or0.75 of the straight H-to-endpoint segment. Each process tests
both WRITE0 and WRITE1 directions, from both phase-zero reference starts, at
480 updates per point. The sequence is H to the specified fraction, endpoint,
the same fraction again, then H. There is no state reset within a loop. Each
process has3864calls; all existing time, memory, raw-data and transfer limits
remain unchanged. The aim is to locate where separated recurrent histories
persist, providing a spatial bracket for later continuation/stability work.
No finite-tail separation alone will be called a fold or a causal proof.

This design adapts to the already observed0.5-fraction results. Both spatial
fractions and both directions will be retained regardless of outcome. Fresh
capacity admission, exact committed source, actual-input preflight, tested
shard packaging and ready return workers are required before launch. No further
experiment is automatically queued after this pair.

## Finite disturbance follow-up declared 2026-10-29T03:57:28Z

The three-quarter WRITE0 point retained two separated period-six histories
after 480 updates. Test their finite-horizon response to one small state
disturbance before attributing local attraction to either branch. This is an
adapted exploratory selection from the existing fraction-loop evidence.

Use the exact terminal pair from stage 0 of the independently audited
three-quarter loop. Form three independent groups: unperturbed, positive
disturbance and negative disturbance. Both histories receive the same unit
direction drawn from PCG64 with seed 20261029. The disturbance norm is 1e-5
times the RMS L2 norm of the final six phases of both source histories
(12 state vectors). Fix alpha 0.07293697809562831 and log10 temperature
-0.15361287955001818. Run 240 updates per group, without resets within a
group, for 729 model calls including the prefix and cache comparisons.

Each fresh undisturbed history must qualify as primitive period six, and
the pair must remain separated. Evaluate each disturbed history against
its corresponding fresh undisturbed cycle using the existing maximum
phase-aligned L2 gap. Return requires primitive period-six recurrence and
gap at most 1e-8. Report switches, wrong periods and unresolved tails
separately. The one direction, one amplitude and fixed 240-update horizon
do not establish local stability, basin volume, a fold, causal hysteresis
or an advantage over controls outside the candidate region.

Launch one effective-float64, one-thread process from an isolated committed
checkout only after actual-input preflight and a ready return worker. Bounds
are 900 seconds internal, 1080 seconds external plus 30 seconds termination grace,
8 GiB RSS, 16 GiB virtual memory, 160 MiB raw evidence, 224 MiB archive and 320 MiB
transfer reservation. Preserve every outcome and do not extend or retry
to obtain a preferred result. Keep operator-owned frontier trials separate
with the idle maintenance interlock; no frontier API calls are part of this run.


## Stability measurement prerequisite declared 2026-10-29T04:53:03Z

Following the adjudicated Pro consult, first test the six-step product and
small-subspace estimator on synthetic stable, unstable, non-normal and
noncommuting six-phase systems. Include an invariant-start example that misses
an unstable mode and a complex-pair case; a small Ritz residual is not spectral
coverage. These are model-free instrument checks, not neural stability evidence.

Then reuse the existing scalar effective-float64 derivative validator on each
literal terminal history from the three-quarter WRITE0 packet. Both use the
fixed original seed-zero unit tangent, controls alpha0.07293697809562831 and
log10_temperature-0.15361287955001818, and finite-difference steps0.01,0.005,
0.0025,0.00125,0.0001,0.00001. This is19calls per history (two prefixes, stock
base, native base/repeat/full, one JVP and twelve finite differences),38total.
Run the two model processes sequentially, one CPU thread each, with a300second
internal limit each, one900second outer limit plus30second grace,8GiB RSS and
16GiB virtual memory. Return all raw arrays including failed rows. No retry or
post-outcome step-size tuning belongs to this design. The existing quadratic
halving gate and1e-6 relative error criterion remain unchanged.

Use the previously audited literal input pair, not a newly selected phase.
Validate source/input hashes and extraction without model loading before launch.
The two derivative checks test one state/direction per history, not all phases,
monodromy multipliers, orbit stability or causal hysteresis. If either check
fails, report its error curve and preserve the result before a disclosed
follow-up. Use an isolated committed checkout,160MiB raw/224MiB archive caps,
320MiB return reservation,8MiB upload cap and the idle observatory interlock.
A ready bounded completion worker must precede model execution.


## Composed derivative prerequisite declared 2026-10-29T05:24:36Z

Adapt the passed two-state one-step checks to the six-update map F^6. Keep the
same literal terminal states, controls, seed-zero unit tangent and finite-difference ladder of six step sizes. Each composition carries the full primal state and
unnormalized tangent; do not reset to saved phases. Evaluate stock, native,
repeat, full-prefix, JVP and twelve plus/minus difference trajectories, each
for six updates. Two prefix preparations plus17times6forwards give104calls
perhistory208total. Six-step derivative agreement is not a Floquet spectrum
or a proof of attraction; closure and spectral coverage remain untested.

Retain the preexisting derivative gate (at least two consecutive halving orders
between1.5and2.5 and minimum relative error<=1e-6), exact repeat, and
cached/full-primal tolerance1e-10. The independent auditor checks every
intermediate chain link, tangent link and all six cached/full differences,
not just final arrays. Preserve failed ladders without adapting the steps
to obtain a pass. All endpoint differences and elapsed costs remain raw.

Two sequential one-thread processes:480seconds internal each,510external each,
1200seconds outer plus30termination grace,8GiBRSS16GiBvirtual memory.
384MiBraw448MiBarchive640MiBtransfer reservation,8MiBsource/input upload.
Use a new isolated committed checkout, fresh capacity/ownership/traffic
admission, existing idle observatory interlock and ready bounded return worker.
Return failed/incomplete results too. No frontier trial or paid API is launched
by this numerical test; the separately authorized Pro consultation continues.


## Smaller-step diagnostic amendment declared 2026-10-29T05:32:58Z

The original six-update ladder failed the minimum relative accuracy gate on
both histories (2.4169371902290213e-6 and3.698675554295451e-6), despite repeated
primal and cache gates passing. Independent audit verified all232files and
retained both failures. Between h1e-4 and1e-5, errors decreased approximately
100fold. Test the specific hypothesis that finite-difference truncation is
still dominant, versus a discrepancy floor, with one newly declared ladder:
1e-5,5e-6,2.5e-6,1.25e-6,6.25e-7,3.125e-7. All other states, directions, controls,
104calls perhistory, gates and resource bounds remain fixed. The auditor names
this profile `smaller-step-truncation-diagnostic`; the original remains
`original-six-size-ladder`. No old output or acceptance threshold changes.

The shared h1e-5 anchor must reproduce its raw error within relative1e-6
(or absolute1e-12) before interpreting cross-run trends. A continued quadratic
trend with minimum<=1e-6 qualifies this sampled derivative; a plateau or failed
order/accuracy remains unresolved and blocks spectral interpretation. No
automatic further refinement or retry. This is an adapted instrument diagnosis,
not independent replication, stability evidence or a rescue of the first run.

## Approach to the actual WRITE0 endpoint

The nominal writer jumps directly to its endpoint. History loss at an
intermediate control that it never visits cannot by itself explain that write.
The next adapted exploratory test inserts one prescribed approach point before
the actual endpoint, seeking surviving endpoint states for a later arrival-state
intervention. This is stage A only; it does not authorize an automatic refinement
or a subsequent pulse or disturbance campaign.

Start with the same literal two-history terminal pair at the three-quarter
WRITE0 point. Carry both states through 480 updates at fraction 0.875, then
through 480 updates at the actual WRITE0 endpoint. Alpha stays
0.07293697809562831; the new log10 temperatures are respectively
-0.15355643977500907 and -0.1535. There is no intervening reset, phase selection
or replacement by a reference state. The numerical history labels are inherited
from the initial pair and are not reassigned by nearest common-hold label.

After each stage, inspect both saved trajectories for periods 1 through 64.
The planned period-six path requires maximum terminal phase recurrence at most
1e-8, proper-divisor phase gaps greater than 1e-4, and phase-aligned pair
separation greater than 1e-4. Save the complete two-by-two distance table to the
previous cycles. For each lane, require its distance to its own previous cycle
plus 1e-4 to be smaller than its distance to the other previous cycle. This is
a conservative geometric tracking rule, not proof of continuous branch
identity. Merging, changed periods, failed recurrence or ambiguous tracking at
the inserted point stops the endpoint path and preserves the outcome.

Finding two distinct endpoint candidates would justify a separately bounded
comparison of nominal and endpoint-seeded writes. Failure to find them does
not establish branch disappearance: finite forward iteration can miss a
surviving basin. Neither outcome alone establishes causal hysteresis, a fold,
local stability or specificity to shrimp geometry.

The maximum is 966 two-lane ordinary forwards: 960 updates, two prefix
preparations and four cached/full comparison calls. There are no JVPs or model
calls hidden in readout. Use one CPU thread and effective float64, an isolated
committed source checkout and the existing precision/cache checks. Proposed
limits are 1200 seconds internally, 1260 seconds externally plus 30 seconds
termination grace, 8 GiB RSS and 16 GiB virtual memory. Admit the job only after
checking current host capacity, transfer reservations and idle human trials.
Bound this stage-A run's numerical raw evidence at 128 MiB and the complete
returned package, including source and operator logs, at 160 MiB. The archive
cap is 192 MiB, transfer reservation 640 MiB and source upload cap 8 MiB.
The transfer reservation covers two attempted stage archives and one complete
fallback archive, even without deduplication, plus upload, manifests and a
32 MiB transport reserve. An offline failure-return test rejected the earlier
256 MiB draft before launch; no scientific or numerical limit changed.
Check the actual source/input/file contracts before launch;
return each closed stage and all partial failures through a ready bounded
worker. The shared monthly transfer maximum remains 7 TB.

## One fixed midpoint approach after the direct path merged

The direct P-to-Q approach completed 480 updates at Q. Both histories passed
the finite-tail primitive-period-six checks but became numerically coincident;
the pair-separation and history-1 lineage gates failed. The conditional E0
stage was not run. The independent audit and immutable source are bound in
`receipts/public-endpoint-approach.json`. This motivates one adapted comparison,
not an automatic sequence of finer scans.

Start from the identical literal P(0.75) terminal pair, carry both histories
through 480 updates at M(0.8125), and then carry the actual states through
480 updates at Q(0.875) only if M passes every unchanged gate. Alpha stays
0.07293697809562831. M has log10 temperature -0.1535846596625136 and Q has
-0.15355643977500907, derived as H + fraction times (E0 - H). Pin the full
binary64 constants, input packet and history order. There is no new P
settling, reset, phase selection, reference-state replacement or E0 stage.

At each stage check periods 1 through 64, period-six maximum terminal phase
recurrence at most 1e-8, proper-divisor gaps greater than 1e-4, and
phase-aligned pair separation greater than 1e-4. Record all cyclic shifts
for the two-by-two geometric distance table to the previous cycles. Each
unchanged lane must be nearer its own prior cycle by the same strict 1e-4
margin. M is compared with the original P cycles; Q with the actual last
six M states. A failed gate at M closes the unrun Q marker with zero calls.
Keep individual gate failures separate: ambiguous lineage is not necessarily
a merger. Execution, precision, cache or return failures are not scientific
nulls. No failure triggers a reset, extra dwell or another midpoint.

Use the audited direct-Q run as the historical comparator, preserving its
original source and raw hashes. At Q, save an additional descriptive
two-by-two distance table to those direct-Q cycles. This table does not alter
the stopping rule or lane labels. A separated recurrent pair at Q would
demonstrate different finite outcomes along these two prescribed paths.
Another merger would not establish branch disappearance. The new path adds
480 M updates, so smaller parameter increments, extra dwell and changed
arrival phase are inseparable in this comparison. Neither outcome explains
the nominal writer's direct jump to E0 or establishes causal hysteresis.

The fixed schedule requires 483 forwards if it stops at M, or 966 forwards
if both stages execute: each stage has one prefix, two cache-comparison and
480 two-lane update calls. There are no JVPs or hidden settling calls. Keep
the prior one-thread native-float64 instrument and bounds: 1200 seconds
internal, 1260 external with 30 seconds termination grace, 1440 outer job,
8 GiB RSS, 16 GiB virtual memory, 128 MiB numerical raw, 160 MiB complete
package, 192 MiB per archive, 640 MiB reserved transfer and 8 MiB upload.
The bounded return worker has 2400 seconds including helper cleanup. Verify
fresh capacity, task ownership, idle human trials, the shared transfer ledger,
actual source-upload size and a ready return worker before launch. Test both
named schedules and actual file contracts without a model first. Retain every
outcome and partial artifact; a later intervention needs a new recorded plan.

## One fixed-M periodic-orbit correction after finite arrivals merge

The two literal P histories merged after 480 updates at both Q and M. That
does not tell us whether another periodic solution survives at M but is missed
by forward settling. The next discriminating experiment should be one bounded,
fixed-M periodic-orbit correction from the original H1 P state. It should not
add another automatic parameter midpoint or an attraction claim.

This is a proposed adapted exploratory numerical experiment, after seeing the
direct-Q and midpoint merger results. No model call was made to prepare it.
The private preparation archive supplies the creation time, source hashes,
measured timing/storage inputs and exact budget; the committed runner fixes
the experimental controls and call limits.

**Question and input.** Let G(x)=F_M^6(x), R(x)=G(x)-x, and
A_x(v)=D G(x)v-v. Solve R(x)=0 at the *single* control
alpha=0.07293697809562831, log10_temperature=-0.1535846596625136 (M, fraction
0.8125). Start at `source_terminal_states[1]` from the unchanged literal P
packet, with P at fraction 0.75. Use only that seed and phase. Use the merged M
lane-0 terminal state as a positive recurrence/cache comparator, not as another
search seed. Do not add Q, E0, another phase, another initial condition, or a
parameter derivative.

The passed midpoint audit reports period six for both histories, maximum
phase residuals 2.0047933743508787e-14 and 1.3411116280484043e-14, and
phase-aligned pair gap 1.4420307222287048e-15. H1 fails geometric lineage; Q was
not run. Its source commit is 23a34b09a3ab097bc3c0791ad21076f1ab7619be. The
existing limited spectrum concerns P, not M: it cannot establish that A_x is
invertible at M, justify the Newton basin, or certify an attracting orbit.

**Why this test.** A corrector can find a fixed point of G that forward settling
does not approach, including an unstable one. Another dwell or smaller jump
would test further arrivals but would still confound branch persistence with
basin selection. A full forward-AD Jacobian would cost 896*6=5,376 forwards at
one base, before checks, without proving global absence. Pseudo-arclength
continuation is more informative near a fold but adds parameter derivatives
and a separate branch-tracking protocol. Defer it; do not silently switch to it
when this bounded corrector fails.

**Fixed corrector.** Use at most four Newton iterations. At each current x_k:

- Keep its already recorded ordinary G(x_k) and R(x_k). Stop the corrector if
  the Euclidean residual norm is at most 1e-10. Otherwise form b=-R(x_k).
- Use unrestarted real GMRES with at most eight Arnoldi actions. Each action is
  A_x(v), implemented as exactly six forward-mode model evaluations followed
  by subtracting v. Reset the reference to the identical x_k for every action;
  carry state and tangent only *within* that six-update composition. Save all
  bases, Hessenberg coefficients, actual unmodified action outputs, least-
  squares coefficients/rank/singular values, and projected residuals. Two-pass
  orthogonalization and the existing saved-matvec decomposition check are
  reusable; projected residual alone cannot accept a Newton step.
- Evaluate A_x(delta) freshly once. Require
  ||A_x(delta)+R(x_k)||/||R(x_k)|| <= 0.1, and agreement with the saved W*y
  combination to relative 1e-10, using denominator
  max(||A_x(delta)||,||W*y||,1e-14). Record both full vectors and absolute as
  well as relative residuals. An insufficient fresh linear residual stops unresolved; inconsistent actions
  fail the instrument. Do not restart GMRES or add directions. At most one such fresh action per iteration.
- Try damping factors (1, 0.5, 0.25) in that order. Use a permanent trust ball
  centered on the original H1 P terminal state, radius
  2*||x_P,H1-x_P,H0||_2. The pinned radius is 0.005421550304749058, derived
  from the exact selected-phase gap 0.002710775152374529. An out-of-ball trial
  is recorded and skipped without a model call. Never clip or recenter it.
  Accept the first finite in-ball trial whose residual norm is at most
  (1-1e-4*lambda)*||R(x_k)||, or at most 1e-10. Reuse that accepted ordinary
  residual on the next iteration. Preserve every rejected in-ball evaluation.
- No accepted damping candidate, a singular/insufficient Krylov solve, or
  exhaustion of four iterations is a scientific unresolved outcome. A callback
  exception, nonfinite value, lost tangent, or exhausted instrument budget is
  an execution/instrument failure. Preserve them separately.

The trust radius is a disclosed search restriction, not an enclosure of a
branch. In a discrete map there is no automatic continuous-time phase-neutral
mode: do not add a phase equation or remove a multiplier just because this is
a period-six problem. Exact phase matching is bookkeeping; all-shift distances
belong in the final orbit-identity analysis, not in the residual equation.
The trust ball includes the known M solution. Keep that possible outcome;
do not add deflation or reject it in order to seek a preferred root.

**Initial 91-forward qualification.** One scalar native64 FeedbackMap at M
serves all of the following, with one shared, immutable fixed-prefix cache.
All norms below are Euclidean unless explicitly Linf.

| Initial component | Ordinary | Differentiated | Prefix |
|---|---:|---:|---:|
| Prepare the M prefix | 0 | 0 | 1 |
| M comparator: two carried six-step cycles, then full-prefix calls at its first six cached input states | 18 | 0 | 0 |
| H1 P seed: six cached steps, then full-prefix calls at those six input states | 12 | 0 | 0 |
| Seed JVP in v0, v1, and 0.6*v0-0.8*v1 | 0 | 18 | 0 |
| Seed central differences in v0: plus/minus at three h values, six steps each | 36 | 0 | 0 |
| Repeat the seed's six-step cached chain | 6 | 0 | 0 |
| Total | 72 | 18 | 1 |

Set v0 and v1 once from PCG64 seeds 0 and 1, each producing 896 float64
standard-normal entries and normalized with the same explicit NumPy norm
convention; archive the exact final arrays. Do not alternate Torch/NumPy
normalization during audit replay. Use h=(1e-5,5e-6,2.5e-6). Require minimum
central-difference relative error <=1e-6 and both successive halving orders
in [1.5,2.5], with relative denominator ||D G(x0)v0||. A zero/vanishing
denominator or zero error cannot manufacture a convergence order. No changed
h ladder is automatic after a failed check. Initial linearity must be within
1e-10 relative, denominator max(actual norm,predicted norm,1e-14).

The comparator's maximum lag-six residual over the two cycles and its
known-phase agreement with the saved M lane-0 cycle must be <=1e-10. All six
full/cache differences must be <=1e-10 Linf. All JVP primals must agree with
the corresponding ordinary G(x) within 1e-10 Linf, including every later
Newton base. Seed-repeat state arrays must match exactly. Persist fresh norm
and rotary observations, patch inventory, native64 mode, Torch intraop/interop
counts and all four one-thread environment settings. The FD gate tests the
initial M state/direction only; it is not a global derivative error bound.
It also does not bound relative error in A=DG-I near a unit multiplier, where
subtraction and ill-conditioning matter. Fresh ordinary residuals and final
replay, rather than a projected linear solve, decide numerical qualification.

The runner may pass its explicitly bound, checked repeated-seed residual to
the solver as `initial_residual`. Then the helper makes at most 12 additional
residual callbacks and 36 action callbacks. Alternatively its initial residual
callback may implement the reserved seed repeat. Never do both and silently
add six forwards. Each later accepted trial supplies the next outer residual.

**Conditional final 24 forwards.** Only if the corrector reaches its residual
threshold, take 12 ordinary carried updates from that exact candidate, six
full-prefix calls at the first cycle's cached input states, and six repeated
cached updates from the candidate. Require all phase lag-six residuals <=1e-10,
full/cache differences <=1e-10 Linf, and exact repeat. For proper divisors
1,2,3 require the maximum phase gap to exceed 1e-4; otherwise report the lower
period, not a new period-six solution. Report all six shift distances to the
known M cycle and both original P histories. A candidate is distinct from the
known M cycle only if min_shift(max_phase_L2_gap)>1e-4. Report geometric H1/H0
proximity and the old 1e-4 margin descriptively; it does not prove continuous
branch identity. Do not rotate the numerical seed or relabel histories.

**Hard call and resource envelope.** There is no finite-return/attraction probe
in this pilot. The maximum is
91 + 4*(8*6 + 6 + 3*6) + 24 = **403 model forwards**:
234 differentiated, 168 ordinary (18 full-prefix and 150 cached), and one
prefix preparation. Each composed action already returns its primal; a JVP
is not another separate ordinary forward. Shorter paths retain unused slots;
they do not authorize another seed. Actual attempted calls, including failed
calls, must be counted before execution. Preflight and independent replay of
saved arrays consume zero model calls.

The two returned P-spectrum histories measured cached-forward medians
0.4399--0.4497 s, full-prefix medians 1.1715--1.2129 s and JVP medians
4.2612--4.3030 s. Their peak RSS was below 5.234 GiB. These are sizing inputs,
not a guaranteed performance bound at M. The full pilot projects to about
19 minutes including startup/I/O at those rates, or about 23 minutes using
the largest observed per-kind times. Use one CPU thread, 8 GiB RSS, 16 GiB virtual memory, 1,800 s
internal numerical wall time, 1,860 s external with 30 s termination grace,
2,040 s outer job and 3,600 s bounded return worker. Admit from fresh measured
headroom, idle task-owned work and a verified ready return path.

Retaining the current scalar evidence format gives 1,230,590 bytes per
ordinary-call NPZ and 1,245,414 per differentiated-call NPZ. Preserve all
probability arrays: the 402 non-prefix raw call files alone total about
0.464 GiB. Propose 768 MiB numerical raw cap, 896 MiB whole package cap,
1,024 MiB archive cap, 8 MiB input/source upload cap and a conservative 4 GiB
transfer reservation covering incremental shards, full return, one complete
fallback and overhead. The endpoint runner's 128 MiB raw cap cannot be reused.
Do not download or upload weights; hash the existing pinned model at remote
preflight, as in the completed experiments.

**Model-free prerequisites before any launch.** The new pure solver and runner
contract tests must exercise these with explicit expected outcomes:

- Known stable, unstable and non-normal affine six-phase compositions: recover
  the analytical root, retain true residuals and permit transient growth in a
  stable non-normal case. Root convergence must not be labelled stability.
- A nonlinear persistent-bistability toy whose finite arrival lands on the
  other attractor, plus an unstable distinct root. Test the existence-versus-
  attraction outcome distinction, not an assumption that Newton always finds
  the intended stable branch.
- A known saddle-node residual before/at/after the fold. Missing roots,
  singular A and exhausted correction must report unresolved, never infer
  disappearance from solver failure. Include a surviving root outside the
  trust ball to demonstrate the deliberate false-negative limitation.
- Zero initial residual, happy breakdown, rank-deficient least squares,
  non-normal projected/fresh-residual disagreement, insufficient eight-vector
  coverage, lower period, wrong phase/seed, and an in-ball rejected damping
  trial. Preserve exact zero instead of taking log(0) or dividing by it.
- Mutating callbacks, wrong shape/dtype, lost tangent, nonfinite/overflow,
  corrupted cache/input bindings, and a callback that raises halfway through
  a six-step composition. Fail and retain partial inputs/actions/call counts.
- Actual orchestration filenames: qualification stop, no-damping acceptance,
  four-iteration exhaustion, root matching the known M orbit, distinct root,
  failed final qualification and execution failure. Check actual 403-call
  maximum, explicit supplied-initial-residual accounting, prefix counted once,
  no hidden retries, complete final/partial manifests, and independent closure
  of completed qualification/outer-step/final groups for early return.

Reuse `composed_feedback_action.py`, the scalar `FeedbackMap` and validated
native64 instrumentation, plus the real Arnoldi evidence ideas from
`public_spectral.py`. Do not reuse the ordinary two-lane backend as a JVP
backend, nor let the spectral `run_protocol` enforce a periodic reference at
the nonperiodic Newton seed. Prefer a new narrowly fixed root-correction
runner and a pure bounded solver over changing the finished endpoint runner.
The new runner should expose only input/output/preflight paths; M, history 1,
phase, solver limits and controls are immutable experiment constants.

Bind the original literal packet, its original source/derivative audits and
the passed M audit/result/final tail, rehashing exact repeated provenance only
once and rejecting conflicting records. The new independent checker should
rebuild each six-step chain and A(v)=Jv-v from saved outputs; verify fixed
base resets, W/H/Q and least-squares relations, both raw inner residuals,
trust/Armijo decisions, final recurrence/identity and all counts. Save code,
versions, model/steering hashes, raw states/probabilities/tangents, rejected
trials and each immutable stage close. Its pass establishes saved-evidence
consistency, not a new model replay or a mathematical certificate.

**Decision table after the pilot.**

| Outcome | Warranted conclusion | Next decision, not an automatic extension |
|---|---|---|
| Instrument, comparator or FD failure | Corrector readiness at M is unresolved | Fix the documented instrument issue before interpreting dynamics |
| Corrector exhausts, leaves trust region, or only finds the known M cycle | This single bounded search did not recover another root | Consider a separately specified continuation method; do not refine midpoints automatically |
| Corrector finds a distinct primitive P6 numerical solution | A second periodic solution at M is supported despite merger of the tested arrivals | Separately freeze a finite-return/stability test if attraction matters |
| Distinct root is geometrically closer to H1 P | Additional geometric association only | Continuous H1 branch identity still needs continuation |
| Root is lower-period or final replay/cache/closure fails | The intended second P6 candidate is not qualified | Retain the discovered/failed object without upgrading it |

A distinct root may be unstable after an attracting branch loses stability.
Thus even success does not settle basin switching versus loss of an attractor;
it separates periodic-solution persistence from failure of forward settling.
Neither a small residual nor GMRES convergence certifies a true root of an
exact real-arithmetic map. Newton convergence/invertibility at M is an
untested mathematical assumption, not supplied by the limited spectrum at P.
All outputs remain ordinary finite-precision numerical observations, with no
branch-loss, attracting-memory, global basin or endpoint claim.


## 2026-10-29 09:23 UTC: fixed signed diagnostic at the stalled second iterate

Adapted exploratory numerical work, evidence rung 0. Repository base `85286e0f9eb1d0f9a19ddc65d18df1898cce2351`. The completed fixed-M corrector accepted one half step and then rejected all three prescribed second-step trials. Before enlarging that search, test its literal second iterate and literal Newton direction. This diagnostic distinguishes local directional consistency from finite-step curvature; it cannot infer branch loss from a rejected search.

Use the literal stalled second base x and literal unnormalized saved Newton step s at M. Let G=F_M^6, R=G-I, d=DG(x)s and a=d-s. Retain all signed points x±lambda*s for lambda=2^-k, k=0,...,12. There is no adaptive grid, solver, preferred point, extra damping or root acceptance.

API: analyze_directional_ladder(*, base, step, base_output, dg_step, plus_inputs, minus_inputs, plus_outputs, minus_outputs, replay_state_l2, replay_dg_l2, replay_action_l2). Base/step/output/action are finite float64 vectors; signed arrays have shape (13,n). Inputs must equal NumPy's float64 base±lambda*step bitwise. This checks the prescribed point, while the actual displacement rounding errors are still measured. An incomplete or malformed ladder raises ValueError; the caller must retain its partial raw evidence and classify execution failure separately.

Before running the ladder, the runner remains responsible for effective-float64 observations, full patch inventory, one-thread settings, exact source/input pins, fresh cached/full base comparisons and JVP primal comparisons at every phase (Linf<=1e-10), historical base replay (Linf<=1e-10), and an exact ordinary repeat. Fresh versus historical d and a should each agree to relative1e-10, using max(new norm,old norm,1e-14). The helper does not turn a failed instrument gate into a tolerated floor. These checks retain the proposed 25 initial calls:1prefix+18ordinary+6JVP; the26 signed six-step chains add156, total181. Runner resource and return contracts remain separate prerequisites.

For each lambda compute C_D=(G_plus-G_minus)/(2lambda) and C_A=(R_plus-R_minus)/(2lambda), using the actual signed input states in R_plus/minus. Record errors C_D-d and C_A-a separately. Retain actual chord c=(x_plus-x_minus)/(2lambda), c-s, and the floating arithmetic identity gap C_A-(C_D-c). Never replace the residual central difference with C_D-s. Record both Taylor vectors T_plus=R_plus-R0-lambda*a and T_minus=R_minus-R0+lambda*a, every signed residual norm, and both linear predictions.

Freeze epsilon as float64 machine epsilon. Let rho_state be the maximum measured L2 discrepancy across fresh/historical base, cached/full, repeated base and JVP-primal comparisons; rho_D and rho_A are the measured old/fresh d and a discrepancies. Define L_lambda=max(||x||,||x_plus||,||x_minus||,||G0||,||G_plus||,||G_minus||), eta=max(rho_state,64epsilon*L_lambda). The absolute diagnostic floors are f_D=eta/lambda+rho_D and f_A=2eta/lambda+rho_A+||c-s||. For each sign, f_T=4eta+lambda*rho_A+||actual_offset∓lambda*s||+64epsilon*(||R_signed||+||R0||+lambda||a||). These are empirical arithmetic/replay screening values, not rigorous output-error bounds or uncertainty intervals; directional amplification of offset errors is not globally bounded by them. No measured error is reduced by a floor.

A signed point's offset is resolved only if max(||offset_error_plus||,||offset_error_minus||)/(lambda||s||)<=1e-8. A zero intended displacement is unresolved. Relative derivative quantities use ||d|| and ||a|| separately; a denominator<=1e-14 produces None and cannot pass. Agreement needs both raw relative errors<=1e-6 and both relative diagnostic floors<=1e-6 at the same resolved lambda. This does not itself establish an order.

For each of four absolute-error series (D, A, T_plus, T_minus), compute log2(error_large)-log2(error_small) only when both operands are strictly positive, both offsets are resolved and each error is strictly above10times its diagnostic floor. Otherwise retain order=None with the exact reason. A qualifying order lies in[1.5,2.5]. D and A must share a block of two consecutive qualifying halvings; the two Taylor signs must separately share such a block. Keep every qualifying block and agreement index, not a best point. Exact zeros and roundoff-limited affine agreement remain useful recorded observations but do not manufacture a quadratic order.

The helper returns status completed for valid full evidence. derivative_qualified requires common agreement plus the shared derivative-order block; signed_taylor_qualified requires the shared signed-Taylor block. Outcomes are sampled_directional_curvature_supported when both hold; derivative_agreement_taylor_unresolved when only the derivative criterion holds; agreement_without_resolved_orders when common agreement exists without derivative qualification; otherwise directional_check_unresolved. A failed gate alone does not identify a faulty JVP. Every outcome retains13 rows,12 adjacent comparisons per series, full diagnostic vectors, floors, masks and all indices. serializable(result) provides JSON-safe values including None and zeros.

The production runner is `tools/run_public_stalled_directional.py`; its pure analysis is `tools/stalled_directional_analysis.py`. The canonical prior independent audit, full correction evidence and final saved second-direction JVP file are pinned by SHA-256 in the runner. They bind the literal base, unnormalized direction, old residual/action and the three historical positive trials. A historical positive replay failure is checked after all 26 signed chains, so it cannot select the retained ladder. There is no second state or direction and no numerical retry.

Use one native64 Qwen CPU process and one thread. Hard limits are 181 model forwards, 900 seconds internally, 960 seconds externally with 30 seconds termination grace, 1,140 seconds for the outer job, and 2,400 seconds for the verified return worker. Memory limits are 8 GiB RSS and 16 GiB virtual memory. The raw cap is 320 MiB, with 16 MiB reserved for finalization; package and archive caps are 352 and 384 MiB. The transfer reservation is 2 GiB, including incremental return, full return and fallback; the upload cap is 8 MiB. These are this job's limits, within the unchanged monthly 7 TB maximum. Existing remote weights are hashed, not transferred again.

The fixed qualification group uses 25 calls; each of 13 signed groups uses 12. Each completed group closes immutably for early retrieval. Final analysis inputs, vectors and JSON are created after the last group and must be verified through the complete final manifest. Save full probabilities, every intermediate state/tangent, effective precision observations and all resource/count outcomes. Failed and unrun groups are explicit. A complete scientific null has status `completed` and outcome `directional_unresolved`; it is distinct from a missing instrument precondition or incomplete execution. A qualified directional result does not establish a periodic root, attraction, stability, causal hysteresis or communication.

Measured sizing uses the previous same-host scalar timings: approximately 110 seconds of model work, or 133 seconds using the observed per-class maxima, before loading, hashing and I/O. The 180 raw non-prefix files project to 221,595,144 bytes. These are resource estimates, not promised performance. Admit only after a fresh check of task-owned processes, idle human trials, memory, disk, current shared traffic reservations and a verified ready return worker. Use one isolated committed source checkout and retain the remote evidence until its complete local return is verified.


## 2026-10-29 09:50 UTC: one bounded restart with a longer fixed damping ladder

Adapted exploratory numerical work, evidence rung 0; repository base
`146bb69c3dade3245958dc359bc2fcd1acbdf100`. The completed signed diagnostic qualified local derivative
consistency at the literal stalled iterate and found its first improving
positive trial at 1/16. These observations motivate a new correction run;
they do not supply a root or a preferred new starting point. The completed
corrector and signed diagnostic remain unchanged, with all failed trials.

The new profile is `public-stalled-restart`, implemented by
`tools/run_public_stalled_restart.py`. Start from the literal second stalled
iterate, float64 C-order state hash
`2213f700fba2862ff97a6ac4408c8650026cb4ed169cab4cde02d3da32f3fa52`.
Keep the same M controls, prefix, pinned Qwen revision, scalar native64 map
and one CPU thread. Recompute every proposed direction with fresh actions;
do not start from the improving diagnostic trial or reuse its residual.

The trust ball stays centered on the original P history-1 terminal state,
hash `5120fd09919019b9db5e450f7116663a1675a90686f5787ee194b5540a333038`,
with radius 0.005421550304749058. The new helper's explicit `trust_center`
must preserve this original anchor independently of the restart state.
Record both restart and center distances; only center distance gates
admission. Reject a malformed or out-of-ball initial state before callbacks.
No clipping, normalization, moving center or deflation is allowed. The known
M comparator remains admissible, and convergence to it is retained.

Keep four Newton iterations, eight unrestarted GMRES actions per iteration,
two-pass modified Gram--Schmidt, a fresh full-space action, linear relative
tolerance 0.1, action consistency 1e-10, Armijo coefficient 1e-4, root L2
tolerance 1e-10, and breakdown threshold 1e-13. The only search adaptation is
the fixed damping sequence (1,1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256).
Accept the first in-ball point satisfying the existing actual-residual
Armijo rule. Carry that measured residual into the next iteration. Retain
every evaluated rejection and every skipped out-of-ball point. Exhaustion
at the last damping or fourth iteration is unresolved; no ladder extension,
second seed, attraction probe or automatic retry follows.

Repeat the earlier corrector's full 91-call instrument at the restart:
one prefix, 72 ordinary and 18 differentiated calls. This includes the
known-M comparator, full/cache, two fixed directions and their combination,
three central-difference scales, precision operations and exact repeat.
Keep its original tolerances and all recorded raw directions. Compare the
fresh restart residual with the pinned historical residual at Linf<=1e-10,
without an additional forward. The reserved initial repeat is either the
helper's initial callback or an explicitly bound supplied residual, never
both. Qualification from the signed diagnostic cannot replace these fresh
checks. Failed qualification stops distinctly from an unresolved search.

There are six immutable close groups: instrument, four Newton iterations,
and conditional final check. Each Newton iteration has at most 108 scalar
forwards: eight six-step actions, one fresh six-step action, and nine
six-step trials. The maximum is 91+4*108+24=547 calls: 234 differentiated,
312 ordinary (294 cached and 18 full-prefix), and one prefix. Without the
conditional final check the maximum is 523. Helper limits are 37 residual
callbacks including the reserved repeat and 36 Jacobian callbacks. Count
attempts before execution; retain partial chains, unused and failed groups.

Only a residual-qualified candidate triggers the unchanged 24-call final
check: twelve carried updates, six full/cache comparisons and six exact
repeats. Require fresh residual and maximum all-phase lag-six closure
<=1e-10, cached/full Linf<=1e-10, exact repeat, and proper-divisor maximum
phase gaps for 1,2,3 greater than 1e-4. Save all six phase-shift distances
to the known M cycle and both original P histories. A primitive P6 candidate
is distinct only when min_shift(max_phase_L2_gap)>1e-4 from the known M
cycle. Retain the existing outcomes `root_search_unresolved`,
`root_qualification_unresolved`, `distinct_periodic_candidate` and
`merged_cycle_candidate`. Geometric association is not continuous branch
identity. No result establishes attraction, full-spectrum stability,
nonexistence of another orbit, branch disappearance or causal hysteresis.

The runner pins thirteen input files: the original seven packet/audit/M
comparator files, the three preceding corrector audit/evidence/action files,
and the passed signed-diagnostic audit/result/analysis. Their hashes are
fixed in source; total input size is 7,335,810 bytes. The original recorded
states and known comparator supply provenance and the trust anchor, not
another seed. Source bundle, scripts and probe must fit the remaining
8 MiB upload allowance. No model weights are uploaded or reacquired.

Keep the prior corrector resource envelope: one process/thread, 8 GiB RSS,
16 GiB virtual memory, 1800 s internal, 1860 s external with 30 s grace,
2040 s outer with 30 s grace and a 3600 s bounded return worker. Raw,
package and archive caps are 768, 896 and 1024 MiB; transfer reservation is
4 GiB, covering incremental shards, full return and one complete fallback.
The monthly allowance remains 7 TB. The maximum 546 non-prefix raw NPZ
files project to 675,370,956 bytes; measure a maximum-size synthetic
serialization including duplicated evidence and finalization reserve before
freeze. Retain probabilities. Saved per-kind timings project 1189.29 s at
the slower median and 1326.26 s at observed maxima, before startup and I/O;
these size the pilot and do not guarantee runtime.

Before launch, test default helper equivalence, the original-center versus
restart-center counterexample, rejection through 1/8 followed by acceptance
at 1/16, all-nine rejection, accepted residual reuse, maximum call and
serialization budgets, actual result filenames and all completion outcomes.
Freeze the independent saved-array checker before outcomes. It must replay
the instrument, all carried states/tangents, projected and fresh full-space
actions, original-center trust and Armijo decisions, every group/count and
conditional recurrence/identity. Preserve every failed audit attempt.
Fresh host capacity, task-owned process and idle-trial checks, shared traffic
reservation, isolated committed source, verified return probe and ready
completion worker remain mandatory. Root alone launches; the workers only
retrieve evidence. No fresh rental, model API call or paid review is involved.

## 2026-10-29T10:39:11Z: bounded continuation from P through N to M

This adapted exploratory design follows the independently audited fixed-M
restart's line-search exhaustion. It changes the method to a short sequence
of fixed-parameter root corrections. The purpose is to recover qualified
sampled periodic candidates where finite arrivals merged. Residual reduction
alone does not authorize a longer search. No new outcome has been seen.
The immutable v2 proposal and phase-contract check are bound in
`docs/receipts/public-root-continuation-design.json`.

Fix alpha at 0.07293697809562831 and use only three named controls on H–E0:
P at fraction 0.75, log10(T)=-0.15361287955001818;
N at 0.78125, log10(T)=-0.1535987696062659; and
M at 0.8125, log10(T)=-0.1535846596625136. Pin their binary64 encodings.
P1 is the original `source_terminal_states[1]`, P0 is `[0]`, and the known
M comparator is the original `midpoint-tail.npz:states[-1,0]`. Revalidate
the seven original input files and their source audits. Neither a stalled
iterate nor a preferred rejected trial supplies a seed.

First qualify literal P1 freshly, without polishing: twelve carried updates,
six full-prefix checks and six exact-repeat updates. Require root L2 and
all seven phase lag-six residuals <=1e-10, cached/full Linf <=1e-10,
bitwise-equal repeat, and proper-divisor maximum phase gaps for 1,2,3 >1e-4.
Compare the fresh chronological cycle directly to the saved P1 cycle at
Linf<=1e-10, and require all-shift/minimum maximum-phase L2 separation >1e-4
from saved P0. Save all six shift distances. An unqualified anchor ends
the pilot; no additional P dwell or alternative phase follows.

At N, obtain the same-control comparator from exactly 240 carried updates
starting at literal P0, followed by twelve qualification updates and six
full-prefix checks. Apply the same root, phase, primitive-period and cache
gates, without an extra exact repeat. Stop if it is unqualified. This is an
H0-seeded arrival at N, not a proved continuation of that branch.
At M requalify the pinned known M comparator with twelve carried and six
full-prefix calls, including direct saved-cycle agreement at Linf<=1e-10.

At each target, qualify the previous accepted root as a new seed using
72 calls: six baseline, six full-prefix, six exact-repeat; three six-update
JVP chains for fixed PCG64 directions 0/1 and their 0.6/-0.8 combination;
and six six-update central-difference trajectories with h=1e-5,5e-6,2.5e-6.
Require fresh native64 observations, same-input primal Linf<=1e-10,
nonzero action norm >1e-14, both halving orders in [1.5,2.5], minimum DG
relative finite-difference error <=1e-6, linearity relative error<=1e-10
and exact repetition. Record DR errors using actual offsets separately;
DG agreement alone does not bound cancellation in DR. An instrument failure
is distinct from an unresolved numerical search.
Finite failures of the finite-difference order/error, linearity or resolved
action-norm gates stop as `derivative_unqualified`, a completed scientific
unresolved outcome. Cache disagreement, JVP-primal disagreement, ineffective
precision, nonfinite values or bitwise-repeat defects are `instrument_failed`
and exit nonzero. Both classes retain the failed gate and prevent later groups;
neither can be interpreted as absence of a periodic solution.

Solve R_f(x)=F_f^6(x)-x at fixed N, then conditionally M, with at most TWO
Newton iterations per target, eight unrestarted Arnoldi actions and one
fresh full-space action per iteration. Keep two-pass orthogonalization,
absolute breakdown 1e-13, fresh relative linear residual <=0.1, action
consistency <=1e-10, root L2 <=1e-10, Armijo coefficient 1e-4, and fixed
ordered dampings 2^-k for k=0..8. The permanent trust center is original
P1 and its radius is 2*norm(P1-P0)=0.005421550304749058. Never recenter,
clip, normalize or change phase. Retain every considered/evaluated/rejected
trial and every out-of-ball skip. Supply the fresh repeated-seed residual
through its existing exact state/residual hash contract, with no hidden
initial callback. Per target allow at most 18 new residual callbacks and
18 action callbacks. All later differentiated primals must agree with
their same-base ordinary trajectory.

Only apparent convergence permits the 24-call fresh root/period/cache/repeat
check, using the P qualification thresholds. A qualified candidate matching
the same-control comparator within 1e-4 is a known-cycle outcome and stops
the pilot. Only a qualified, distinct N candidate permits M. Preserve all
six shift distances against same-control, preceding qualified and original
P cycles; cross-control distances are descriptive. Any instrument failure,
unqualified comparator/root, inner/outer exhaustion, known-cycle match or
resource failure prevents all remaining targets. No additional target,
subdivision, seed, damping, iteration or automatic extension is available.

Phase and carry are explicit. Every 13-state trace includes the literal
input at index zero; its chronological comparison cycle is `states[1:7]`.
For a saved cycle x_(t-5)..x_t this fresh cycle is x_(t+1)..x_(t+6), the
same six phases. A direct comparison to `states[:6]` would be shifted.
Root residual is norm(T[6]-T[0]); seven closure values use T[6:13]-T[:7].
Full-prefix inputs T[:6] must produce T[1:7]; exact repeat includes its
input and equals T[:7] bitwise. N arrival is shape(241,896), uses terminal
index240 and chronological tail235:241. The fresh N comparator cycle is
its qualification trace[1:7]; the arrival-tail discrepancy is recorded
without a new gate. The decisive M reference remains the pinned saved M
cycle `states[-6:,0]`; fresh-M comparisons are also retained descriptively.
The N solver starts from literal P1. The M solver starts from the literal
accepted N candidate, never a replay endpoint or rotated representative.

There are 11 immutable close groups with maxima
[25,258,72,108,108,24,18,72,108,108,24]: P anchor (including prefix), N
comparator, N instrument, two N iterations, N final, M comparator,
M instrument, two M iterations, M final. The total cap is 925 attempted
scalar forwards: one prefix, 630 cached ordinary, 42 full-prefix ordinary
and 252 differentiated. A JVP already supplies its primal. Count attempts
before execution and retain failed/partial calls and every unrun group.
Reuse one fixed-prefix cache only after testing that named temperature
switches leave the shared prefix, steering and hooks unchanged. Record
the active control and binary64 values for every raw call.

Use one native64 CPU model/thread, 8 GiB RSS,16 GiB virtual memory,
2400 s internal wall limit,2460 s external with30 s grace,2640 s outer
with30 s grace and4800 s return worker. Require at least12 GiB free disk.
Hard raw/package/archive caps are1536/1792/2048 MiB; upload8 MiB and shared
transfer reservation8 GiB, within the unchanged7 TB monthly maximum.
Reserve32 MiB raw for finalization and64 KiB for terminal rewrite.
The measured per-kind timing projection is about24.56 minutes at medians
or28.50 minutes using observed maxima, before additional overhead; these
are sizing estimates. The per-call uncompressed NPZ subtotal is
1,140,800,808 bytes before duplicated evidence. A complete maximum-path
serialization/return rehearsal must fit all caps before source freeze;
runtime guards still apply. Preserve full probability vectors and all
raw states/tangents, actual offsets, solver matrices and event evidence.

Before launch, test the actual file/control/phase contracts, fixed-center
and supplied-residual behavior, distinct/known roots, every failure/stop,
exact worst-case call ledger, conflicting provenance and full-size return.
Freeze the independent checker before outcomes; it reconstructs every
raw chain, instrument gate, correction/identity decision and returned hash
without producer-analysis imports. Commit code and a prospective run card.
Fresh host process/worker/idle-trial/capacity/traffic admission, an isolated
committed checkout, atomic reservation and verified ready return worker
with a returned probe precede root's launch. No fresh rental, paid model
API or review is involved.

These sampled corrections cannot establish continuous branch identity,
attraction, uniqueness, exact-arithmetic existence or branch loss. An
unresolved target narrows the completed method, not a disappearance
bracket. The nominal writer jumps directly to E0 and never visits N or M;
this pilot therefore cannot identify that writer's causal mechanism.
Qualified candidates would support a separately designed finite-return
or eventual endpoint/arrival intervention. Otherwise reconsider the
method offline instead of extending serial residual reductions.

## Matched-displacement intervention at H

The completed intermediate-control continuation remained unresolved at N.
The next adapted experiment changes the state at the actually used common
hold, without another correction restart. Its complete prospective design is
[the matched-displacement protocol](public-state-intervention.md): three
two-lane groups compare the literal successful 60-step handoff with equal-norm
opposite and orthogonal displacements and both unperturbed reference starts.
All controls remain at H for the new 240-update trajectories. One fixed random
direction, both signs, every failed or unresolved row and all new calls are
retained. This can establish finite generic reset sufficiency if a comparison
switches, but cannot identify the nominal writer's mechanism or rule out
simultaneous loss of an endpoint attractor. Preparation and return sizing must
pass before source freeze and numerical launch.
