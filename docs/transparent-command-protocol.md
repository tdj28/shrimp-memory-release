# Transparent commands for a numerical memory controller

This protocol separates a visible command, its prescribed control action and
the resulting numerical memory state. The initial components were a finite
parser and a conventional digital-latch control. The subsequent
[public-command integration protocol](public-command-memory.md) defines the
parsed-command numerical runner and independent trajectory checks. Its first
run recovered all four requested writes, 0110, and rejected three invalid strings
without calls or state changes; the [verified receipt](receipts/public-command-memory.json)
binds the seven rows and measured cost. The completed language-contract consultation
supports separating these stages; it supplies no experimental result itself.

## Declared language and public action contract

There are exactly two accepted complete strings:

```
Write bit 0 to register R0.
Write bit 1 to register R0.
```

Capitalization, spaces, punctuation and characters are exact. Unknown inputs
are rejected before dispatch, including otherwise meaningful paraphrases.
There is no stripping, normalization, repair, default bit, prefix matching or
inferred instruction. The requested value is explicit to a human reader.

The typed action records the operation, register, bit, attractor label and two
stages. Its public mapping uses the existing finite pulse schedules:

| Requested bit | Label | Endpoint alpha | Endpoint log10 temperature | Endpoint updates |
|---|---|---|---|---|
| 0 | ad | 0.07293697809562831 | -0.15350 | 120 |
| 1 | td | 0.07299785589927693 | -0.15395151820007266 | 120 |

Both actions then prescribe 240 updates at the common hold: alpha
0.07293697809562831 and log10 temperature -0.15395151820007266. These values
come from [the existing pulse design](receipts/p6-write-path-design.json).
The runtime guard binds the complete visible message to every action field
before calling a supplied dispatcher. This module supplies no model dispatcher.
An execution exception propagates without retry and is distinct from a rejected
command.

## Synthetic prerequisite

The literal test oracle is authored separately from the parser and does not
derive expected messages, bits or schedules from producer constants. It covers
the two accepted strings and twelve fixed invalid strings, each from initial
digital state zero and one: 28 contract cases. The invalid categories are unknown
register, unknown bit, missing bit, truncation, negation, question, quotation,
conflicting instructions, duplicate instructions, extra text, an appended
newline command and a Unicode lookalike.

For valid inputs, check the typed bit and every schedule field against that
oracle. For invalid inputs, check zero dispatcher calls and byte-for-byte
unchanged state. Additional fault tests alter the bit, label, operation,
register, endpoint, hold, duration, stage order or field types. Those prepared
actions must fail before any dispatch or state mutation. A dispatcher failure
must propagate without an automatic second call.

The digital latch accepts the same text and applies the declared assignment
directly. It measures conventional finite-contract functionality, with no
numerical attractor or geometric-target requirement. The tests load no model,
run no numerical trajectory and make no network request. Passing these tests
establishes the tested software behavior; it is not an experimental language
reliability percentage.

## Saved-data gate before numerical dispatch

A separate retrospective integration audit now compares parsed actions
against all four hash-bound [isolated write cases](receipts/p6-writes.json).
It checked prescribed controls, step counts, endpoint/hold
continuity, stage cache/full-prefix comparisons and final classifications.
All four actions were compatible; a separate saved-array review passed
1,014 checks and rehashed 173 unique files. See the
[compatibility receipt](receipts/transparent-command-compatibility.json).
The old trajectories remain historical evidence. Attaching a parsed action to
them establishes only retrospective schedule compatibility, not a newly
executed language-to-memory trial.

The historical producer and auditor hard-code isolated, round-trip and byte
schedules. Actual dispatch requires an explicit parsed-message protocol and a
separately specified independent evaluator before a numerical launch. A text
wrapper around the hard-coded schedule would not show that the message caused
the action. The numerical receiver may receive the message and public contract,
but not a private expected-bit field that bypasses parsing.

## Four-write design and completed follow-on

The prospectively declared design proposed one carried-state
sequence from the saved ad phase-zero state: write 0, 1, 1, 0 using the exact
accepted messages. If each preceding write succeeds, the four requests exercise
0-to-0 retention, 0-to-1 switching, 1-to-1 retention and 1-to-0 switching.
Insert a fixed invalid command between successive writes and require no
numerical call or state mutation for those rejected commands. Keep the actual
terminal state after every write, including wrong or unresolved outcomes.

Use the existing effective-float64 map, one thread, 120 endpoint updates and
240 hold updates per write. Rebuild the prefix cache at both stages and retain
both cached/full-prefix comparisons. Each write requires 366 calls: 360 updates,
two prefix preparations and four comparison calls. Four writes therefore have
an envelope of 1,464 calls. The preceding
[four-write run](receipts/p6-writes.json) took 656.24 seconds and
returned a 40,929,280-byte archive. Its 840-second internal and 900-second
external caps, plus 30 seconds of termination grace, are a proposed starting
envelope subject to fresh measured capacity. Resource and transfer admission
remain separate launch prerequisites; the proposed return reservation is
128 MiB, released only after complete verified local return. The executed attempt
met those admission and return conditions; its separate protocol retains the
full design. A later experiment requires its own bounded plan and admission.

Read each completed write with the existing independent numerical gates:
all-phase lag-six residual at most 1e-8; proper divisors 1, 2 and 3 excluded
with maximum phase gap above 1e-4; unique cyclically aligned reference match
within 1e-6. Record frontend rejection, wrong dispatch, correct final state,
wrong final state, unresolved final state and incomplete execution separately.
Wrong or unresolved readouts do not justify resetting the state, extending a
stage or replacing a trial. Execution failures stop the bounded attempt and
retain all attempted and not-started rows.

Run the same visible command stream through the digital latch and measure its
latency separately. Count frontend work, rejection handling, numerical updates,
verification and data handling when reporting the numerical path's cost. Report
the four planned writes and the three expected rejections as different
denominators; rejected inputs cannot inflate accepted-write accuracy.

The completed attempt used 1,464 forwards in 695.58 seconds, with all four
writes correct and all three rejections leaving state bytes unchanged. All
seven rows, 177 producer files and 190 returned files passed independent
saved-data checks. The first auditor stopped on an isolated-checkout path
mapping; a frozen-Git-bound correction passed without rerunning the model.
This result concerns this one dependent sequence under two literal commands.

## Interpretation

This contract deliberately states the intended bit in the message. It tests
ordinary parsing and controller integration, not unrestricted English,
contextual semantic generalization or concealed communication. It changes the
earlier task in which an independently requested geometric symbol had to be
realized while preserving an unrelated PUT command. Its results cannot repair
or replace that task's failed rows. A successful numerical follow-on would
support only its tested visible commands, starting state and write sequence;
it would not establish a general memory guarantee or a benefit specific to
shrimp geometry.
