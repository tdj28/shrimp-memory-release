#!/usr/bin/env python3
"""All eight full-adder input cases (docs/public-full-adder-truth-table.md, fixed 13 Nov 2026 before any call).

The audited Stage 3 adder runner (tools/run_public_adder.py), unchanged except that the digits are one bit each and
come from --x and --y instead of the fixed case 11 + 6. One (x, y) per process; lane 0 carries in 0, lane 1 carries
in 1, so four processes cover all eight cases. The original description follows.

Stage 3: a four-bit ripple-carry adder in one register (docs/public-toggle-sweep.md, fixed 10 Nov 2026 03:40 UTC).

The register holds the carry. Per bit (LSB first), digits x, y: (1) T if x else W; (2) T if y else W -> sum bit;
(3) write E_x if x == y else T -> next carry. T = the Stage 1c toggle, W = idle updates at H, E0/E1 = audited writes;
every move is followed by 240 updates at H and a neutral readout. Moves depend on x and y only; the register is never
read to choose a move. Lane 0 starts with carry 0, lane 1 with carry 1; both share every move. Fixed case 11 + 6.
"""
import copy
import json
from pathlib import Path
import signal
import time
import traceback

import run_public_toggle_sweep as sweep

pilot, prior, memory, engine, reference = sweep.pilot, sweep.prior, sweep.memory, sweep.engine, sweep.reference
PROFILE = 'public-full-adder'
X, Y, BITS = None, None, 1   # X and Y are set from --x and --y
TOGGLE = dict(theta=30, radius=3.5, dwell=12)
WRITE_STEPS = 120


def moves():
    xs = [(X >> i) & 1 for i in range(BITS)]; ys = [(Y >> i) & 1 for i in range(BITS)]
    out = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        out.append(dict(bit=i, step='x', move='T' if x else 'W', role='carry xor x'))
        out.append(dict(bit=i, step='y', move='T' if y else 'W', role='sum bit'))
        out.append(dict(bit=i, step='carry', move=f'E{x}' if x == y else 'T', role='next carry'))
    return out


def expected():
    """Ideal register value after every move, for lane 0 (carry in 0) and lane 1 (carry in 1)."""
    xs = [(X >> i) & 1 for i in range(BITS)]; ys = [(Y >> i) & 1 for i in range(BITS)]
    lanes = []
    for cin in (0, 1):
        r, seq = cin, []
        for x, y in zip(xs, ys):
            r ^= x; seq.append(r)
            r ^= y; seq.append(r)
            r = x if x == y else r ^ 1; seq.append(r)
        lanes.append(seq)
    return [list(p) for p in zip(*lanes)]


def decode(per_move):
    """Sum bits (after each 'y' move) and carry out (after the last move) -> the number, per lane."""
    out = []
    for lane in (0, 1):
        bits = [per_move[3*i + 1][lane] for i in range(BITS)] + [per_move[-1][lane]]
        out.append(None if None in bits else sum(b << i for i, b in enumerate(bits)))
    return out


def controls(move):
    if move == 'T':
        return sweep.target(TOGGLE['theta'], TOGGLE['radius']), TOGGLE['dwell']
    if move == 'W':
        return dict(sweep.HOLD), TOGGLE['dwell']
    return dict(sweep.ENDPOINTS[int(move[1])]), WRITE_STEPS


def plans():
    stages, calls = [], []
    for k, m in enumerate(moves()):
        ct, steps = controls(m['move'])
        for name, c, n in ((f'm{k:02d}-{m["move"]}', ct, steps), (f'm{k:02d}-hold', dict(sweep.HOLD), sweep.HOLD_STEPS)):
            st = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
            st.update(profile=PROFILE, stage_id=len(stages), group_id=0, group='adder', local_stage_id=len(stages), name=name,
                      controls=dict(c), control_bits=pilot.control_bits(c), nominal_controls=dict(c), steps=n,
                      move_index=k, move=m['move'], bit=m['bit'], role=m['role'],
                      historical_comparison_scope='not_applicable_fixed_toggle_exploration')
            stages.append(st)
    for st in stages:
        for category, step, cached in [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)] + [
                ('update', i, True) for i in range(1, st['steps'] + 1)]:
            calls.append(dict(profile=PROFILE, call=len(calls) + 1, group_id=0, group='adder', stage_id=st['stage_id'],
                              stage=st['name'], category=category, step=step, cached=cached, status='not_started'))
    return stages, calls


def run(args, backend_factory=prior.Backend, qualifier=pilot.qualify_inputs, clock=time.monotonic,
        rss=reference.rss_bytes, install_signals=True, require_runtime=True):
    stages, slots = plans()
    recorder = sweep.Recorder(args.output_dir, clock, rss)
    budget = sweep.Budget(recorder, stages, slots)
    want = expected()
    result = dict(profile=PROFILE, status='execution_failed', outcome='not_evaluated', x=X, y=Y, moves=moves(),
                  expected=want, expected_numbers=[X + Y, X + Y + 1], readouts=[], stages=stages, calls=slots)
    handlers = {}
    if install_signals:
        def stop(signum, _frame):
            raise TimeoutError('internal wall cap') if signum == signal.SIGALRM else InterruptedError(f'signal {signum}')
        for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, sweep.MAX_SECONDS)
    try:
        recorder.json('design.json', dict(profile=PROFILE, rules='docs/public-full-adder-truth-table.md', x=X, y=Y, bits=BITS,
                      toggle=TOGGLE, write_steps=WRITE_STEPS, moves=moves(), expected_per_move=want,
                      expected_numbers=[X + Y, X + Y + 1], calls=len(slots)))
        recorder.json('planned-stages.json', stages); recorder.json('planned-calls.json', slots)
        result['source'] = pilot.source_metadata(args, extra_sources=(Path(__file__).resolve(), Path(sweep.__file__).resolve(),
                                                                      pilot.ROOT/'docs/public-toggle-sweep.md',
                                                                      pilot.ROOT/'tools/run_public_adder.py',
                                                                      pilot.ROOT/'docs/public-full-adder-truth-table.md'))
        recorder.json('source.json', result['source'])
        if any(v != '1' for v in pilot.STARTUP_THREADS.values()):
            raise pilot.InstrumentFailure('four thread limits must equal one before process startup')
        refs, provenance = qualifier(args); engine.shared.check_references(refs)
        recorder.json('qualified-inputs.json', provenance); recorder.npz('reference-cycles.npz', states=refs)
        state = refs[:, 0].copy(); recorder.state('adder-start.npy', state)
        if getattr(args, 'preflight_only', False):          # added: stop before the model loads
            result.update(status='preflight_passed', outcome='not_run_preflight')
            return result
        backend = pilot.guarded_backend_factory(backend_factory, require_runtime=require_runtime)(args, recorder, budget)
        if hasattr(backend, 'bindings'):
            recorder.json('model-inputs.json', backend.bindings)
        for st in stages:
            engine.execute_stage(backend, state, st, recorder, budget)
            if st['name'].endswith('-hold'):
                saved, _, _ = prior.reconstruct(recorder.directory, st)
                lanes = [pilot.neutral_readout(saved[1:, lane], refs) for lane in (0, 1)]
                bits = [None if x['qualified_label'] is None else ('ad', 'td').index(x['qualified_label']) for x in lanes]
                k = st['move_index']
                item = dict(move_index=k, move=st['move'], bit=st['bit'], role=st['role'], bits=bits, expected=want[k],
                            match=bits == want[k], terminal_state_sha256=prior.array_hash(state), readouts=lanes)
                result['readouts'].append(item)
                recorder.json(f'readout-{k:02d}.json', item)
        per = [r['bits'] for r in result['readouts']]
        ok = len(per) == len(want) and all(r['match'] for r in result['readouts'])
        result.update(status='completed', outcome='adder_matched' if ok else 'adder_mismatch',
                      matched=sum(r['match'] for r in result['readouts']), numbers=decode(per))
    except BaseException as exc:
        result.update(status=pilot.failure_status(exc), error=f'{type(exc).__name__}: {exc}',
                      traceback=traceback.format_exc())
    finally:
        if install_signals:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
        recorder.finalizing = True
        result.update(model_calls=budget.calls, call_counts=budget.counts, elapsed_seconds=clock() - recorder.started,
                      peak_rss_bytes=rss())
        recorder.stream.close()
        recorder.json('result.json', result, final=True)
        recorder.json('manifest.json', dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
                        if p.is_file() and p.name != 'manifest.json']), final=True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return result


def parser():
    p = pilot.parser()
    p.description = __doc__
    p.add_argument('--x', type=int, choices=(0, 1), required=True)
    p.add_argument('--y', type=int, choices=(0, 1), required=True)
    return p


def main(argv=None):
    global X, Y
    args = parser().parse_args(argv)
    X, Y = args.x, args.y
    r = run(args)
    return 0 if r['status'] in ('completed', 'preflight_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
