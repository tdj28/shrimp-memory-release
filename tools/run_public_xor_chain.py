#!/usr/bin/env python3
"""Stage 1c: XOR in one loop with a toggle pulse (docs/public-toggle-sweep.md, fixed 10 Nov 2026 00:50 UTC).

Both lanes start on the literal reference cycles (A = 0 and A = 1, phase 0) and share every control. The fixed incoming
sequence B = 1, 0, 1, 1, 0, 0, 1, 0 selects, per stage, the toggle pulse T (B = 1) or the same number of updates at H
(B = 0); each is followed by 240 updates at H. State is carried literally across all 16 stages. After every hold both
lanes are read with the sweep's neutral readout; expected stored bits are A XOR (running XOR of B).
"""
import copy
import json
from pathlib import Path
import signal
import time
import traceback

import numpy as np
import run_public_toggle_sweep as sweep

pilot, prior, memory, engine, reference = sweep.pilot, sweep.prior, sweep.memory, sweep.engine, sweep.reference
PROFILE = 'public-xor-chain'
B_SEQUENCE = (1, 0, 1, 1, 0, 0, 1, 0)


def expected():
    out = []
    for a in (0, 1):
        x, bits = a, []
        for b in B_SEQUENCE:
            x ^= b; bits.append(x)
        out.append(bits)
    return [list(p) for p in zip(*out)]          # per stage: [lane0, lane1]


def plans(toggle, pulse_steps):
    stages, calls = [], []
    for k, b in enumerate(B_SEQUENCE):
        for local, (ct, steps, name) in enumerate(((toggle if b else sweep.HOLD, pulse_steps, f'b{k}-{"toggle" if b else "idle"}'),
                                                   (sweep.HOLD, sweep.HOLD_STEPS, f'b{k}-hold'))):
            stage = copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0])
            stage.update(profile=PROFILE, stage_id=len(stages), group_id=0, group='xor-chain', local_stage_id=len(stages),
                         name=name, controls=dict(ct), control_bits=pilot.control_bits(ct), nominal_controls=dict(ct),
                         steps=steps, incoming_bit=b, chain_index=k,
                         historical_comparison_scope='not_applicable_fixed_toggle_exploration')
            stages.append(stage)
    for st in stages:
        for category, step, cached in [('prefix', 0, None), ('comparison', 0, True), ('comparison', 0, False)] + [
                ('update', i, True) for i in range(1, st['steps'] + 1)]:
            calls.append(dict(profile=PROFILE, call=len(calls) + 1, group_id=0, group='xor-chain',
                              stage_id=st['stage_id'], stage=st['name'], category=category, step=step, cached=cached,
                              status='not_started'))
    return stages, calls


def run(args, backend_factory=prior.Backend, qualifier=pilot.qualify_inputs, clock=time.monotonic,
        rss=reference.rss_bytes, install_signals=True, require_runtime=True):
    spec = json.loads(Path(args.toggle).read_text())
    toggle = sweep.target(spec['theta'], spec['radius'])
    stages, slots = plans(toggle, spec['dwell'])
    recorder = sweep.Recorder(args.output_dir, clock, rss)
    budget = sweep.Budget(recorder, stages, slots)
    want = expected()
    result = dict(profile=PROFILE, status='execution_failed', outcome='not_evaluated', toggle=spec,
                  toggle_controls=toggle, b_sequence=list(B_SEQUENCE), expected=want, readouts=[], stages=stages,
                  calls=slots)
    handlers = {}
    if install_signals:
        def stop(signum, _frame):
            raise TimeoutError('internal wall cap') if signum == signal.SIGALRM else InterruptedError(f'signal {signum}')
        for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, sweep.MAX_SECONDS)
    try:
        recorder.json('design.json', dict(profile=PROFILE, rules='docs/public-toggle-sweep.md, Stage 1c', toggle=spec,
                      toggle_controls=toggle, control_bits=pilot.control_bits(toggle), b_sequence=list(B_SEQUENCE),
                      expected_per_stage=want, hold_steps=sweep.HOLD_STEPS, calls=len(slots)))
        recorder.json('planned-stages.json', stages); recorder.json('planned-calls.json', slots)
        result['source'] = pilot.source_metadata(args, extra_sources=(Path(__file__).resolve(), Path(sweep.__file__).resolve(),
                                                                      pilot.ROOT/'docs/public-toggle-sweep.md'))
        recorder.json('source.json', result['source'])
        if any(v != '1' for v in pilot.STARTUP_THREADS.values()):
            raise pilot.InstrumentFailure('four thread limits must equal one before process startup')
        refs, provenance = qualifier(args); engine.shared.check_references(refs)
        recorder.json('qualified-inputs.json', provenance); recorder.npz('reference-cycles.npz', states=refs)
        state = refs[:, 0].copy(); recorder.state('chain-start.npy', state)
        backend = pilot.guarded_backend_factory(backend_factory, require_runtime=require_runtime)(args, recorder, budget)
        if hasattr(backend, 'bindings'):
            recorder.json('model-inputs.json', backend.bindings)
        for st in stages:
            engine.execute_stage(backend, state, st, recorder, budget)
            if st['name'].endswith('-hold'):
                saved, _, _ = prior.reconstruct(recorder.directory, st)
                lanes = [pilot.neutral_readout(saved[1:, lane], refs) for lane in (0, 1)]
                bits = [None if x['qualified_label'] is None else ('ad', 'td').index(x['qualified_label']) for x in lanes]
                item = dict(chain_index=st['chain_index'], incoming_bit=st['incoming_bit'], bits=bits,
                            expected=want[st['chain_index']], match=bits == want[st['chain_index']],
                            terminal_state_sha256=prior.array_hash(state), readouts=lanes)
                result['readouts'].append(item)
                recorder.json(f'readout-{st["chain_index"]}.json', item)
        ok = len(result['readouts']) == 8 and all(r['match'] for r in result['readouts'])
        result.update(status='completed', outcome='xor_chain_matched' if ok else 'xor_chain_mismatch',
                      matched=sum(r['match'] for r in result['readouts']))
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
    p.add_argument('--toggle', type=Path, required=True, help='JSON {theta, radius, dwell} of the chosen toggle')
    return p


def main(argv=None):
    r = run(parser().parse_args(argv))
    return 0 if r['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
