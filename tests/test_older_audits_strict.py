"""The XOR-chain and 11 + 6 adder audits carry the Task 8 validator repair (Codex Task 9). Synthetic archive-format
fixtures built on the runners' own planned scaffolds; no model calls."""
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import audit_public_adder as AU  # noqa: E402
import audit_public_xor_chain as XA  # noqa: E402
import run_public_adder as AD  # noqa: E402
import run_public_xor_chain as XR  # noqa: E402
import task8_audit_common as C  # noqa: E402

spec = importlib.util.spec_from_file_location('task8_fixtures', ROOT/'tests'/'test_task8_revised.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


@pytest.fixture
def refs(tmp_path, monkeypatch):
    R = F.refs()
    f = tmp_path/'refs.npz'
    np.savez(f, states=R)
    for module in (XA, AU):
        monkeypatch.setattr(module, 'REFERENCES_SHA256', C.sha(f))      # the synthetic file stands in for the canonical
    return R


def build_xor(d, R):
    d.mkdir(parents=True)
    F.env_files(d, R, {k: {'sha256': k} for k in ('config', 'vectors', 'weights')})
    stages, calls = XR.plans(dict(C.T), 12)
    prev, want = R[:, 0], XA.expected()
    for st in stages:
        prev = F.fill_stage(d, st, prev, want[st['chain_index']], R)
    F.save(d/'result.json', dict(profile=XR.PROFILE, status='completed', b_sequence=list(XR.B_SEQUENCE),
                                 toggle=dict(theta=30, radius=3.5, dwell=12), toggle_controls=dict(C.T),
                                 stages=stages, calls=F.returned(calls), model_calls=len(calls),
                                 readouts=[dict(chain_index=k, bits=want[k]) for k in range(8)], elapsed_seconds=1.0))
    F.manifest(d)
    return d


def build_adder(d, R):
    d.mkdir(parents=True)
    F.env_files(d, R, {k: {'sha256': k} for k in ('config', 'vectors', 'weights')})
    stages, calls = AD.plans()
    prev, want = R[:, 0], AU.expected()
    for st in stages:
        prev = F.fill_stage(d, st, prev, want[st['move_index']], R)
    F.save(d/'result.json', dict(profile=AD.PROFILE, status='completed', x=11, y=6, moves=AD.moves(), stages=stages,
                                 calls=F.returned(calls), model_calls=len(calls),
                                 readouts=[dict(move_index=k, bits=want[k]) for k in range(12)], elapsed_seconds=1.0))
    F.manifest(d)
    return d


def audit(module, d, tmp_path, *more):
    out = tmp_path/'audit.json'
    rc = module.main(['--output', str(out)] + [x for run in (d, *more) for x in ('--run', str(run))])
    return rc, json.loads(out.read_text())


def test_literal_oracles_agree_with_the_plans():
    assert XA.expected() == XR.expected() and list(XA.B_SEQUENCE) == list(XR.B_SEQUENCE)
    assert [m['move'] for m in AD.moves()] == list(AU.MOVES) and AU.expected() == AD.expected()
    assert AU.decode(AU.expected()) == [17, 18] == [AD.X + AD.Y, AD.X + AD.Y + 1]


def test_valid_runs_pass(tmp_path, refs):
    rc, rep = audit(XA, build_xor(tmp_path/'xor', refs), tmp_path)
    assert rc == 0 and rep['passed'] and rep['schema'] == 'public-xor-chain-audit-v2'
    rc, rep = audit(AU, build_adder(tmp_path/'adder', refs), tmp_path)
    assert rc == 0 and rep['passed'] and rep['runs'][0]['numbers'] == [17, 18]


FAULTS = ['extra_stage', 'missing_stage', 'update_steps_all_one', 'float32_interior_block', 'wrong_initial_digest',
          'wrong_stage_name', 'readouts_reordered', 'unreadable_result']


def apply(d, fault):
    def change(_, r):
        if fault == 'extra_stage':
            r['stages'].append(copy.deepcopy(r['stages'][-1]))
        elif fault == 'missing_stage':
            del r['stages'][5]
        elif fault == 'update_steps_all_one':
            for c in r['calls']:
                if c['category'] == 'update':
                    c['step'] = 1
        elif fault == 'float32_interior_block':
            b = r['stages'][3]['blocks'][4]
            p = d/b['path']
            with np.load(p, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays['states'] = arrays['states'].astype(np.float32)
            np.savez_compressed(p, **arrays)
            b.update(F.binding(p))
        elif fault == 'wrong_initial_digest':
            r['stages'][2]['initial']['sha256'] = '0' * 64
        elif fault == 'wrong_stage_name':
            r['stages'][4]['name'] = 'renamed'
        elif fault == 'readouts_reordered':
            r['readouts'][0], r['readouts'][1] = r['readouts'][1], r['readouts'][0]
    if fault == 'unreadable_result':
        (d/'result.json').write_text('{not json')
        F.manifest(d)
    else:
        F.edit_result(d, change)


@pytest.mark.parametrize('fault', FAULTS)
def test_xor_faults_rejected_with_receipt(tmp_path, refs, fault):
    d = build_xor(tmp_path/'xor', refs)
    good = build_xor(tmp_path/'xor-good', refs)
    apply(d, fault)
    rc, rep = audit(XA, d, tmp_path, good)
    assert rc == 1 and not rep['passed'] and not rep['runs'][0]['passed'] and rep['runs'][0]['problems']
    assert rep['runs'][1]['passed']                     # the other run keeps its own result


@pytest.mark.parametrize('fault', FAULTS + ['wrong_digits'])
def test_adder_faults_rejected_with_receipt(tmp_path, refs, fault):
    d = build_adder(tmp_path/'adder', refs)
    if fault == 'wrong_digits':
        F.edit_result(d, lambda _, r: r.update(x=12))
    else:
        apply(d, fault)
    rc, rep = audit(AU, d, tmp_path)
    assert rc == 1 and not rep['passed'] and rep['runs'][0]['problems']


def test_xor_wrong_toggle_rejected(tmp_path, refs):
    d = build_xor(tmp_path/'xor', refs)
    F.edit_result(d, lambda _, r: r['toggle'].update(theta=60))
    rc, rep = audit(XA, d, tmp_path)
    assert rc == 1 and any('toggle' in q for q in rep['runs'][0]['problems'])


# Codex Task 8/9 recheck, N1: the saved per-call gates reach the older audits through the shared check.
@pytest.mark.parametrize('field,value,category', [('input_unchanged', False, 'update'),
                                                  ('cache_unchanged', 'missing', 'comparison'),
                                                  ('expanded_cache_valid', False, 'update'),
                                                  ('cache_unchanged', False, 'update')])
@pytest.mark.parametrize('which', ['xor', 'adder'])
def test_older_audits_reject_failed_call_gates(tmp_path, refs, which, field, value, category):
    d = (build_xor if which == 'xor' else build_adder)(tmp_path/which, refs)

    def change(_, r):
        c = next(c for c in r['calls'] if c['category'] == category and c['stage_id'] == 2)
        if value == 'missing':
            c.pop(field)
        else:
            c[field] = value
    F.edit_result(d, change)
    rc, rep = audit(XA if which == 'xor' else AU, d, tmp_path)
    assert rc == 1 and not rep['passed'] and any('call ledger' in q for q in rep['runs'][0]['problems'])
