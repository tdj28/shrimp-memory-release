#!/usr/bin/env python3
"""Independent, model-free audit of returned E01 geometry evidence.

This checker never evaluates the neural map. It verifies the saved map/action
chains, solver algebra and prospective acceptance gates. Finite Arnoldi evidence
is not an all-direction stability certificate. Failed runs remain failed runs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import signal
import subprocess
import tarfile
import time
import traceback

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import numpy as np

PROFILE = 'public-e01'
DIM = 896
ALPHA = .07293697809562831
H = -.15395151820007266
E0 = -.1535
P = -.15361287955001818
N = -.1535987696062659
DELTA = E0 - H
CAPS = dict(ordinary=8720, jvp_cycles=1200, augmented=200, seconds=43200,
            rss_bytes=28000000000, raw_bytes=2 * 2**30)
INPUT_PINS = {
    'fold_curve': 'a74e991023755da0c5c55ec0d388b7293d9051e9686a0d149632791b65bf48a2',
    'fold_states': '1f0b36223040eb2d8905d6f200db164fb9e16bb2d40cea136f4fde618ec40233',
    'p_packet': '86da50426fbef108d7f779e1bdb0573aa0926f35c765f835548b4b59df924332',
    'p_states': '57c5ba33135673f7935483fe10f2a7bf6c3a28a9696c4a937c1f1fb875569584',
    'h_states': 'd96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0',
}
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
PREFIX_IDS = [151644, 8948, 198, 2610, 525, 458, 15235, 7404, 448, 2441, 15235, 13,
              1446, 614, 4583, 11290, 13, 31733, 1910, 311, 22729, 8820, 498, 1366,
              13, 151645, 198, 151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
SOURCE_FILES = ('tools/run_public_fold_geometry.py','tools/public_fold_map.py',
                'tools/public_pseudoarclength.py','tools/public_spectral.py',
                'tools/validate_qwen_map.py','tools/run_public_periodic_corrector.py',
                'pyproject.toml','uv.lock')
AMENDED_SOURCE_FILES = SOURCE_FILES + ('tools/audit_public_fold_geometry.py','tests/test_public_fold_geometry.py',
                                      'tests/test_audit_public_fold_geometry.py','docs/public-e01-fold-protocol.md')
OFFSET_NORM_ULPS = 8
AMENDMENT = 'orthogonal-row-order-1'
FIRST_AUDIT_SHA = '4626d5cd811fd356366cccbfedfc97a262eea89ec8304c70b6b5a11cce868327'
FIRST_SOURCE_COMMIT = 'a720efb5265c0a48a7a088c9d380c1aac3a090c5'
CASE_LIMITS = dict(ordinary=8720,jvp_cycles=1200,augmented=200,raw_bytes=2*2**30,
                   package_bytes=3*2**30,archive_bytes=3*2**30,transfer_bytes=12*2**30)


class AuditError(ValueError):
    pass


def norm(a):
    return float(np.linalg.norm(np.asarray(a)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def safe_name(name, flat=False):
    if not isinstance(name, str) or not name:
        raise AuditError('empty/non-string artifact name')
    p = PurePosixPath(name)
    if p.is_absolute() or str(p) != name or any(x in ('.', '..') for x in p.parts) or (flat and p.name != name):
        raise AuditError('unsafe artifact name: ' + name)
    return name


def decode(value):
    """Verify the producer's typed-array JSON envelopes before using values."""
    if isinstance(value, dict):
        if set(value) == {'dtype', 'shape', 'sha256', 'values'}:
            dtype = np.dtype(value['dtype'])
            if dtype.kind not in 'biufc' or dtype.hasobject:
                raise AuditError('unsafe typed array dtype')
            a = np.asarray(value['values'], dtype=dtype)
            if list(a.shape) != value['shape'] or array_hash(a) != value['sha256']:
                raise AuditError('typed array bytes/shape mismatch')
            return a
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def cycle_metrics(trace):
    a = np.asarray(trace)
    if a.dtype != np.float64 or a.shape != (13, DIM) or not np.isfinite(a).all():
        raise AuditError('invalid primitive-six validation trace')
    closure = np.linalg.norm(a[6:12] - a[:6], axis=1)
    gaps = {str(k): float(np.linalg.norm(a[k:k+6] - a[:6], axis=1).max()) for k in (1, 2, 3)}
    return dict(passed=bool(closure.max() <= 1e-10 and min(gaps.values()) > 1e-4),
                closure_l2=closure.tolist(), closure_max=float(closure.max()), proper_divisor_gaps=gaps)


def cyclic_distances(first, second):
    return np.array([np.linalg.norm(np.roll(first, -k, axis=0) - second, axis=1).max() for k in range(6)])


def derivative_gate(rows):
    orders, longest, run = [], 0, 0
    for left, right in zip(rows, rows[1:]):
        order = None
        if math.isclose(left['h'] / right['h'], 2., rel_tol=1e-10) and left['relative_error'] > 0 and right['relative_error'] > 0:
            order = math.log2(left['relative_error'] / right['relative_error'])
        qualified = order is not None and 1.5 <= order <= 2.5
        run = run + 1 if qualified else 0
        longest = max(longest, run)
        orders.append(dict(h_large=left['h'], h_small=right['h'], order=order, qualifies=qualified))
    best = min(r['relative_error'] for r in rows)
    return dict(passed=bool(longest >= 2 and best <= 1e-6), orders=orders,
                minimum_relative_error=best, relative_tolerance=1e-6,
                required_consecutive_quadratic_halvings=2, observed_longest_run=longest)


class Audit:
    def __init__(self, run, *, synthetic=False):
        self.run = Path(run).resolve()
        self.synthetic = synthetic
        self.started = time.monotonic()
        self.assertions = 0
        self.bindings = {}
        self.cycles = []
        self.by_label = defaultdict(list)
        self.solver_calls = defaultdict(dict)
        self.solver_results = {}
        self.prefixes = []
        self.alphas = []
        self.warnings = []
        self.derived_offset_reductions = []
        self.recorded_offset_reductions = []
        self.offset_events = {}
        self.offset_starts = {}
        self.used_offset_events = set()
        self.case = None
        self.amended = False

    def same_binding(self, actual, expected, message):
        self.require(actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256'], message)

    def case_context(self,path=None,pin=None):
        """Check the one disclosed amendment's prior accepted evidence, not its science again."""
        args=self.source['arguments'];remote=args.get('case_context');remote_pin=args.get('case_context_sha256')
        if path is None and pin is None and remote is None and remote_pin is None:
            self.require(self.result.get('attempt_index',0)==0 and self.result.get('case_context') is None,
                         'first attempt cannot claim amendment metadata')
            self.require(self.result.get('fold_equation_order',['orbit','null','normalization'])==['orbit','null','normalization'],
                         'original fold equation order')
            return
        self.require(path is not None and pin is not None and remote is not None and remote_pin==pin,
                     'caller and producer case-context pins are required together')
        path=Path(path).resolve();binding=self.bind(path)
        self.require(binding['sha256']==pin,'caller case-context digest')
        doc=self.read(path)
        self.require(doc.get('profile')=='public-e01-case-context' and doc.get('amendment')==AMENDMENT
                     and doc.get('attempt_index')==1,'fixed first amendment identity')
        keys=('prior_audit','prior_result','prior_source','started_utc','input_packet','input_arrays',
              'prior_final_manifest','prior_transfer_closeout','prior_maintenance_closeout','prior_completion',
              'prior_terminal_process_check','prior_transfer_account','prior_local_return_verification')
        bound={};values={}
        for key in keys:
            self.require(key in doc,'missing case-context evidence '+key)
            row=doc[key];p=Path(row['path'])
            if not p.is_absolute():p=path.parent/safe_name(row['path'],flat=True)
            bound[key]=self.bind(p,row)
            if key=='started_utc':values[key]=p.read_text().strip()
            elif key!='input_arrays':values[key]=json.loads(p.read_text())
        prior=values['prior_audit'];previous=values['prior_result'];oldsource=values['prior_source']
        if not self.synthetic:
            self.require(bound['prior_audit']['sha256']==FIRST_AUDIT_SHA,'exact accepted first geometry audit')
            self.require(prior['source_commit']==FIRST_SOURCE_COMMIT,'exact original numerical source')
        for key,value in dict(status='audit_passed',integrity_status='passed',profile=PROFILE,synthetic=False,
                              scientific_status='completed',outcome='primary_fold_unresolved',pulse_eligible=False,
                              return_status='manifest_verified',closed_verified_return=True).items():
            self.require(prior.get(key)==value,'accepted prior audit prerequisite '+key)
        self.require(prior.get('pulse_admission',{}).get('eligible') is False,'no prior pulse admission')
        aliases=dict(prior_result='result',prior_source='source',started_utc='started_utc',input_packet='input_packet',
                     input_arrays='input_arrays',prior_final_manifest='final_manifest',prior_transfer_closeout='transfer_closeout',
                     prior_maintenance_closeout='maintenance_closeout',prior_completion='completion',
                     prior_terminal_process_check='terminal_process_check')
        index={r['path']:r for r in prior['bindings']}
        for key,record in bound.items():
            original=doc['original_bindings'][key]
            self.same_binding(record,original,'relocated original evidence '+key)
            if key=='prior_audit':continue
            accepted=prior[aliases[key]] if key in aliases else index.get(original['path'])
            self.require(accepted is not None,'prior audit did not bind '+key)
            self.require(original==accepted,'exact original accepted binding '+key)
        self.require(previous.get('profile')==PROFILE and previous.get('status')=='completed'
                     and previous.get('outcome')=='primary_fold_unresolved'
                     and previous.get('attempt_index',0)==0 and previous.get('case_context') is None,
                     'one original finite unresolved geometry attempt')
        self.require(previous['source_commit']==prior['source_commit']==oldsource['commit']==doc['source_commit'],
                     'prior source/result/audit identity')
        self.equal(previous['caps'],CAPS,'prior geometry caps')
        self.require(previous['counts']==prior['counts']==doc['prior_counts'],'exact accepted prior counts')
        counts=previous['counts']
        for key in ('ordinary','jvp_cycles','augmented','model_invocations'):
            self.require(type(counts[key]) is int and counts[key]>=0,'prior counter type '+key)
        for key in ('elapsed_seconds','peak_rss_bytes'):
            self.require(math.isfinite(counts[key]) and counts[key]>=0,'prior resource '+key)
        self.same_binding(self.packet_binding,bound['input_packet'],'unchanged case input packet')
        self.same_binding(self.bind(Path(self.packet_binding['path']).parent/'inputs.npz'),bound['input_arrays'],'unchanged case input arrays')
        text=values['started_utc'];start=datetime.strptime(text,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
        self.require(text==prior['campaign_started_utc']==doc['case_started_utc'],'original actual campaign UTC')
        self.require(doc['geometry_deadline_unix']==start+43200 and doc['campaign_deadline_unix']==start+72000
                     and prior['campaign_deadline_unix']==start+72000,'unchanged absolute case deadlines')
        final=values['prior_final_manifest'];closed=values['prior_transfer_closeout'];account=values['prior_transfer_account']
        mh=bound['prior_final_manifest']['sha256'];entries=len(final['files'])
        self.require(values['prior_local_return_verification']['status']=='verified'
                     and values['prior_local_return_verification']['manifest_sha256']==mh,'prior verified return')
        self.require(closed['state']=='complete' and closed['local_return_verified'] is True
                     and closed['manifest_sha256']==mh and closed['verified_manifest_entries']==entries,'prior closed reservation')
        self.require(values['prior_terminal_process_check']['no_owned_numerical_processes'] is True
                     and values['prior_terminal_process_check']['remaining_numerical_processes']==[],'prior owned process closure')
        self.require(values['prior_maintenance_closeout']['active'] is False
                     and values['prior_maintenance_closeout']['return_manifest_sha256']==mh,'prior maintenance closure')
        self.require(values['prior_completion']['status']=='returned_pending_scientific_audit'
                     and values['prior_completion']['exit_status']==0
                     and values['prior_completion']['transfer_reservation_closed'] is True,'prior completion proof')
        usage=dict(ordinary=counts['ordinary'],jvp_cycles=counts['jvp_cycles'],augmented=counts['augmented'],
                   model_invocations=counts['model_invocations'],raw_bytes=prior['raw_inventory']['raw_bytes'],
                   package_bytes=sum(r['bytes'] for r in final['files']),archive_bytes=final['archive']['bytes'],
                   transfer_bytes=sum(r['bytes'] for r in account['attempts']))
        self.require(usage['transfer_bytes']==closed['charged_transfer_bytes']==prior['transport']['charged_bytes'],
                     'accepted prior charged attempts')
        self.require(doc['case_limits']==CASE_LIMITS and doc['prior_usage']==usage,'fixed case limits and exact prior debit')
        remaining={k:limit-usage[k] for k,limit in CASE_LIMITS.items()}
        self.require(doc['remaining_limits']==remaining and all(v>0 for v in remaining.values()),'exact remaining case allowances')
        qualified=self.read('qualified-case-context.json')
        self.require(qualified['document']==doc,'qualified case-context document copy')
        self.same_binding(qualified['binding'],binding,'qualified case-context binding')
        self.require(qualified['prior_counts']==counts and qualified['prior_source_commit']==prior['source_commit'],
                     'qualified prior attempt metadata')
        for key,row in bound.items():self.same_binding(qualified['verified_bindings'][key],row,'qualified relocated prior '+key)
        self.require(qualified['geometry_deadline_unix']==start+43200 and qualified['campaign_deadline_unix']==start+72000
                     and qualified['raw_limit']==remaining['raw_bytes'] and qualified['prior_raw_bytes']==usage['raw_bytes'],
                     'qualified context limits')
        self.same_binding(self.result['case_context'],binding,'result context binding')
        self.require(self.result['attempt_index']==1 and self.result['amendment']==AMENDMENT,'result amendment identity')
        self.require(self.result['prior_source_commit']==prior['source_commit']
                     and self.result['geometry_deadline_unix']==start+43200 and self.result['campaign_deadline_unix']==start+72000
                     and self.result['prior_raw_bytes']==usage['raw_bytes'] and self.result['raw_limit_bytes']==remaining['raw_bytes'],
                     'result retains original case allowances')
        self.require(self.result['fold_equation_order']==['orbit','normalization','null'],'fixed amended equation order')
        self.case=dict(binding=binding,document=doc,bindings=bound,prior_counts=counts,remaining=remaining,
                       campaign_started_utc=text,geometry_deadline_unix=start+43200,campaign_deadline_unix=start+72000)
        self.amended=True
        self.require(sorted(r['path'] for r in self.source['files'])==sorted(AMENDED_SOURCE_FILES),'amended complete source/checker inventory')

    def require(self, condition, message):
        self.assertions += 1
        if not condition:
            raise AuditError(message)

    def bounded(self):
        self.require(time.monotonic() - self.started <= 600, 'checker 600-second limit')
        self.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= 2 * 2**30, 'checker 2-GiB RSS limit')

    def bind(self, path, expected=None):
        p = Path(path)
        self.require(not p.is_symlink() and p.is_file(), 'missing/symlink file: ' + str(p))
        p = p.resolve()
        old = self.bindings.get(str(p))
        row = old or dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
        if expected is not None:
            self.require(type(expected['bytes']) is int and expected['bytes'] >= 0, 'binding byte type')
            self.require(row['bytes'] == expected['bytes'] and row['sha256'] == expected['sha256'], 'binding mismatch: ' + str(p))
        self.bindings[str(p)] = row
        return row

    def read(self, name, *, typed=False):
        p = Path(name) if Path(name).is_absolute() else self.run / safe_name(name, flat=True)
        self.bind(p)
        result = json.loads(p.read_text())
        return decode(result) if typed else result

    def arrays(self, name):
        p = self.run / safe_name(name, flat=True)
        self.bind(p)
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}

    def close(self, actual, expected, message, *, rtol=2e-11, atol=2e-13):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), message + ': shape/finite')
        self.require(bool(np.allclose(a, b, rtol=rtol, atol=atol)), message)

    def exact(self, actual, expected, message):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), message)

    def equal(self, actual, expected, message):
        if isinstance(expected, np.ndarray):
            self.exact(actual, expected, message)
        elif isinstance(expected, dict):
            self.require(isinstance(actual, dict), message)
            for k, v in expected.items():
                self.require(k in actual, message + ': missing ' + k)
                self.equal(actual[k], v, message + '.' + k)
        elif isinstance(expected, (list, tuple)):
            self.require(len(actual) == len(expected), message + ': length')
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.equal(a, b, message + f'[{i}]')
        elif isinstance(expected, (float, np.floating)):
            self.close(actual, expected, message)
        else:
            self.require(actual == expected, message)

    def inventory(self):
        manifest = self.read('manifest.json')
        rows = manifest['files']
        names = [safe_name(r['path'], flat=True) for r in rows]
        self.require(len(set(names)) == len(names), 'duplicate raw manifest path')
        actual = {p.name for p in self.run.iterdir() if p.is_file()}
        self.require(actual == set(names) | {'manifest.json'}, 'raw manifest is not the complete flat inventory')
        self.require(all(not p.is_symlink() and (p.is_file() or p.name == 'return-shards') for p in self.run.iterdir()), 'unexpected raw directory/symlink')
        for row in rows:
            self.bind(self.run / row['path'], row)
        total = sum(r['bytes'] for r in rows)
        self.require(total == manifest['actual_bytes_before_manifest'], 'raw byte count')
        self.require(total + (self.run / 'manifest.json').stat().st_size <= CAPS['raw_bytes'], 'raw evidence cap')
        owned = set()
        checkpoints = sorted(self.run.glob('checkpoint-*.json'))
        for i, path in enumerate(checkpoints, 1):
            row = self.read(path)
            self.require(path.name == f'checkpoint-{i:04d}.json' and row['checkpoint_id'] == i and row['profile'] == PROFILE and row['closed'] is True, 'checkpoint identity/closure')
            local = [safe_name(r['path'], flat=True) for r in row['artifacts']]
            self.require(len(local) == len(set(local)) and not (set(local) & owned), 'checkpoint ownership must be disjoint')
            self.require(not any(n in ('result.json', 'events.jsonl', 'manifest.json') or n.startswith('checkpoint-') for n in local), 'mutable/marker artifact in checkpoint')
            for record in row['artifacts']:
                self.bind(self.run / record['path'], record)
            owned.update(local)
        return dict(raw_files=len(rows)+1, raw_bytes=total+(self.run/'manifest.json').stat().st_size,
                    checkpoints=len(checkpoints), checkpoint_artifacts=len(owned))

    def archive(self, path, expected):
        seen = set()
        with tarfile.open(path, 'r:') as tar:
            for member in tar:
                self.require(member.isfile() and member.name in expected and member.name not in seen, 'archive member inventory/type')
                row = expected[member.name]
                self.require(member.size == row['bytes'], 'archive member size')
                h = hashlib.sha256()
                with tar.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024**2), b''):
                        h.update(block)
                self.require(h.hexdigest() == row['sha256'], 'archive member hash')
                seen.add(member.name)
        self.require(seen == set(expected), 'archive incomplete')

    def transport(self):
        job = self.run.parent
        if self.synthetic:
            return dict(status='not_checked_synthetic', closed_verified_return=False)
        final = self.read(job/'final-manifest.json')
        self.require((job/'final-manifest.json').stat().st_size <= 4*2**20, 'final manifest cap')
        rows = final['files']; names = [safe_name(r['path']) for r in rows]
        limits=self.case['remaining'] if self.amended else CASE_LIMITS
        self.require(len(set(names)) == len(names) and sum(r['bytes'] for r in rows) <= limits['package_bytes'], 'complete package cap/inventory')
        index = dict(zip(names, rows))
        for name, row in index.items():
            self.require((job/name).resolve().is_relative_to(job.resolve()), 'returned path escaped job')
            self.bind(job/name, row)
        self.require(all('run/'+p.name in index for p in self.run.iterdir() if p.is_file()), 'final return omitted raw file')
        transferred = {safe_name(r['path']): r for r in final['transferred_files']}
        self.require(len(transferred) == len(final['transferred_files']), 'duplicate final transfer path')
        self.require(all(index.get(k) == r for k,r in transferred.items()), 'transferred binding differs')
        self.require(type(final['already_local_files']) is int and final['already_local_files'] == len(rows)-len(transferred), 'already-local count')
        ar = final['archive']; self.require(ar['bytes'] <= limits['archive_bytes'], 'final archive cap')
        archive = job/safe_name(ar['path'], flat=True); self.bind(archive, ar); self.archive(archive, transferred)
        for directory in sorted((self.run/'return-shards').glob('checkpoint-*')):
            if not (directory/'verified.json').exists():
                self.warnings.append('Unverified incremental shard retained: '+str(directory))
                continue
            sm = self.read(directory/'manifest.json'); sv = self.read(directory/'verified.json')
            self.require(sha(directory/'manifest.json') == sv['manifest_sha256'], 'checkpoint verification hash')
            checkpoint = self.read(f"checkpoint-{sm['checkpoint_id']:04d}.json")
            cp = self.run/f"checkpoint-{sm['checkpoint_id']:04d}.json"
            wanted = checkpoint['artifacts'] + [dict(path=cp.name,bytes=cp.stat().st_size,sha256=sha(cp))]
            self.equal(sm['files'], wanted, 'checkpoint shard exact marker payload')
            self.bind(directory/'archive.tar', sm['archive'])
            self.archive(directory/'archive.tar', {r['path']:r for r in wanted})
        verify = self.read(job/'local-return-verification.json')
        terminal = self.read(job/'terminal-process-check.json')
        account = self.read(job/'transfer-account.json'); closed = self.read(job/'transfer-closeout.json')
        maintenance = self.read(job/'maintenance-closeout.json'); completion = self.read(job/'completion.json')
        self.require('terminal-process-check.json' in index, 'terminal process proof absent from complete manifest')
        self.require(terminal['no_owned_numerical_processes'] is True and terminal['remaining_numerical_processes'] == [], 'owned process closure')
        mh = sha(job/'final-manifest.json')
        self.require(verify['status']=='verified' and verify['manifest_sha256']==mh and verify['files']==len(rows), 'return verification')
        self.require(account['reserved_bytes']==limits['transfer_bytes'] and account['overhead_reserve']==32*2**20, 'transfer reservation envelope')
        self.require(all(type(r['bytes']) is int and r['bytes']>=0 for r in account['attempts']), 'transfer attempt bytes')
        charged = sum(r['bytes'] for r in account['attempts'])
        self.require(charged+32*2**20<=limits['transfer_bytes'], 'cumulative transfer limit')
        self.require(closed['id']==job.name and closed['state']=='complete' and closed['local_return_verified'] is True and closed['manifest_sha256']==mh, 'closed own reservation')
        self.require(closed['reserved_bytes']==limits['transfer_bytes'] and closed['verified_manifest_entries']==len(rows), 'closed reservation envelope/inventory')
        self.require(closed['charged_transfer_bytes']==charged and closed['transfer_account_sha256']==sha(job/'transfer-account.json') and closed['terminal_process_sha256']==sha(job/'terminal-process-check.json'), 'closed account/process binding')
        self.require(maintenance['active'] is False and maintenance['validation_id']==job.name and maintenance['return_manifest_sha256']==mh, 'maintenance closure')
        exitcode = int((job/'exit-status.txt').read_text()); self.bind(job/'exit-status.txt')
        expected = 'returned_pending_scientific_audit' if exitcode==0 else 'failed_run_evidence_returned'
        self.require(completion['status']==expected and completion['exit_status']==exitcode and completion['files']==len(rows) and completion['transfer_reservation_closed'] is True, 'completion closure')
        self.require(completion['maintenance']['status']=='cleared' and completion['maintenance']['validation_id']==job.name, 'completion maintenance')
        if self.amended:
            prior=self.case['document']['prior_usage']['transfer_bytes']
            for doc in (account,closed):
                self.require(doc['case_context_sha256']==self.case['binding']['sha256']
                             and doc['case_prior_charged_transfer_bytes']==prior,'cumulative transport context/debit')
            self.require(closed['case_charged_transfer_bytes']==prior+charged,'cumulative transport closeout charge')
            self.require('case-context.json' in index and 'case-started-utc.txt' in index,'returned original campaign context/start')
            self.same_binding(self.bind(job/'case-context.json'),self.case['binding'],'returned case-context bytes')
            for key,row in self.case['bindings'].items():
                name='case-started-utc.txt' if key=='started_utc' else Path(self.case['document'][key]['path']).name
                self.require(name in index,'returned prior context evidence '+name)
                self.same_binding(self.bind(job/name),row,'returned prior context bytes '+key)
        return dict(status='manifest_verified', closed_verified_return=True, files=len(rows), charged_bytes=charged, exit_status=exitcode)

    def provenance(self, source_root, source_commit, packet_path, packet_sha):
        source = self.read('source.json')
        self.require(source['commit']==source_commit and re.fullmatch('[0-9a-f]{40}', source_commit) is not None, 'frozen source commit')
        self.require(source['working_tree_status']==[], 'producer numerical source was dirty')
        root = Path(source_root).resolve()
        self.require(sorted(r['path'] for r in source['files']) in (sorted(SOURCE_FILES),sorted(AMENDED_SOURCE_FILES)), 'complete fixed numerical source inventory')
        for row in source['files']:
            rel = safe_name(row['path'])
            self.bind(root/rel, row)
            if not self.synthetic:
                raw = subprocess.check_output(['git','show',source_commit+':'+rel],cwd=root)
                self.require(len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'], 'frozen Git source differs: '+rel)
        self.source = source
        self.require(source['arguments']['input_packet_sha256']==packet_sha, 'caller/producer packet pin')
        packet = self.read(Path(packet_path).resolve())
        self.require(sha(packet_path)==packet_sha, 'input packet digest')
        self.equal(self.read('qualified-inputs.json'), packet, 'qualified packet copy')
        self.require(packet['parent_indices']==[21,22] and packet['p_phase_order']==[480,475,476,477,478,479] and packet['model_calls']==0, 'packet identity')
        for k,pin in INPUT_PINS.items():
            record = packet['sources'][k]
            self.bind(record['path'], record)
            if not self.synthetic:self.require(record['sha256']==pin, 'literal original input pin '+k)
        arrays = Path(packet_path).parent/'inputs.npz'
        self.bind(arrays, packet['arrays'])
        with np.load(arrays,allow_pickle=False) as z:self.inputs={k:z[k].copy() for k in z.files}
        a=self.inputs
        for k,shape in dict(fold_states=(2,DIM),fold_alphas=(2,),fold_logtemps=(2,),p_terminal=(2,DIM),p_cycles=(6,2,DIM),h_cycles=(2,6,DIM),state_scale=()).items():
            self.require(a[k].dtype==np.float64 and a[k].shape==shape and np.isfinite(a[k]).all(), 'literal input shape '+k)
        original=np.load(packet['sources']['fold_states']['path'],allow_pickle=False)
        curve=json.loads(Path(packet['sources']['fold_curve']['path']).read_text())
        self.exact(a['fold_states'],original[[21,22]],'literal parent rows')
        self.exact(a['fold_alphas'],np.array([curve['points'][i]['theta'][1]/curve['args']['alpha_scale'] for i in (21,22)]),'parent alpha')
        self.exact(a['fold_logtemps'],np.array([curve['points'][i]['theta'][0] for i in (21,22)]),'parent temperature')
        with np.load(packet['sources']['p_states']['path'],allow_pickle=False) as z:
            self.exact(a['p_terminal'],z['source_terminal_states'],'literal P terminals')
            self.exact(a['p_cycles'],np.concatenate((z['source_cycle_states'][-1:],z['source_cycle_states'][:-1])),'P phase order')
        with np.load(packet['sources']['h_states']['path'],allow_pickle=False) as z:self.exact(a['h_cycles'],z['states'],'literal H references')
        self.exact(a['p_cycles'][0],a['p_terminal'],'P representative')
        self.scale=float(a['state_scale']); self.close(self.scale,norm(a['p_terminal'][1]-a['p_terminal'][0]),'literal state scale',rtol=2e-15,atol=0)
        self.require(self.scale>0,'positive scale')
        self.packet=packet; self.packet_binding=self.bind(packet_path)

    def events(self):
        active = None
        alpha = None
        previous_time = -1.
        self.bind(self.run/'events.jsonl')
        with (self.run/'events.jsonl').open() as stream:
            for line_index,line in enumerate(stream):
                self.bounded()
                event = json.loads(line)
                elapsed = event['elapsed_seconds']
                self.require(math.isfinite(elapsed) and elapsed >= previous_time, 'event chronology')
                previous_time = elapsed
                kind = event['event']
                if kind == 'prefix_ready':
                    self.require(event['mode']=='native64' and event['length']==37, 'fixed prefix event')
                    self.prefixes.append(event)
                elif kind == 'alpha_selected':
                    alpha=event['alpha']; self.alphas.append(event)
                elif kind in ('fold_offset_reduction','fold_offset_cycles'):
                    top=event.get('top_file')
                    self.require(top is not None,'offset event without raw mixed action')
                    safe_name(top,flat=True)
                    self.require(type(event['augmented_action']) is int and event['augmented_action']>0,'offset action identity')
                    self.require(type(event['norm']) in (int,float) and math.isfinite(event['norm']) and event['norm']>=0
                                 and float(event['norm']).hex()==event['norm_hex'],'exact recorded offset norm encoding')
                    self.require(type(event['h']) in (int,float) and math.isfinite(event['h']) and event['h']>0,'recorded offset step')
                    if kind=='fold_offset_reduction':
                        self.require(event['status']=='started' and top not in self.offset_starts,'unique offset start event')
                        self.require(all(event[k] is None for k in ('plus_file','minus_file','vector_file')),'offset start precedes signed calls')
                        self.offset_starts[top]=dict(event,line=line_index,active_callback=active)
                    else:
                        self.require(event['status'] in ('completed','incomplete') and top not in self.offset_events,'unique offset terminal event')
                        self.require(top in self.offset_starts,'offset terminal without start')
                        begin=self.offset_starts[top]
                        for key in ('augmented_action','h','norm','norm_hex','input_file','z_sha256','d_sha256','y_sha256','dy_sha256','top_file'):
                            self.require(event[key]==begin[key],'unchanged offset reduction metadata '+key)
                        self.require(active==begin['active_callback'],'offset callback scope')
                        self.offset_events[top]=dict(event,line=line_index,start_line=begin['line'],active_callback=active)
                elif kind == 'cycle':
                    name=safe_name(event['file'],flat=True); z=self.arrays(name)
                    self.require(set(z)=={'states','tangents','alpha','logtemp','dlogtemp','top5'}, 'raw cycle keys')
                    for key in ('alpha','logtemp','dlogtemp'):
                        self.require(z[key].shape==() and z[key].dtype==np.float64 and np.isfinite(z[key]), 'raw control '+key)
                    self.require(float(z['alpha'])==alpha, 'raw cycle uses selected alpha')
                    states=z['states']; tangents=z['tangents']; steps=event['steps_completed']
                    self.require(states.dtype==np.float64 and states.shape==(steps+1,DIM) and np.isfinite(states).all(), 'saved state trace')
                    differentiated=event['differentiated']
                    self.require(type(differentiated) is bool and type(event['cached']) is bool, 'cycle action flags')
                    if differentiated:
                        self.require(tangents.dtype==np.float64 and np.isfinite(tangents).all(), 'saved tangent precision/finite')
                        if tangents.shape!=states.shape:
                            # The adapter appends the finite primal before it
                            # checks a returned tangent. Preserve that final
                            # partial action when the tangent itself failed.
                            self.require(event['failed'] is not None and steps>=1 and tangents.shape==(steps,DIM), 'saved tangent trace')
                            failed=self.arrays(self.find_npz(event['label']+'-failed-return'))
                            self.require('received_tangent' in failed and
                                         (failed['received_tangent'].shape!=(DIM,) or failed['received_tangent'].dtype!=np.float64 or not np.isfinite(failed['received_tangent']).all()),
                                         'partial tangent requires retained invalid returned tangent')
                    else:
                        self.require(tangents.shape==(0,) and float(z['dlogtemp'])==0, 'ordinary call tangent')
                    planned=1 if event['label']=='pilot-full' else (12 if event['label'].endswith('-orbit') or event['label'] in ('p0-anchor','p1-anchor','parent-21','parent-22') else 6)
                    self.require(0<=steps<=planned, 'cycle update range')
                    if event['failed'] is None:
                        self.require(steps==planned and len(event['precision'])==steps and z['top5'].shape==(steps,2,5), 'complete cycle record')
                    else:
                        self.require(isinstance(event['failed'].get('type'),str), 'failed cycle type retained')
                    self.require(len(event['precision'])>=steps, 'missing completed-update precision')
                    for observation in event['precision']:
                        self.require(all(observation.get(group,{}).get(key)=='torch.float64'
                            for group,keys in {'norm':('input','reduction','weight','output'),
                                               'rotary':('input','inverse_frequency','angles','cos','sin')}.items() for key in keys), 'effective float64 observation')
                    if z['top5'].size:
                        self.require(z['top5'].ndim==3 and z['top5'].shape[1:]==(2,5) and np.isfinite(z['top5']).all(), 'top-five diagnostic shape/finite')
                        ids=z['top5'][:,0]; probs=z['top5'][:,1]
                        self.require(np.all(ids==ids.astype(np.int64)) and np.all((ids>=0)&(ids<151936)), 'top-five token indices')
                        self.require(np.all((probs>=0)&(probs<=1)) and np.all(np.diff(probs,axis=1)<=0), 'top-five probabilities')
                    row=dict(event=event,z=z,name=name,line=line_index,planned_steps=planned)
                    self.cycles.append(row); self.by_label[event['label']].append(row)
                    if active is not None:self.solver_calls[active[0]][active[1]]['cycles'].append(row)
                elif kind in ('fold-solver','branch-solver','branch-refinement','native-root'):
                    session=kind+(':'+event['stage'] if 'stage' in event else '')
                    item=event['solver_event']; event_type=item['event']
                    if event_type=='call_begin':
                        self.require(active is None,'nested solver callback')
                        call=decode(item['call']); ident=call['id']
                        self.require(ident not in self.solver_calls[session],'duplicate solver callback')
                        self.solver_calls[session][ident]=dict(begin=call,cycles=[])
                        active=(session,ident)
                    elif event_type=='call_end':
                        call=decode(item['call']); ident=call['id']
                        if call['called']:
                            self.require(active==(session,ident),'callback end identity')
                            self.solver_calls[session][ident]['end']=call
                            active=None
                        else:
                            self.require(call['status'].startswith('not_called_'),'uninvoked callback status')
                            self.solver_calls[session][ident]=dict(end=call,cycles=[])
                    elif event_type=='run_end':
                        self.solver_results[session]=decode(item['result'])
        self.active_callback=active
        self.last_event_seconds=previous_time
        self.require(len({c['name'] for c in self.cycles})==len(self.cycles),'duplicate cycle evidence reference')
        ids=[r['augmented_action'] for r in self.offset_events.values()]
        self.require(ids==sorted(set(ids)),'offset events retain increasing unique augmented action IDs')
        for p in self.run.glob('*-failed-return.npz'):
            z=self.arrays(p.name)
            self.require('received_state' in z and 'received_probabilities' in z,'failed returned output preserved')
        # A returned map chain is saved once per invocation. It is not silently
        # reconstructed from a solver residual or an earlier model output.
        all_cycle_names={p.name for p in self.run.glob('*.npz') if set(self.arrays(p.name))=={'states','tangents','alpha','logtemp','dlogtemp','top5'}}
        self.require(all_cycle_names=={c['name'] for c in self.cycles}, 'orphan/unlogged raw cycle')

    def one(self,label,index=0):
        self.require(label in self.by_label and len(self.by_label[label])>index,'missing raw label '+label)
        return self.by_label[label][index]

    def expect_cycle(self,c,x,temp,*,v=None,dlog=0.,alpha=ALPHA,label=None,complete=True):
        z=c['z']; ev=c['event']
        if label is not None:self.require(ev['label']==label,'callback raw label '+label)
        self.exact(z['states'][0],np.asarray(x,dtype=np.float64),'callback raw initial state')
        self.require(float(z['alpha'])==float(alpha) and float(z['logtemp'])==float(temp) and float(z['dlogtemp'])==float(dlog),'callback physical controls')
        self.require(ev['differentiated']==(v is not None),'callback tangent kind')
        if v is not None:self.exact(z['tangents'][0],np.asarray(v,dtype=np.float64),'callback raw initial tangent')
        if complete:self.require(ev['failed'] is None,'complete callback uses failed raw cycle')

    def unpack(self,y):
        y=np.asarray(y); return y[:-1]*self.scale,H+float(y[-1])*DELTA

    def offset_norm(self,base,direction,h,offsets):
        """Replay only a derived FD offset's unsaved reduction scalar.

        NumPy/BLAS reductions can differ by an ULP across CPU hosts. The
        producer saves the actual signed input states and temperatures, but
        not ||direction||. Search a fixed eight-ULP neighborhood of the local
        reduction and require an EXACT common reconstruction of every saved
        sign. This is an arithmetic-provenance allowance, not a numerical
        residual gate or certified rounding bound. Literal bases and all
        subsequent saved chains remain byte-exact.
        """
        local=norm(direction)
        self.require(math.isfinite(local) and local>0,'finite nonzero offset direction')
        candidates=[(0,local)];lower=upper=local
        for step in range(1,OFFSET_NORM_ULPS+1):
            lower=float(np.nextafter(lower,-np.inf));upper=float(np.nextafter(upper,np.inf))
            candidates.extend(((-step,lower),(step,upper)))
        for ulps,value in candidates:
            if value<=0 or not math.isfinite(value):continue
            matched=True
            for sign,raw in offsets:
                x,t=self.unpack(base+sign*h*direction/value);z=raw['z']
                if not (z['states'][0].dtype==np.float64 and np.array_equal(z['states'][0],x)
                        and float(z['logtemp'])==t):
                    matched=False;break
            if matched:
                if ulps:
                    gaps=[float(np.max(abs(raw['z']['states'][0]-self.unpack(base+sign*h*direction/local)[0]))) for sign,raw in offsets]
                    self.derived_offset_reductions.append(dict(files=[raw.get('name',raw['event']['label']) for _,raw in offsets],
                        local_direction_norm=local,replayed_direction_norm=value,norm_ulp_offset=ulps,
                        maximum_local_state_difference=max(gaps),allowed_norm_ulps=OFFSET_NORM_ULPS,
                        saved_signed_inputs_and_temperatures_exact=True))
                return value
        self.require(False,'derived fold offset has no exact common norm reconstruction within eight ULPs')

    def recorded_offset_norm(self,base,direction,h,cycles,*,complete,full_y,full_d):
        """Use a recorded scalar literally; never replace it by local reduction."""
        if not cycles:return norm(direction)
        first=cycles[0];name=first['name'];event=self.offset_events.get(name)
        if event is None:
            # A failure during the leading mixed action precedes the scalar
            # computation. It cannot have emitted a signed offset.
            self.require(not self.amended or (not complete and len(cycles)==1 and first['event']['failed'] is not None),
                         'amended fold action requires recorded offset scalar')
            return None
        self.require(event['h']==h,'recorded offset uses fixed finite-difference step')
        self.require(event['y_sha256']==array_hash(base) and event['dy_sha256']==array_hash(direction),'recorded offset input hashes')
        self.require(event['z_sha256']==array_hash(full_y) and event['d_sha256']==array_hash(full_d),'recorded full augmented input hashes')
        local=norm(direction);value=float(event['norm'])
        self.require(math.isfinite(local),'local offset norm finite')
        lower=upper=local
        for _ in range(OFFSET_NORM_ULPS):
            lower=float(np.nextafter(lower,-np.inf));upper=float(np.nextafter(upper,np.inf))
        self.require((local==0 and value==0) or (local>0 and value>0 and lower<=value<=upper),
                     'recorded offset norm exceeds eight-ULP arithmetic check')
        expected=dict(top_file=name,plus_file=None,minus_file=None,vector_file=None)
        for raw in cycles[1:]:
            key={'fold-fd-plus':'plus_file','fold-fd-minus':'minus_file','fold-vector':'vector_file'}.get(raw['event']['label'])
            self.require(key is not None and expected[key] is None,'recorded offset nested-call identity')
            expected[key]=raw['name']
        for key,wanted in expected.items():self.require(event[key]==wanted,'recorded offset actual '+key)
        self.require(event['status']==('completed' if complete else 'incomplete'),'recorded offset completion category')
        if all('line' in raw for raw in cycles):
            self.require(event['line']>max(raw['line'] for raw in cycles),'offset event follows its saved actions')
            self.require(cycles[0]['line']<event['start_line'] and all(raw['line']>event['start_line'] for raw in cycles[1:]),
                         'recorded scalar precedes signed/vector actions')
        self.used_offset_events.add(name)
        observation=dict(top_file=name,augmented_action=event['augmented_action'],h=h,recorded_direction_norm=value,
                         local_direction_norm=local,allowed_norm_ulps=OFFSET_NORM_ULPS,exact_recorded_scalar_used=True)
        if not any(r['top_file']==name for r in self.recorded_offset_reductions):self.recorded_offset_reductions.append(observation)
        return value

    def solver_fold_rows(self,value):
        if not self.amended:return value
        return np.concatenate((value[:DIM],value[-1:],value[DIM:-1]))

    def offset_evidence(self):
        """Check recorded reductions even when an instrument check stopped early."""
        by_name={c['name']:c for c in self.cycles};used=set()
        for name,event in self.offset_events.items():
            self.require(event['h'] in (1e-3,5e-4,2.5e-4),'prospective augmented step ladder')
            input_name=safe_name(event['input_file'],flat=True);z=self.arrays(input_name);used.add(input_name)
            self.require(set(z)=={'z','d','augmented_action'} and z['z'].shape==z['d'].shape==(1793,)
                         and z['z'].dtype==z['d'].dtype==np.float64 and np.isfinite(z['z']).all() and np.isfinite(z['d']).all(),
                         'saved augmented action input arrays')
            self.require(z['augmented_action'].dtype==np.int64 and z['augmented_action'].shape==()
                         and int(z['augmented_action'])==event['augmented_action'],'saved action sequence')
            names=[event[k] for k in ('top_file','plus_file','minus_file','vector_file') if event[k] is not None]
            self.require(len(names)==len(set(names)) and all(n in by_name for n in names),'offset raw file linkage')
            cycles=[by_name[n] for n in names]
            if event['status']=='completed':
                self.callback_value(dict(kind='action',y=z['z'],direction=z['d']),cycles,'fold-solver:primary',
                                    action_h=event['h'],solver_rows=False)
            else:
                # The failure may precede the second signed call. No callback
                # output or scientific gate is inferred from the retained part.
                self.failed_callback_inputs(dict(kind='action',y=z['z'],direction=z['d']),cycles,
                                            'fold-solver:primary',action_h=event['h'])
        inputs={p.name for p in self.run.glob('*-fold-action-input.npz')}
        missing=inputs-used
        if missing:
            self.require(self.result['status'] in ('instrument_failed','execution_failed','incomplete') and len(missing)==1,
                         'unassociated augmented input requires one terminal failed action')
            p=next(iter(missing));z=self.arrays(p)
            self.require(int(z['augmented_action'])==self.result['counts']['augmented'], 'terminal augmented input action index')
            self.require(z['z'].shape==z['d'].shape==(1793,) and z['z'].dtype==z['d'].dtype==np.float64
                         and np.isfinite(z['z']).all() and np.isfinite(z['d']).all(),'terminal augmented input retained')
        unmatched=set(self.offset_starts)-set(self.offset_events)
        self.require(not unmatched or (len(unmatched)==1 and self.result['status'] in ('instrument_failed','execution_failed','incomplete')),
                     'missing terminal offset event retained only on failure')

    def callback_value(self,call,cycles,session,action_h=None,*,solver_rows=True):
        """Reconstruct each callback algebra from its independently saved calls."""
        y=np.asarray(call['y']); action=call['kind']=='action'; d=call.get('direction')
        if y.size==1793:
            self.require(session.startswith('fold-solver:'),'extended call outside fixed fold solve')
            h=action_h if action_h is not None else (1e-3 if session.endswith(':primary') else 5e-4)
            base=y[:897]; vector=y[897:]; x,temp=self.unpack(base)
            c=self.normalization
            if not action:
                self.require(len(cycles)==1,'extended residual map count')
                q=cycles[0]; self.expect_cycle(q,x,temp,v=vector,label='fold-residual')
                value=np.r_[(q['z']['states'][-1]-x)/self.scale,q['z']['tangents'][-1]-vector,c@vector-1]
                return self.solver_fold_rows(value) if solver_rows else value
            dy=d[:897]; dv=d[897:]
            self.require(len(cycles)==1+2*(norm(dy)>0)+(norm(dv)>0),'extended action nested map count')
            q=cycles[0]; self.expect_cycle(q,x,temp,v=dy[:-1]*self.scale,dlog=dy[-1]*DELTA,label='mixed-jvp')
            recorded=self.recorded_offset_norm(base,dy,h,cycles,complete=True,full_y=y,full_d=d)
            top=(q['z']['tangents'][-1]-dy[:-1]*self.scale)/self.scale
            index=1; mixed=np.zeros(DIM)
            if norm(dy)>0:
                reduction=recorded if recorded is not None else self.offset_norm(base,dy,h,[(1,cycles[index]),(-1,cycles[index+1])])
                xp,tp=self.unpack(base+h*dy/reduction); xm,tm=self.unpack(base-h*dy/reduction)
                self.expect_cycle(cycles[index],xp,tp,v=vector,label='fold-fd-plus')
                self.expect_cycle(cycles[index+1],xm,tm,v=vector,label='fold-fd-minus')
                mixed=(cycles[index]['z']['tangents'][-1]-cycles[index+1]['z']['tangents'][-1])*(reduction/(2*h)); index+=2
            if norm(dv)>0:
                self.expect_cycle(cycles[index],x,temp,v=dv,label='fold-vector')
                mixed+=cycles[index]['z']['tangents'][-1]-dv
            value=np.r_[top,mixed,c@dv]
            return self.solver_fold_rows(value) if solver_rows else value
        if y.size==DIM:
            self.require(session.startswith('native-root:'),'fixed root call identity')
            temp=self.result['B_log10_temperature'] if session.endswith('-B') else N
            y=np.r_[y,(temp-H)/DELTA]
            if action:d=np.r_[d,0.]
        self.require(y.size==DIM+1 and len(cycles)==1,'scaled callback shape/count')
        x,temp=self.unpack(y); q=cycles[0]
        if action:
            self.expect_cycle(q,x,temp,v=d[:-1]*self.scale,dlog=d[-1]*DELTA,label='mixed-jvp')
            return (q['z']['tangents'][-1]-d[:-1]*self.scale)/self.scale
        self.expect_cycle(q,x,temp,label='residual')
        return (q['z']['states'][-1]-x)/self.scale

    def failed_callback_inputs(self,call,cycles,session,action_h=None):
        """A failed action may end within its fixed nested-call schedule."""
        y=np.asarray(call['y']);action=call['kind']=='action';d=call.get('direction');wanted=[]
        def add(label,point,vector=None,temperature_direction=0.):
            x,t=self.unpack(point);wanted.append((label,x,t,vector,temperature_direction))
        if y.size==1793:
            self.require(session.startswith('fold-solver:'),'failed extended callback identity')
            base=y[:897];v=y[897:];h=action_h if action_h is not None else (1e-3 if session.endswith(':primary') else 5e-4)
            if not action:add('fold-residual',base,v)
            else:
                dy=d[:897];dv=d[897:]
                recorded=self.recorded_offset_norm(base,dy,h,cycles,complete=False,full_y=y,full_d=d)
                add('mixed-jvp',base,dy[:-1]*self.scale,dy[-1]*DELTA)
                if norm(dy)>0:
                    offsets=[(sign,cycles[i]) for i,sign in ((1,1),(2,-1)) if i<len(cycles)]
                    reduction=recorded if recorded is not None else (self.offset_norm(base,dy,h,offsets) if offsets else norm(dy))
                    add('fold-fd-plus',base+h*dy/reduction,v)
                    add('fold-fd-minus',base-h*dy/reduction,v)
                if norm(dv)>0:add('fold-vector',base,dv)
        else:
            if y.size==DIM:
                self.require(session.startswith('native-root:'),'failed native callback identity')
                t=self.result['B_log10_temperature'] if session.endswith('-B') else N
                y=np.r_[y,(t-H)/DELTA]
                if action:d=np.r_[d,0.]
            self.require(y.size==897,'failed scaled callback shape')
            if action:add('mixed-jvp',y,d[:-1]*self.scale,d[-1]*DELTA)
            else:add('residual',y)
        self.require(len(cycles)<=len(wanted),'failed callback exceeded fixed nested calls')
        for i,(raw,(label,x,t,v,dt)) in enumerate(zip(cycles,wanted)):
            self.expect_cycle(raw,x,t,v=v,dlog=dt,label=label,complete=False)
            if raw['event']['failed'] is not None:self.require(i==len(cycles)-1,'callback continued after failed map')

    def solver_call_links(self,session,result):
        calls=result['calls']; observed=self.solver_calls[session]
        self.require([c['id'] for c in calls]==list(range(len(calls))),'sequential solver call IDs')
        self.require(set(observed).issubset(set(range(len(calls)))),'orphan callback event')
        counts=Counter()
        for call in calls:
            ident=call['id']; self.require(ident in observed,'missing callback event')
            evidence=observed[ident]
            if 'end' in evidence:self.equal(call,evidence['end'],'callback/result record equality')
            else:self.require(result['status']=='failed','missing callback end on successful solver')
            if call['called']:counts[call['kind']+'_calls']+=1
            if call['status']=='completed':
                self.close(call['output'],self.callback_value(call,evidence['cycles'],session),'callback output from saved raw actions',rtol=5e-12,atol=2e-13)
            elif call['called']:
                self.require(call['status']=='failed' and result['status']=='failed','failed callback classification')
                self.failed_callback_inputs(call,evidence['cycles'],session)
            else:self.require(not evidence['cycles'],'uninvoked callback produced raw calls')
        self.equal(result['counts'],dict(residual_calls=counts['residual_calls'],action_calls=counts['action_calls']),'callback attempted counts')
        self.require(counts['residual_calls']<=result['config']['max_residual_calls'] and counts['action_calls']<=result['config']['max_action_calls'],'callback caps')

    def gates(self,residual,blocks,config):
        checks={k:dict(norm=norm(residual[a:b]),tolerance=tol) for k,(a,b,tol) in blocks.items()}
        for row in checks.values():row['passed']=row['norm']<=row['tolerance']
        full=norm(residual)
        return dict(full_norm=full,full_tolerance=config['residual_tolerance'],
                    full_passed=full<=config['residual_tolerance'],blocks=checks,
                    passed=bool(full<=config['residual_tolerance'] and all(v['passed'] for v in checks.values())))

    def call_output(self,calls,ident,y,kind,*,direction=None,predictor=None,tangent=None):
        self.require(type(ident) is int and 0<=ident<len(calls),'referenced call ID')
        call=calls[ident]
        self.require(call['kind']==kind and call['status']=='completed','referenced completed callback')
        self.exact(call['y'],y,'callback base fixed for linear/Newton solve')
        if direction is not None:self.exact(call['direction'],direction,'linear action direction')
        output=call['output']
        if tangent is not None:
            extra=tangent@(direction if kind=='action' else y-predictor)
            output=np.r_[output,extra]
        return output

    def linear(self,row,y,rhs,calls,config,*,predictor=None,tangent=None):
        self.close(row['rhs'],rhs,'linear exact right-hand side')
        self.close(row['rhs_norm'],norm(rhs),'linear rhs norm')
        actions=row['actions']; self.require(len(actions)<=config['max_linear_actions'],'linear action budget includes fresh')
        self.require(len(row['krylov_residuals'])<=config['gmres_restart'] and all(math.isfinite(v) and v>=0 for v in row['krylov_residuals']),'GMRES inner count/finite residuals')
        for i,action in enumerate(actions):
            wanted=self.call_output(calls,action['call_id'],y,'action',direction=action['direction'],predictor=predictor,tangent=tangent)
            self.close(action['output'],wanted,'linear recorded full action')
            self.require(action['fresh'] is (i==len(actions)-1 and 'fresh_action' in row),'only final action is fresh')
        if 'fresh_action' not in row:
            self.require(not row.get('passed',False),'linear solve qualified without fresh check')
            return False
        self.require(actions and actions[-1]['fresh'],'missing fresh full action')
        self.exact(actions[-1]['direction'],row['solution'],'fresh action on actual solution')
        fresh=actions[-1]['output']; error=fresh-rhs
        limit=max(config['linear_absolute_tolerance'],config['linear_relative_tolerance']*norm(rhs))
        passed=row['gmres_info']==0 and norm(error)<=limit
        self.close(row['fresh_action'],fresh,'fresh full action duplicate')
        self.close(row['residual'],error,'fresh full linear residual')
        self.close(row['residual_norm'],norm(error),'fresh full linear residual norm')
        self.close(row['residual_limit'],limit,'fresh full linear tolerance')
        self.close(row['relative_residual'],norm(error)/max(norm(rhs),1e-300),'fresh full relative residual')
        self.require(row['passed']==passed,'GMRES status AND fresh residual gate')
        expected='gmres_not_converged' if row['gmres_info']!=0 else ('qualified' if passed else 'fresh_linear_residual_failed')
        self.require(row['reason']==expected,'linear failure reason')
        return passed

    def newton(self,row,initial,blocks,calls,config,*,predictor=None,tangent=None):
        self.exact(row['initial_y'],initial,'Newton initial state')
        if 'residual' not in row:
            self.require(row['status'] in ('failed','unresolved'),'Newton missing initial residual status')
            return
        iterations=row['iterations']; self.require(len(iterations)<=config['max_newton_iterations'],'Newton iteration cap')
        # The first residual is retained; accepted damping residuals are carried.
        if iterations:
            first_y=iterations[0]['y']; first_r=iterations[0]['residual']
        else:first_y=row['y']; first_r=row['residual']
        self.exact(first_y,initial,'Newton first base')
        initial_calls=[c for c in calls if c['purpose']=='initial_residual' and c['status']=='completed' and np.array_equal(c['y'],initial)]
        self.require(bool(initial_calls),'Newton initial residual callback')
        # More than one identical residual may occur in different attempts; all
        # satisfy the same exact base, but the raw call ledger keeps them distinct.
        raw=initial_calls[0]['output']
        if tangent is not None:raw=np.r_[raw,tangent@(initial-predictor)]
        self.close(first_r,raw,'Newton initial full residual')
        y=first_y; residual=first_r
        for index,it in enumerate(iterations):
            self.require(it['index']==index,'Newton iteration index')
            self.exact(it['y'],y,'Newton carried iterate')
            self.close(it['residual'],residual,'Newton carried residual')
            gates=self.gates(residual,blocks,config); self.equal(it['gates'],gates,'Newton current gates')
            self.require(not gates['passed'],'Newton action after convergence')
            if not it['linear']:
                self.require(it['status'] in ('failed','unresolved'),'missing linear failure record'); break
            linear_ok=self.linear(it['linear'],y,-residual,calls,config,predictor=predictor,tangent=tangent)
            trials=it['trials']
            if not linear_ok:
                self.require(not trials and it['status'] in ('failed','unresolved'),'trials after failed linear solve'); break
            self.require(len(trials)<=len(config['damping']),'damping count')
            accepted=False
            for j,tr in enumerate(trials):
                damping=config['damping'][j]
                self.require(tr['damping']==damping,'fixed damping ladder')
                trial=y+damping*it['linear']['solution']; self.close(tr['y'],trial,'damping literal trial',rtol=2e-14,atol=2e-14)
                if 'residual' not in tr:
                    self.require(tr['status'] in ('failed','unresolved') and j==len(trials)-1,'partial trial status'); break
                r=self.call_output(calls,tr['residual_call_id'],tr['y'],'residual',predictor=predictor,tangent=tangent)
                self.close(tr['residual'],r,'trial actual callback residual')
                g=self.gates(r,blocks,config); self.equal(tr['gates'],g,'trial full/block gates')
                accept=g['passed'] or norm(r)<=(1-config['sufficient_decrease']*damping)*norm(residual)
                self.require(tr['accepted']==accept and tr['status']==('accepted' if accept else 'rejected'),'Armijo-or-full gate')
                if accept:
                    self.require(j==len(trials)-1,'trial evaluated after accepted damping')
                    y=tr['y']; residual=r; accepted=True; break
            if accepted:self.require(it['status']=='accepted','accepted Newton iteration status')
            else:
                self.require(it['status'] in ('unresolved','failed'),'exhausted/partial iteration status')
                self.require(index==len(iterations)-1,'continued after unaccepted iteration')
        self.exact(row['y'],y,'Newton terminal state')
        self.close(row['residual'],residual,'Newton terminal residual')
        terminal=self.gates(residual,blocks,config)
        if row['status']=='converged':
            self.equal(row['gates'],terminal,'Newton terminal gates')
            self.require(terminal['passed'] and row['reason']=='full_residual_qualified','Newton convergence is complete-system convergence')
        elif row['status']=='unresolved':
            self.require(not terminal['passed'],'unresolved solver already passed full gates')
        else:self.require(row['status']=='failed','Newton terminal status')

    def tangent(self,row,y,reference,calls,config):
        self.exact(row['reference'],reference,'fixed tangent border')
        rhs=np.zeros_like(y); rhs[-1]=1.
        if not row['linear']:
            self.require(row['status'] in ('failed','unresolved'),'missing tangent solve'); return None
        good=self.linear(row['linear'],y,rhs,calls,config,predictor=y,tangent=reference)
        if not good:
            self.require(row['status'] in ('failed','unresolved'),'failed tangent linear status'); return None
        if 'tangent' not in row:
            self.require(row['status'] in ('failed','unresolved'),'missing normalized tangent'); return None
        direction=row['linear']['solution']; unit=direction/norm(direction)
        if unit@reference<0:unit=-unit
        self.close(row['unnormalized'],direction,'tangent unnormalized solution')
        self.close(row['tangent'],unit,'one explicit tangent normalization',rtol=3e-14,atol=3e-14)
        actual=self.call_output(calls,row['normalized_action_call_id'],y,'action',direction=row['tangent'])
        self.close(row['normalized_action'],actual,'fresh normalized tangent action')
        error=abs(norm(row['tangent'])-1); residual=norm(actual); orient=float(row['tangent']@reference)
        self.close(row['norm_error'],error,'tangent unit error',atol=2e-15)
        self.close(row['null_residual_norm'],residual,'tangent null residual')
        self.close(row['orientation'],orient,'tangent orientation')
        passed=error<=config['tangent_norm_tolerance'] and residual<=config['tangent_tolerance'] and orient>0
        self.require(row['status']==('qualified' if passed else 'unresolved'),'fresh tangent gate')
        return row['tangent'] if passed else None

    def solver(self,session,result):
        config=result['config']; self.solver_call_links(session,result)
        for key,value in dict(max_newton_iterations=6,gmres_restart=10,max_linear_actions=12,max_residual_calls=128,max_action_calls=1024,
                              linear_relative_tolerance=1e-3,linear_absolute_tolerance=1e-14,sufficient_decrease=1e-4).items():
            self.require(config[key]==value,'fixed solver setting '+key)
        self.equal(config['damping'],[1.,.5,.25],'fixed damping')
        calls=result['calls']
        if result['kind']=='square_system':
            initial=result.get('initial_y')
            if initial is None:
                self.require(result['status']=='failed','missing square initial iterate'); return
            if session.startswith('fold-solver:'):
                blocks=({'orbit':(0,896,1e-10/self.scale),'normalization':(896,897,1e-10),'null':(897,1793,1e-8)} if self.amended else
                        {'orbit':(0,896,1e-10/self.scale),'null':(896,1792,1e-8),'normalization':(1792,1793,1e-10)})
                expected=self.arrays(self.find_npz('interpolated-seed'))['z']
                self.exact(initial,expected,'common original fold seed for both refinements')
                self.require(config['residual_tolerance']==1e-8,'fold full tolerance')
            else:
                blocks={'equations':(0,896,1e-10/self.scale)}
                self.require(config['residual_tolerance']==1e-10/self.scale,'native full tolerance')
            self.equal(result['residual_blocks'],{k:list(v) for k,v in blocks.items()},'exact-cover residual blocks')
            self.newton(result,initial,blocks,calls,config)
            return
        self.require(result['kind']=='pseudoarclength_continuation','known solver kind')
        for key,value in dict(equation_tolerance=1e-10/self.scale,arc_tolerance=1e-10,residual_tolerance=1e-8,
                              tangent_tolerance=1e-8,tangent_norm_tolerance=1e-12,max_correction_to_step_ratio=1.,minimum_tangent_dot=.5,max_rejected_steps=4).items():
            self.require(config[key]==value,'fixed continuation setting '+key)
        first=session=='branch-solver'
        self.require(config['max_attempted_steps']==(16 if first else 8) and config['max_accepted_steps']==(12 if first else 4),'fixed continuation envelope')
        self.require(result['stop_on_turn'] is True,'must stop on first qualified turn')
        blocks={'equation':(0,896,1e-10/self.scale),'arc':(896,897,1e-10)}
        if not result['points']:
            self.require(result['status'] in ('failed','unresolved'),'empty branch terminal status'); return
        points=result['points']; y=points[0]['y']
        seed=self.call_output(calls,result['seed_residual_call_id'],y,'residual')
        self.close(result['seed_residual'],seed,'literal branch seed residual')
        self.require(norm(seed)<=min(config['equation_tolerance'],config['residual_tolerance']),'qualified branch seed')
        tangent=self.tangent(result['initial_tangent'],y,result['initial_tangent']['reference'],calls,config)
        self.require(tangent is not None,'branch has points without initial tangent')
        self.exact(points[0]['tangent'],tangent,'seed tangent')
        accepted=0; rejected=0; ladder=0; turns=[]
        self.require(len(result['attempts'])<=config['max_attempted_steps'],'continuation attempts bound')
        for i,attempt in enumerate(result['attempts']):
            self.require(attempt['index']==i and attempt['base_point']==accepted,'continuation exact base index')
            self.exact(attempt['tangent'],tangent,'fixed predictor tangent')
            step=config['step_sizes'][ladder]
            self.require(attempt['step_size']==step,'fixed step ladder')
            predictor=y+step*tangent; self.close(attempt['predictor'],predictor,'literal arclength predictor',rtol=2e-14,atol=2e-14)
            corrector=attempt['corrector']
            self.newton(corrector,attempt['predictor'],blocks,calls,config,predictor=attempt['predictor'],tangent=tangent)
            fresh=None
            if corrector['status']=='converged':
                distance=norm(corrector['y']-attempt['predictor']); limit=config['max_correction_to_step_ratio']*step
                self.close(attempt['correction_distance'],distance,'correction distance')
                self.close(attempt['correction_limit'],limit,'correction limit')
                self.require(attempt['correction_distance_passed']==(distance<=limit),'local correction distance gate')
                if distance<=limit:
                    fresh=self.tangent(attempt['tangent_update'],corrector['y'],tangent,calls,config)
                    if fresh is not None:
                        dot=float(tangent@fresh); self.close(attempt['tangent_dot'],dot,'consecutive tangent dot')
                        self.require(attempt['tangent_alignment_passed']==(dot>=.5),'local tangent alignment gate')
                        if dot<.5:fresh=None
            if attempt['status']=='accepted':
                self.require(fresh is not None,'accepted step without full local/tangent gates')
                accepted+=1; point=points[accepted]
                self.require(point['index']==accepted and point['attempt']==i and attempt['point_index']==accepted,'accepted point indices')
                self.exact(point['y'],corrector['y'],'accepted branch state')
                self.close(point['residual'],corrector['residual'][:-1],'accepted branch residual')
                self.exact(point['tangent'],fresh,'accepted branch tangent')
                if tangent[-1]*fresh[-1]<0 or (fresh[-1]==0 and tangent[-1]!=0):
                    turns.append(dict(point_indices=[accepted-1,accepted],parameter_tangents=[float(tangent[-1]),float(fresh[-1])],status='numerical_turn_bracket'))
                y=point['y']; tangent=fresh; ladder=0
            elif attempt['status']=='rejected':
                self.require(fresh is None,'rejected step passed every admission gate'); rejected+=1; ladder+=1
            else:
                self.require(result['status'] in ('failed','unresolved') and i==len(result['attempts'])-1,'partial continuation status')
        self.require(len(points)==accepted+1 and result['accepted_steps']==accepted and result['rejected_steps']==rejected,'continuation actual point counts')
        self.equal(result['turn_brackets'],turns,'actual tangent sign crossing brackets')
        if result['reason']=='qualified_tangent_turn':self.require(result['status']=='completed' and len(turns)==1,'qualified first turn')

    def find_npz(self,label):
        paths=list(self.run.glob('*-'+label+'.npz'))
        self.require(len(paths)==1,'unique saved array '+label)
        return paths[0].name

    def spectral(self,label,y,alpha,seed):
        z=self.arrays(self.find_npz(label+'-arnoldi'))
        q,h,w=z['q'],z['h'],z['matvecs']; m=w.shape[1]
        self.require(1<=m<=8 and q.shape==(DIM,m+1) and h.shape==(m+1,m) and w.shape==(DIM,m),'Arnoldi raw shapes')
        self.require(all(np.isfinite(x).all() for x in (q,h,w,z['values'],z['vectors'],z['ritz_residuals'])),'Arnoldi finite evidence')
        start=np.random.default_rng(seed).normal(size=DIM); start/=norm(start)
        self.close(q[:,0],start,'fixed PCG64 Arnoldi start',rtol=0,atol=8*np.finfo(float).eps)
        cycles=self.by_label[label+'-action']; self.require(len(cycles)==m,'Arnoldi raw action count')
        x,temp=self.unpack(y)
        for j,call in enumerate(cycles):
            self.expect_cycle(call,x,temp,v=q[:,j],alpha=alpha,label=label+'-action')
            self.exact(w[:,j],call['z']['tangents'][-1],'Arnoldi actual unnormalized action')
            residual=w[:,j].copy(); column=np.zeros(m+1)
            for _ in range(2):
                for i in range(j+1):
                    coefficient=float(q[:,i]@residual);column[i]+=coefficient;residual-=coefficient*q[:,i]
            column[j+1]=norm(residual)
            arithmetic=64*np.finfo(float).eps*max(norm(w[:,j]),1e-14)
            self.close(h[:,j],column,'two-pass MGS coefficients',rtol=1e-12,atol=arithmetic)
            if h[j+1,j]>1e-13:
                # Audit the unnormalized recurrence. Tiny residual normalization
                # can amplify host reduction roundoff; no loose Q replay is used.
                self.close(q[:,j+1]*h[j+1,j],residual,'unnormalized Arnoldi residual',rtol=0,atol=arithmetic)
            else:self.require(j==m-1 and np.count_nonzero(q[:,j+1])==0,'happy-breakdown terminal column')
        orth=norm(q[:,:m].T@q[:,:m]-np.eye(m)); decomposition=norm(w-q@h)/max(norm(w),1e-14)
        basis=orth<=1e-10 and decomposition<=1e-10
        values=z['values']; vectors=z['vectors']
        self.require(values.shape==(m,) and vectors.shape==(DIM,m),'Ritz vector dimensions')
        for i in range(m):
            v=vectors[:,i]; co=np.linalg.lstsq(q[:,:m],v,rcond=None)[0]
            self.close(norm(v),1.,'Ritz full vector normalization',atol=2e-12)
            self.close(q[:,:m]@co,v,'Ritz vector in saved Krylov span',rtol=1e-10,atol=1e-12)
            self.require(norm(h[:m,:m]@co-values[i]*co)<=1e-10*max(norm(h[:m,:m]),1.),'projected Ritz equation')
            self.close(z['ritz_residuals'][i],norm(w@co-values[i]*v),'saved raw-action Ritz residual',rtol=2e-8,atol=2e-12)
        selected=int(np.argmin(abs(values-1))); chosen=vectors[:,selected]
        evidence=sorted(self.run.glob('*-'+label+'-eigenpair.npz'))
        order=np.argsort(-abs(values))[:3]
        self.require(len(evidence)<=len(order),'fresh eigenpair count')
        real=self.by_label[label+'-fresh-real']; imag=self.by_label[label+'-fresh-imag']; ni=0
        pairs=[]
        for j,path in enumerate(evidence):
            i=int(order[j]);e=self.arrays(path.name);v=vectors[:,i];lam=values[i]
            self.exact(e['vector'],v,'fresh eigenpair exact vector')
            self.close(e['eigenvalue'],lam,'fresh eigenvalue')
            self.expect_cycle(real[j],x,temp,v=v.real,alpha=alpha,label=label+'-fresh-real')
            av=real[j]['z']['tangents'][-1].astype(complex)
            if norm(v.imag)>1e-14:
                self.expect_cycle(imag[ni],x,temp,v=v.imag,alpha=alpha,label=label+'-fresh-imag')
                av+=1j*imag[ni]['z']['tangents'][-1];ni+=1
            self.close(e['action'],av,'fresh complex action assembled from real and imaginary calls')
            residual=norm(av-lam*v)/max(norm(av),abs(lam)*norm(v),1e-14)
            pairs.append(dict(index=i,value=[float(lam.real),float(lam.imag)],modulus=float(abs(lam)),normalized_residual=residual,qualified=residual<=1e-6))
        self.require(len(real)==len(evidence) and len(imag)==ni,'fresh eigenpair raw inventory')
        row=dict(basis_passed=basis,pairs=pairs,selected_index=selected,
                 selected_value=[float(values[selected].real),float(values[selected].imag)],
                 observed_other_distance_from_one=float(min(abs(np.delete(values,selected)-1))) if m>1 else None)
        if (self.run/(label+'-spectrum.json')).exists():
            self.require(len(evidence)==len(order) and norm(chosen.imag)<=1e-8 and m>1,'complete real critical-mode record')
            self.equal(self.read(label+'-spectrum.json'),row,'fresh spectral metrics')
        self.spectra[label]=row
        vector=chosen.real/norm(chosen.real) if norm(chosen.real)>0 and norm(chosen.imag)<=1e-8 else None
        return vector,row

    def anchors_and_pilot(self):
        a=self.inputs
        self.prerequisite_gates={}
        for k in (0,1):
            name=f'p{k}-anchor.json'
            if not (self.run/name).exists():continue
            raw=self.one(f'p{k}-anchor');self.expect_cycle(raw,a['p_terminal'][k],P,label=f'p{k}-anchor')
            gate=cycle_metrics(raw['z']['states'])
            self.equal(self.read(name),gate,'literal P primitive gate')
            self.prerequisite_gates[f'p{k}']=gate['passed']
        aligned=[]
        for k in (0,1):
            name=f'parent-{21+k}-bridge.json'
            if not (self.run/name).exists():continue
            raw=self.one(f'parent-{21+k}')
            self.expect_cycle(raw,a['fold_states'][k],a['fold_logtemps'][k],alpha=a['fold_alphas'][k],label=f'parent-{21+k}')
            metrics=cycle_metrics(raw['z']['states']); distances=cyclic_distances(raw['z']['states'][:6],a['p_cycles'][:,1]); shift=int(np.argmin(distances))
            self.equal(self.read(name),dict(gate=metrics,alignment=dict(shift=shift,distances=distances.tolist())),'parent bridge and actual phase alignment')
            self.prerequisite_gates[f'parent-{21+k}']=metrics['passed']
            aligned.append(np.roll(raw['z']['states'][:6],-shift,axis=0))
        self.aligned=aligned
        if not (self.run/'derivative-pilot.json').exists():return
        self.require(len(aligned)==2,'pilot requires both parent bridges')
        y=np.r_[aligned[0][0]/self.scale,(a['fold_logtemps'][0]-H)/DELTA];x,temp=self.unpack(y);alpha=a['fold_alphas'][0]
        mixed=self.one('pilot-mixed');primal=self.one('pilot-primal');repeat=self.one('pilot-repeat')
        rng=np.random.default_rng(74);dx=rng.normal(size=DIM);dx/=norm(dx)
        d=np.r_[dx/self.scale,1/self.scale]
        self.close(mixed['z']['tangents'][0],d[:-1]*self.scale,'pilot direction provenance',rtol=0,atol=8*np.finfo(float).eps)
        self.expect_cycle(mixed,x,temp,v=mixed['z']['tangents'][0],dlog=d[-1]*DELTA,alpha=alpha,label='pilot-mixed')
        self.expect_cycle(primal,x,temp,alpha=alpha,label='pilot-primal');self.expect_cycle(repeat,x,temp,alpha=alpha,label='pilot-repeat')
        gaps=[float(np.max(abs(mixed['z']['states']-primal['z']['states']))),float(np.max(abs(primal['z']['states']-repeat['z']['states'])))]
        self.require(len(self.by_label['pilot-full'])==2,'two fixed cache phases')
        for call,k in zip(self.by_label['pilot-full'],(0,3)):
            self.expect_cycle(call,primal['z']['states'][k],temp,alpha=alpha,label='pilot-full')
            self.require(call['event']['cached'] is False,'actual full-prefix comparison')
            gaps.append(float(np.max(abs(call['z']['states'][-1]-primal['z']['states'][k+1]))))
        rows=[];hs=(4e-5,2e-5,1e-5,5e-6,2.5e-6)
        self.require(len(self.by_label['pilot-fd-plus'])==len(self.by_label['pilot-fd-minus'])==5,'complete signed pilot ladder')
        for i,h in enumerate(hs):
            plus=self.one('pilot-fd-plus',i);minus=self.one('pilot-fd-minus',i)
            for sign,call in ((1,plus),(-1,minus)):
                xx,tt=self.unpack(y+sign*h*d)
                self.close(call['z']['states'][0],xx,'actual mixed finite-difference offset',rtol=0,atol=16*np.finfo(float).eps*max(float(np.max(abs(x))),1.))
                self.expect_cycle(call,call['z']['states'][0],tt,alpha=alpha,label='pilot-fd-plus' if sign==1 else 'pilot-fd-minus')
            fd=(plus['z']['states'][-1]-minus['z']['states'][-1])/(2*h)
            rows.append(dict(h=h,relative_error=norm(fd-mixed['z']['tangents'][-1])/max(norm(mixed['z']['tangents'][-1]),1e-14)))
        end=self.one('pilot-fd-minus',4)['line']
        selected=[q for q in self.by_label['mixed-jvp'] if q['line']>end][:3]
        self.require(len(selected)==3,'mixed action linearity calls')
        u=np.r_[dx/self.scale,0.];v=np.r_[np.zeros(DIM),1/self.scale]; outputs=[]
        for call,di in zip(selected,(u,v,.6*u-.8*v)):
            self.close(call['z']['tangents'][0],di[:-1]*self.scale,'linearity direction',rtol=0,atol=8*np.finfo(float).eps)
            self.expect_cycle(call,x,temp,v=call['z']['tangents'][0],dlog=di[-1]*DELTA,alpha=alpha,label='mixed-jvp')
            outputs.append((call['z']['tangents'][-1]-call['z']['tangents'][0])/self.scale)
        linear=norm(outputs[2]-(.6*outputs[0]-.8*outputs[1]))/max(norm(outputs[2]),1e-14)
        gate=derivative_gate(rows)
        row=dict(passed=bool(max(gaps)<=1e-10 and gaps[1]==0 and gate['passed'] and linear<=1e-10),primal_cache_gaps=gaps,fd_rows=rows,fd_gate=gate,linearity_relative=linear)
        self.equal(self.read('derivative-pilot.json'),row,'independent mixed AD pilot')
        if 'pilot' in self.result:self.equal(self.result['pilot'],row,'result pilot copy')
        self.pilot=row

    def fold_candidate(self,label,alpha):
        name=self.find_npz(label+'-candidate');saved=self.arrays(name);z=saved['z'];c=saved['normalization_covector']
        self.require(z.dtype==np.float64 and z.shape==(1793,) and np.isfinite(z).all(),'fold candidate full shape')
        self.close(c,self.normalization,'fixed parent critical normalization covector',rtol=0,atol=4e-15)
        y=z[:897];v=z[897:];x,temp=self.unpack(y)
        orbit=self.one(label+'-orbit');self.expect_cycle(orbit,x,temp,alpha=alpha,label=label+'-orbit')
        before=[q for q in self.by_label['fold-residual'] if q['line']<orbit['line']]
        self.require(bool(before),'fresh fold residual before orbit gate');fresh=before[-1]
        self.expect_cycle(fresh,x,temp,v=v,alpha=alpha,label='fold-residual')
        rr=np.r_[(fresh['z']['states'][-1]-x)/self.scale,fresh['z']['tangents'][-1]-v,c@v-1]
        gate=cycle_metrics(orbit['z']['states'])
        row=dict(orbit_l2=norm(rr[:896])*self.scale,null_l2=norm(rr[896:1792]),normalization=abs(float(rr[-1])),
                 vector_norm=norm(v),log10_temperature=temp,alpha=float(alpha),cycle=gate)
        row['passed']=bool(gate['passed'] and row['orbit_l2']<=1e-10 and row['null_l2']<=1e-7 and row['normalization']<=1e-10 and 1e-4<norm(v)<1e4)
        self.equal(self.read(label+'-gate.json'),row,'fresh complete fold gate')
        self.exact(saved['cycle'],orbit['z']['states'][:6],'candidate actual ordinary six-cycle')
        self.candidates[label]=dict(z=z,gate=row,cycle=saved['cycle'])
        return row

    def augmented_checks(self,label):
        path=self.find_npz(label+'-augmented-check');z=self.arrays(path)
        seed=self.arrays(self.find_npz('interpolated-seed'))['z'];h=1e-3 if label=='primary' else 5e-4
        serial=int(path.split('-',1)[0]); previous=int(self.find_npz('interpolated-seed').split('-',1)[0])
        if label=='refined':previous=int(self.find_npz('primary-candidate').split('-',1)[0])
        cycles=[c for c in self.cycles if previous<int(c['name'].split('-',1)[0])<serial]
        self.require(len(cycles)==18,'complete augmented action consistency call count')
        rng=np.random.default_rng(75);d=rng.normal(size=1793);d/=norm(d);d2=rng.normal(size=1793);d2/=norm(d2)
        self.close(z['direction'],d,'fixed augmented check direction',rtol=0,atol=8*np.finfo(float).eps)
        self.close(z['second_direction'],d2,'fixed second augmented direction',rtol=0,atol=8*np.finfo(float).eps)
        d=z['direction'];d2=z['second_direction'];session='fold-solver:'+label
        def action(direction,part,hh):
            return self.callback_value(dict(y=seed,kind='action',direction=direction),part,session,action_h=hh,solver_rows=False)
        ah=action(d,cycles[:4],h);half=action(d,cycles[4:8],h/2)
        rp=self.callback_value(dict(y=seed+1e-4*d,kind='residual'),cycles[8:9],session,solver_rows=False)
        rm=self.callback_value(dict(y=seed-1e-4*d,kind='residual'),cycles[9:10],session,solver_rows=False)
        full=(rp-rm)/(2e-4); a2=action(d2,cycles[10:14],h);am=action(.6*d-.8*d2,cycles[14:18],h)
        for key,value in dict(action=ah,half_action=half,full_residual_fd=full,second_action=a2,mixed_action=am).items():self.close(z[key],value,'augmented saved '+key,rtol=5e-12,atol=2e-13)
        relative=norm(ah-half)/max(norm(half),1e-14);gap=norm(ah-full)/max(norm(full),1e-14);linear=norm(am-(.6*ah-.8*a2))/max(norm(am),1e-14)
        expected=dict(fd_step=h,halving_relative=relative,full_residual_fd_relative=gap,linearity_relative=linear,passed=max(relative,gap,linear)<=1e-4)
        self.equal(self.read(label+'-augmented-instrument.json'),expected,'independent augmented action gates')
        self.prerequisite_gates[label+'-augmented']=expected['passed']

    def turn(self,result):
        self.require(result['status']=='completed' and result['reason']=='qualified_tangent_turn' and len(result['turn_brackets'])==1,'qualified turn prerequisite')
        ids=result['turn_brackets'][0]['point_indices'];left,right=[result['points'][i] for i in ids]
        ya,yb=left['y'],right['y'];ta,tb=left['tangent'],right['tangent'];length=norm(yb-ya)
        slopes=np.array([ta[-1],tb[-1]])*length
        self.require(slopes[0]>0>slopes[1],'maximum turn orientation')
        f0,f1=ya[-1],yb[-1];m0,m1=slopes
        coeff=np.array([2*(f0-f1)+m0+m1,3*(f1-f0)-2*m0-m1,m0,f0])
        candidates=[r.real for r in np.roots([3*coeff[0],2*coeff[1],coeff[2]]) if abs(r.imag)<1e-10 and 0<r.real<1 and 6*coeff[0]*r.real+2*coeff[1]<0]
        self.require(len(candidates)==1,'unambiguous Hermite maximum');t=float(candidates[0])
        weights=np.array([2*t**3-3*t*t+1,t**3-2*t*t+t,-2*t**3+3*t*t,t**3-t*t])
        y=weights[0]*ya+weights[1]*length*ta+weights[2]*yb+weights[3]*length*tb
        eta=((coeff[0]*t+coeff[1])*t+coeff[2])*t+coeff[3]
        return dict(log10_temperature=H+DELTA*eta,parameter=t,chord_length=length,coefficients=coeff.tolist(),point_indices=ids,scaled_state_estimate=y[:-1].tolist())

    def geometry(self):
        self.spectra={};self.candidates={};self.pilot=None;self.normalization=None
        self.anchors_and_pilot()
        if list(self.run.glob('*-parent21-arnoldi.npz')):
            y=np.r_[self.aligned[0][0]/self.scale,(self.inputs['fold_logtemps'][0]-H)/DELTA]
            self.normalization,_=self.spectral('parent21',y,self.inputs['fold_alphas'][0],0)
        if (self.run/'parent21-gate.json').exists():self.fold_candidate('parent21',self.inputs['fold_alphas'][0])
        if list(self.run.glob('*-interpolated-seed.npz')):
            seed=self.arrays(self.find_npz('interpolated-seed'))
            w=(ALPHA-self.inputs['fold_alphas'][0])/(self.inputs['fold_alphas'][1]-self.inputs['fold_alphas'][0])
            x=(1-w)*self.aligned[0][0]+w*self.aligned[1][0]
            temp=(1-w)*self.inputs['fold_logtemps'][0]+w*self.inputs['fold_logtemps'][1]
            self.close(seed['weight'],w,'literal interpolation weight')
            self.close(seed['z'],np.r_[x/self.scale,(temp-H)/DELTA,self.normalization],'phase-aligned literal interpolated seed',rtol=3e-14,atol=2e-13)
            self.close(seed['c'],self.normalization,'fixed seed normalization covector',rtol=0,atol=4e-15)
        for label in ('primary','refined'):
            if (self.run/(label+'-augmented-instrument.json')).exists():self.augmented_checks(label)
            if (self.run/(label+'-gate.json')).exists():self.fold_candidate(label,ALPHA)
        for session,record in self.solver_results.items():
            name=({'fold-solver:primary':'primary-solver.json','fold-solver:refined':'refined-solver.json',
                   'branch-solver':'p1-branch.json','branch-refinement':'p1-branch-refinement.json'}.get(session))
            if session.startswith('native-root:'):name=session.split(':',1)[1]+'-solver.json'
            if name and (self.run/name).exists():self.equal(self.read(name,typed=True),record,'solver file/event identity')
            self.solver(session,record)
        if (self.run/'failed-solver.json').exists():
            failed=self.read('failed-solver.json',typed=True)
            self.require(failed['status']=='failed','fatal solver remains failed')
            self.require(any(r.get('error_type')==failed.get('error_type') and r.get('counts')==failed.get('counts') for r in self.solver_results.values()),'failed solver retained in event record')
        gate_status={}
        if 'refined' in self.candidates and list(self.run.glob('*-target-fold-arnoldi.npz')):
            fold=self.candidates['refined']; y=fold['z'][:897]; _,spectrum=self.spectral('target-fold',y,ALPHA,1)
            spread=abs(self.candidates['primary']['gate']['log10_temperature']-fold['gate']['log10_temperature'])
            self.equal(self.result['fold'],dict(primary=self.candidates['primary']['gate'],refined=fold['gate'],refinement_spread=spread,spectrum=spectrum),'result fold evidence')
            gate_status['fold_refinement_or_mode_unresolved']=not (spread<=1e-6 and spectrum['basis_passed'] and all(q['qualified'] for q in spectrum['pairs'])
                and abs(complex(*spectrum['selected_value'])-1)<=1e-6
                and all(q['modulus']<1-1e-6 for q in spectrum['pairs'] if q['index']!=spectrum['selected_index'])
                and spectrum['observed_other_distance_from_one']>=.01)
        branch=self.solver_results.get('branch-solver');refinement=self.solver_results.get('branch-refinement')
        if branch is not None:
            self.equal(decode(self.result['branch']),branch,'result branch copy')
            if branch['points']:
                p=np.r_[self.inputs['p_terminal'][1]/self.scale,(P-H)/DELTA]
                self.exact(branch['points'][0]['y'],p,'original P1 branch seed')
                border=np.zeros(897);border[-1]=1.
                self.exact(branch['initial_tangent']['reference'],border,'literal increasing-temperature initial border')
                step=norm(self.candidates['refined']['z'][:897]-p)/5
                self.equal(branch['config']['step_sizes'],[step,step/2,step/4],'main fixed continuation ladder')
            gate_status['p1_branch_link_unresolved']=not (branch['status']=='completed' and branch['reason']=='qualified_tangent_turn')
        if refinement is not None:
            self.require(branch is not None and not gate_status['p1_branch_link_unresolved'],'refinement requires qualified coarse turn')
            left=branch['points'][branch['turn_brackets'][0]['point_indices'][0]]
            if refinement['points']:self.exact(refinement['points'][0]['y'],left['y'],'refinement same pre-turn representative')
            half=[q for q in branch['attempts'] if q['status']=='accepted'][-1]['step_size']/2
            self.equal(refinement['config']['step_sizes'],[half,half/2,half/4],'halved local refinement ladder')
            self.exact(refinement['initial_tangent']['reference'],left['tangent'],'refinement original tangent')
            gate_status['p1_turn_refinement_unresolved']=not (refinement['status']=='completed' and refinement['reason']=='qualified_tangent_turn')
        if 'branch_link' in self.result:
            coarse=self.turn(branch);fine=self.turn(refinement)
            temperatures=[self.candidates[k]['gate']['log10_temperature'] for k in ('primary','refined')]+[coarse['log10_temperature'],fine['log10_temperature']]
            spread=max(temperatures)-min(temperatures)
            states=[np.array(q['scaled_state_estimate'])*self.scale for q in (coarse,fine)]
            states += [self.unpack(self.candidates[k]['z'][:897])[0] for k in ('primary','refined')]
            distances=[norm(a-b) for i,a in enumerate(states) for b in states[i+1:]]
            linked=dict(passed=spread<=1e-6 and max(distances)<=1e-6,coarse=coarse,refined=fine,temperature_spread=spread,state_spread_l2=max(distances))
            self.equal(self.result['branch_link'],linked,'full state and all-four temperature branch link')
            gate_status['fold_branch_location_disagreement']=not linked['passed']
        sides=[]
        if refinement and refinement['turn_brackets']:
            for k,index in enumerate(refinement['turn_brackets'][0]['point_indices']):
                if not list(self.run.glob(f'*-branch-side-{k}-arnoldi.npz')):continue
                vector,spec=self.spectral(f'branch-side-{k}',refinement['points'][index]['y'],ALPHA,k)
                critical=complex(*spec['selected_value']); alignment=abs(float(vector@self.normalization))
                qualified=[q for q in spec['pairs'] if q['index']==spec['selected_index'] and q['qualified']]
                passed=bool(spec['basis_passed'] and len(qualified)==1 and abs(critical.imag)<=1e-8 and (abs(critical)<1-1e-5 if k==0 else critical.real>1+1e-5)
                            and alignment>.5 and all(q['qualified'] and q['modulus']<1-1e-6 for q in spec['pairs'] if q['index']!=spec['selected_index']))
                sides.append(dict(passed=passed,spectrum=spec,critical_alignment=alignment))
        if 'branch_sides' in self.result:
            self.require(len(sides)==2,'complete bracket-side diagnostics')
            self.equal(self.result['branch_sides'],sides,'fresh incoming/returning mode gates')
            gate_status['branch_stability_change_unresolved']=not all(q['passed'] for q in sides)
        if 'comparison_margin' in self.result.get('fold',{}):
            margin=max(1e-6,5*self.result['branch_link']['temperature_spread'])
            foldt=self.candidates['refined']['gate']['log10_temperature'];order=P+margin<foldt<N-margin
            self.close(self.result['fold']['comparison_margin'],margin,'fixed full-range fold margin')
            self.require(self.result['fold']['P_below_N_above']==order,'P/N directional ordering')
            gate_status['fold_margin_unresolved']=not order
        native=[]
        if 'B_log10_temperature' in self.result:
            B=(P+self.candidates['refined']['gate']['log10_temperature'])/2
            self.require(self.result['B_log10_temperature']==B,'B is exact frozen midpoint')
            for history,temp,label in ((0,B,'native-0-B'),(1,B,'native-1-B'),(0,N,'native-0-N')):
                record=self.solver_results.get('native-root:'+label)
                if record is None:continue
                initial=self.inputs['p_terminal'][history]/self.scale
                if label=='native-0-N':initial=native[0]['root']['y']
                if 'initial_y' in record:self.exact(record['initial_y'],initial,'native original source history')
                gate_status[label+'_unresolved']=record['status']!='converged'
                if not (self.run/(label+'-gate.json')).exists():continue
                call=self.one(label+'-orbit');self.expect_cycle(call,record['y']*self.scale,temp,label=label+'-orbit')
                gate=cycle_metrics(call['z']['states']);self.equal(self.read(label+'-gate.json'),gate,'native fresh primitive-six gate')
                native.append(dict(root=record,cycle=call['z']['states'][:6],gate=gate))
                gate_status[label+'_gate_failed']=not gate['passed']
        if 'native_controls' in self.result:
            self.require(len(native)==3,'all native controls preserved')
            saved=self.arrays(self.find_npz('native-controls'))
            self.exact(saved['B_cycles'],np.stack([q['cycle'] for q in native[:2]]),'both native B cycles')
            self.exact(saved['N_survivor_cycle'],native[2]['cycle'],'native N survivor cycle')
            self.require(float(saved['B_logtemp'])==self.result['B_log10_temperature'],'native B physical control')
            self.exact(saved['fold_z'],self.candidates['refined']['z'],'native controls bind selected fold')
            separation=float(cyclic_distances(native[0]['cycle'],native[1]['cycle']).min())
            self.equal(self.result['native_controls'],dict(B_cycle_separation=separation,all_primitive_six=True),'all-alignment native distinction')
            gate_status['native_B_distinction_unresolved']=separation<=1e-4
        for label in ('primary','refined'):
            record=self.solver_results.get('fold-solver:'+label)
            if record is not None:gate_status[label+'_fold_unresolved']=record['status']!='converged'
            if label in self.candidates:gate_status[label+'_fold_gate_failed']=not self.candidates[label]['gate']['passed']
        outcome=self.result['outcome']
        if self.result['status']=='completed':
            if outcome=='geometry_qualified_return_for_audit':
                self.require(self.pilot is not None and self.pilot['passed'] and self.candidates.get('parent21',{}).get('gate',{}).get('passed'), 'qualified parent/pilot prerequisites')
                wanted=('p0','p1','parent-21','parent-22','primary-augmented','refined-augmented')
                self.require(all(self.prerequisite_gates.get(k) is True for k in wanted),'all anchor/bridge/augmented prerequisites')
                parent=self.spectra.get('parent21',{})
                self.require(parent.get('basis_passed') is True and parent.get('pairs') and parent['pairs'][0]['qualified'] is True,'parent fresh-mode prerequisite')
                self.require(len(native)==3 and len(sides)==2 and 'branch_link' in self.result and all(not v for v in gate_status.values()),'every geometry admission gate')
            elif outcome=='pilot_qualified':
                self.require(self.source['arguments']['pilot_only'] is True and self.pilot['passed'] and self.candidates['parent21']['gate']['passed'],'bounded pilot-only outcome')
            elif outcome=='finite_geometry_unresolved':
                self.require(self.result.get('root_error','').startswith('NumericalUnresolved:'),'finite ambiguity retains original category')
            else:self.require(outcome in gate_status and gate_status[outcome],'scientific stop matches failed gate')
        self.gate_status=gate_status

    def counters(self):
        counts=self.result['counts'];self.equal(self.result['caps'],CAPS,'unchanged prospective caps')
        for key in ('ordinary','jvp_cycles','augmented','model_invocations'):
            self.require(type(counts[key]) is int and counts[key]>=0,'counter type '+key)
        for key in ('ordinary','jvp_cycles','augmented'):self.require(counts[key]<=CAPS[key],'bounded '+key)
        ordinary=len(self.prefixes)+sum(c['event']['steps_completed'] for c in self.cycles if not c['event']['differentiated'])
        invocations=len(self.prefixes)+sum(c['event']['steps_completed'] for c in self.cycles)
        jvps=sum(c['event']['differentiated'] for c in self.cycles)
        uncertain=[c for c in self.cycles if c['event']['failed'] is not None and c['event']['steps_completed']<c['planned_steps']]
        # A failed update can be counted before returning any arrays. No second
        # unseen call is allowed, and complete runs require exact equality.
        tail=1 if self.result['status'] not in ('completed','preflight_passed') else 0
        self.require(jvps<=counts['jvp_cycles']<=jvps+tail,'attempted six-step JVP count')
        self.require(ordinary<=counts['ordinary']<=ordinary+min(1,len(uncertain)+tail),'ordinary attempted-call reconciliation')
        self.require(invocations<=counts['model_invocations']<=invocations+min(1,len(uncertain)+tail),'model invocation reconciliation')
        completed_checks=sum((self.run/(name+'-augmented-instrument.json')).exists() for name in ('primary','refined'))*4
        solver_aug=sum(call['called'] and call['kind']=='action' and np.asarray(call['y']).size==1793 for result in self.solver_results.values() for call in result['calls'])
        known=completed_checks+solver_aug
        if self.result['status']=='completed' and self.result['outcome']!='finite_geometry_unresolved':self.require(counts['augmented']==known,'complete augmented count')
        else:self.require(known<=counts['augmented']<=known+4,'partial augmented attempt count')
        self.require(counts['elapsed_seconds']>=self.last_event_seconds and counts['peak_rss_bytes']>0,'elapsed/high-water metadata')
        if self.result['status']=='completed':
            self.require(counts['elapsed_seconds']<=43200 and counts['peak_rss_bytes']<CAPS['rss_bytes'],'completed geometry resource gates')
        if 'pilot_counts' in self.result:self.require(self.result['pilot_counts']['elapsed_seconds']<=1800,'budgeted pilot completion')
        prior=self.case['prior_counts'] if self.amended else {}
        cumulative=dict(counts)
        for key in ('ordinary','jvp_cycles','augmented','model_invocations','elapsed_seconds'):
            cumulative[key]+=prior.get(key,0)
        cumulative['peak_rss_bytes']=max(counts['peak_rss_bytes'],prior.get('peak_rss_bytes',0))
        if self.amended or 'case_counts' in self.result:
            self.require(self.result['case_counts']==cumulative,'exact fresh plus accepted prior case counters')
        for key in ('ordinary','jvp_cycles','augmented'):
            self.require(cumulative[key]<=CAPS[key],'original cumulative case cap '+key)
        if self.amended:
            started=self.result['started_unix'];finished=self.result['finished_unix']
            original=datetime.strptime(self.case['campaign_started_utc'],'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
            self.require(math.isfinite(started) and math.isfinite(finished) and original<=started<=finished,'amended attempt wall chronology')
            if self.result['status'] in ('completed','preflight_passed'):
                self.require(finished<=self.case['geometry_deadline_unix'] and cumulative['elapsed_seconds']<=CAPS['seconds']
                             and cumulative['peak_rss_bytes']<CAPS['rss_bytes'],'original absolute and cumulative geometry limits')
        self.case_counts=cumulative
        for path in self.run.glob('checkpoint-*.json'):
            row=self.read(path);c=row['summary'].get('counts')
            if c:
                self.require(all(c[k]<=counts[k] for k in ('ordinary','jvp_cycles','augmented','model_invocations')),'checkpoint counters cannot exceed terminal counters')
        self.counter_reconciliation=dict(saved_completed_model_invocations=invocations,attempted_model_invocations=counts['model_invocations'],
                                         failed_unreturned_attempts=counts['model_invocations']-invocations,
                                         ordinary=counts['ordinary'],six_step_jvps=counts['jvp_cycles'],augmented=counts['augmented'])

    def backend(self):
        planned=[]
        for value in (ALPHA,*self.inputs['fold_alphas'],self.inputs['fold_alphas'][0],ALPHA):
            if not planned or float(value)!=planned[-1]:planned.append(float(value))
        observed=[e['alpha'] for e in self.alphas]
        self.require(observed==planned[:len(observed)],'literal alpha selection sequence and fixed target alpha')
        if not (self.run/'backend-readiness.json').exists():
            self.require(not self.cycles or self.synthetic,'raw calls without backend readiness');return
        row=self.read('backend-readiness.json')
        self.require(row['threads']==[1,1] and all(row['thread_environment'][k]=='1' for k in THREADS),'producer numerical threads')
        model=self.read('model-inputs.json')
        if not self.synthetic:
            pins={'weights':'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe',
                  'config':'18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45',
                  'tokenizer':'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539',
                  'tokenizer_config':'5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583',
                  'vectors':'b630bb7bef06bde87f8e6a7ca5e8672f83d24bc0585bf19048e9e494a4357f41'}
            self.require(model['model']=='Qwen/Qwen2.5-0.5B-Instruct' and model['revision']=='7ae557604adf67be50417f59c2c2f167def9a775','pinned model revision')
            self.require(all(model[k]['sha256']==v for k,v in pins.items()),'recorded pinned model/config/tokenizer/vector hashes')
            self.require(model['prefix_ids']==PREFIX_IDS,'fixed literal prefix tokens')


def audit(run_dir,source_root,source_commit,input_packet,input_packet_sha256,*,synthetic=False,case_context=None,case_context_sha256=None):
    a=Audit(run_dir,synthetic=synthetic)
    raw=a.inventory();a.result=a.read('result.json')
    a.require(a.result['profile']==PROFILE and a.result['source_commit']==source_commit,'result frozen identity')
    a.require(a.result['status'] in ('completed','preflight_passed','execution_failed','instrument_failed','incomplete'),'original result terminal category')
    a.provenance(source_root,source_commit,input_packet,input_packet_sha256)
    a.case_context(case_context,case_context_sha256)
    if a.amended:a.require(raw['raw_bytes']<=a.case['remaining']['raw_bytes'],'cumulative raw evidence allowance')
    transport=a.transport();a.events();a.backend()
    if a.result['status']!='preflight_passed':a.geometry()
    else:
        a.require(a.result['outcome']=='not_run_preflight' and not a.cycles and not a.prefixes,'zero-call preflight')
    a.offset_evidence();a.counters();a.bounded()
    if a.result['status'] in ('instrument_failed','execution_failed','incomplete'):
        root=a.result.get('root_error','')
        a.require(bool(root),'original failure type retained')
        if a.result['status']=='instrument_failed':a.require(root.startswith('InstrumentError:'),'instrument failure category')
        if a.result['status']=='incomplete':a.require(root.split(':',1)[0] in ('LimitError','MemoryError','TimeoutError','KeyboardInterrupt'),'resource/interruption category')
    if not synthetic:
        a.require((transport['exit_status']==0)==(a.result['status'] in ('completed','preflight_passed')),'wrapper/result exit status')
    eligible=(not synthetic and transport['closed_verified_return'] and a.result['status']=='completed' and a.result['outcome']=='geometry_qualified_return_for_audit')
    report=dict(status='audit_passed',integrity_status='passed',profile=PROFILE,synthetic=synthetic,
                source_commit=source_commit,scientific_status=a.result['status'],outcome=a.result['outcome'],
                pulse_eligible=eligible,pulse_admission=dict(eligible=eligible,reasons=[] if eligible else ['Geometry is not a complete, closed, independently qualified production result.']),
                return_status=transport['status'],closed_verified_return=transport['closed_verified_return'],
                counts=a.result['counts'],elapsed_seconds=a.result['counts']['elapsed_seconds'],
                case_counts=a.case_counts,attempt_index=1 if a.amended else 0,
                case_context=a.case['binding'] if a.amended else None,
                prior_counts=a.case['prior_counts'] if a.amended else None,
                qualified_case_context=a.bind(a.run/'qualified-case-context.json') if a.amended else None,
                amendment=AMENDMENT if a.amended else None,
                model_calls=0,network_calls=0,assertions=a.assertions,raw_inventory=raw,transport=transport,
                counter_reconciliation=a.counter_reconciliation,warnings=a.warnings,
                derived_offset_reductions=a.derived_offset_reductions,
                recorded_offset_reductions=a.recorded_offset_reductions,
                scope='Independent saved-evidence algebra and fixed-gate audit. Eight-direction Arnoldi diagnostics do not certify full spectral coverage; native roots do not certify global attraction.',
                fold_logtemp=a.result.get('fold',{}).get('refined',{}).get('log10_temperature'),
                B=a.result.get('B_log10_temperature'),P=P,N=N,
                result=a.bind(a.run/'result.json'),source=a.bind(a.run/'source.json'),checker=a.bind(__file__),
                input_packet=a.packet_binding,input_arrays=a.bind(Path(input_packet).parent/'inputs.npz'),
                original_input_bindings=a.packet['sources'],native_controls=None,
                campaign_started_utc=None,campaign_deadline_unix=None,started_utc=None,attempt_started_utc=None)
    native=list(a.run.glob('*-native-controls.npz'))
    if native:
        a.require(len(native)==1,'unique native controls artifact');report['native_controls']=a.bind(native[0])
    if not synthetic:
        job=a.run.parent
        for key,name in dict(final_manifest='final-manifest.json',completion='completion.json',terminal_process_check='terminal-process-check.json',
                             transfer_closeout='transfer-closeout.json',maintenance_closeout='maintenance-closeout.json',attempt_started_utc='started-utc.txt').items():
            report[key]=a.bind(job/name)
        text=(job/'started-utc.txt').read_text().strip();start=datetime.strptime(text,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        a.require('started-utc.txt' in {r['path'] for r in a.read(job/'final-manifest.json')['files']},'campaign start is in complete returned inventory')
        if a.amended:
            a.require(start.timestamp()<=a.result['started_unix']<a.case['geometry_deadline_unix'],'attempt wrapper start before runner/deadline')
            report.update(started_utc=a.bind(job/'case-started-utc.txt'),
                          campaign_started_utc=a.case['campaign_started_utc'],campaign_deadline_unix=a.case['campaign_deadline_unix'],
                          geometry_deadline_unix=a.case['geometry_deadline_unix'])
        else:
            report.update(started_utc=report['attempt_started_utc'],campaign_started_utc=text,campaign_deadline_unix=start.timestamp()+72000)
    report['assertions']=a.assertions
    report['bindings']=list(a.bindings.values())
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','source-root','output-dir','input-packet'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--input-packet-sha256',required=True)
    p.add_argument('--case-context',type=Path);p.add_argument('--case-context-sha256')
    return p


def main(args):
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def stop(*_):raise TimeoutError('independent geometry audit 600-second limit')
    prior=signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,600)
    try:
        if any(os.environ.get(k)!='1' for k in THREADS):raise AuditError('four external numerical thread limits must equal one')
        report=audit(args.run_dir,args.source_root,args.source_commit,args.input_packet,args.input_packet_sha256,
                     case_context=args.case_context,case_context_sha256=args.case_context_sha256)
        status=0
    except BaseException as error:
        report=dict(status='audit_failed',integrity_status='failed',profile=PROFILE,synthetic=False,
                    pulse_eligible=False,pulse_admission=dict(eligible=False,reasons=[str(error)]),
                    model_calls=0,network_calls=0,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        status=1
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
    report['audit_elapsed_seconds']=time.monotonic()-started
    report['audit_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['utc']=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
    report['invocation']={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    report['thread_environment']={k:os.environ.get(k) for k in THREADS}
    target=args.output_dir/'independent-review.json'
    with target.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],report=str(target),sha256=sha(target),pulse_eligible=report['pulse_eligible'])))
    return status


if __name__=='__main__':raise SystemExit(main(parser().parse_args()))
