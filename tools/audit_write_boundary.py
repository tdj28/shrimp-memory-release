"""Recompute the prospective panel's fate scores from saved single-state trajectories.

Also checks the size and SHA-256 of every raw file the row records cite, and copies the run's own record of the
source files it executed, noting whether each equals the file in this checkout. No model call.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def fate(states, refs):
    a, b = states[-12:-6], states[-6:]
    def gap(x, y):
        return np.linalg.norm(x-y, axis=1).max()
    residual = float(gap(a, b))
    periods_excluded = all(gap(b, np.roll(b, j, axis=0)) > 1e-4 for j in (1,2,3))
    matches = [i for i, r in enumerate(refs)
               if min(gap(b,np.roll(r,j,axis=0)) for j in range(6)) <= 1e-6]
    bit = matches[0] if len(matches)==1 and periods_excluded and residual<=1e-8 else None
    return bit, residual


def cited_files(rows):
    found = {}
    def walk(o):
        if isinstance(o, dict):
            if {'path', 'bytes', 'sha256'} <= set(o):
                require(found.setdefault(o['path'], (o['bytes'], o['sha256'])) == (o['bytes'], o['sha256']), 'Citation conflict')
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(rows)
    return found


def check(run, references=None):
    saved = read(run/'results.json')
    attempt = read(run/'attempt.json')
    plan = read(run/'trial-plan.json')['trials']
    require(saved['model_calls']==1944 and len(plan)==len(saved['rows'])==6, 'Incomplete panel')
    require(saved['protocol']=='boundary' and 'error' not in saved, 'Execution failure')
    input_path = run/'fixed-trial-inputs.npz'
    with np.load(input_path, allow_pickle=False) as z:
        refs=z['references']
    # Verify the extracted references against the canonical file used by the release.
    canonical = references or Path(attempt['arguments']['references'])
    require(sha(canonical)=='d96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0','Reference identity')
    with np.load(canonical, allow_pickle=False) as z:
        require(refs.tobytes()==z['states'].tobytes(),'Reference values')
    records = []
    folded = (0,0,0,1,1,1)
    hold_alpha, hold_temp = 0.07293697809562831, -0.15395151820007266
    for i, (planned,row) in enumerate(zip(plan,saved['rows'],strict=True)):
        offset=(-2e-6,4e-6,12e-6)[i//2]
        duration=(54,102)[i%2]
        require(planned['trial_id']==row['trial_id']==i and row['status']=='completed','Trial identity')
        require(planned['predicted_bit']==row['predicted_bit']==folded[i] and
                planned['duration_only_bit']==row['duration_only_bit']==i%2,'Prediction identity')
        start=refs[0,0]
        require(len(row['stages'])==2,'Stage count')
        for stage,n,alpha in zip(row['stages'],(duration,240),(0.07299164917758323+offset,hold_alpha),strict=True):
            require(stage['steps']==stage['completed_steps']==n and stage['status']=='completed','Stage completion')
            require(stage['cache_comparison']['passed'] is True,'Cache qualification')
            require(stage['controls']==dict(alpha=alpha,log10_temperature=hold_temp),'Stage controls')
            record=stage['trajectory']
            path=run/record['path']
            require(sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],'Trajectory binding')
            with np.load(path, allow_pickle=False) as archive:
                states=archive['states']
                require(float(archive['alpha'])==alpha and float(archive['log10_temperature'])==hold_temp,'Archive controls')
            require(states.dtype==np.float64 and states.shape==(n+1,896) and np.isfinite(states).all(),'State contract')
            require(states[0].tobytes()==start.tobytes(),'State continuity')
            start=states[-1]
        bit,residual=fate(states[1:],refs)
        category='unresolved' if bit is None else 'correct' if bit==folded[i] else 'wrong'
        require(category==row['outcome']['classification'],'Readout disagreement')
        records.append(dict(offset=offset,updates=duration,fold=folded[i],duration_only=i%2,measured=bit,
                            closure=residual, endpoint=row['stages'][0]['trajectory'],hold=row['stages'][1]['trajectory']))
    cited = cited_files(saved['rows'])
    for name, (size, digest) in cited.items():
        path = run/name
        require(path.parent == run and path.stat().st_size == size and sha(path) == digest, 'Raw file binding '+name)
    sources = [dict(path='tools/'+name, sha256=digest, equals_this_checkout=(ROOT/'tools'/name).is_file() and sha(ROOT/'tools'/name) == digest)
               for name, digest in attempt['source_hashes'].items()]
    return dict(kind='prospective-write-boundary', plan='docs/prospective-write-boundary.md', model_calls_by_this_check=0,
                source_commit=attempt['source_commit'], source_commit_repository='private study repository',
                run_started_utc=attempt['utc'], source_files=sources, cited_raw_files_verified=len(cited),
                utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                calls=saved['model_calls'],seconds=saved['elapsed_seconds'],peak_rss_bytes=saved['peak_rss_bytes'],
                rows=records,fold_score=sum(r['fold']==r['measured'] for r in records),
                duration_score=sum(r['duration_only']==r['measured'] for r in records),
                results_sha256=sha(run/'results.json'), plan_sha256=sha(run/'trial-plan.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--references',type=Path,help='Relocated canonical reference-cycles.npz')
    args=parser.parse_args()
    receipt=check(args.run,args.references)
    with args.output.open('x') as out:
        json.dump(receipt,out,indent=2,allow_nan=False)
        out.write('\n')
    print([(r['fold'],r['duration_only'],r['measured']) for r in receipt['rows']])
