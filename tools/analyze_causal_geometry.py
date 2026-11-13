#!/usr/bin/env python3
"""Frozen prefix-only readout comparison on saved public numerical-memory evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import audit_p6_recovery as gates
from streaming_readout import StreamingReadout
import streaming_readout

LABELS=('ad','td')
REFERENCE_SHA='d96f923d537a6bcd6b86be7fe91f29a4cba6a46e5e036c8dd408817281395fe0'
COHORTS=(('fresh-readout',8,600),('transverse-readout',8,600),
    ('transverse-intermediate',8,600),('transverse-amplitude',8,600),
    ('transverse-direction2',8,600),('p6-recovery',26,120))
MAX_SECONDS=120; EXTERNAL_SECONDS=180; MAX_RSS=1024**3; MIN_AVAILABLE=8*1024**3
MAX_OUTPUT=8*1024**2; WORKING_OUTPUT=5*1024**2


def require(value,message):
    if not value: raise ValueError(message)


def binding(path):
    p=Path(path); h=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return dict(path=str(p.absolute()),bytes=p.stat().st_size,sha256=h.hexdigest())


def rss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def utc():return subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()


def plan():
    return dict(schema_version=1,evidence='adapted retrospective saved-evidence analysis',
        cohorts=[dict(name=n,rows=c,updates=s,role='primary' if s==600 else 'separate horizon controls') for n,c,s in COHORTS],
        planned_rows=66,reference_sha256=REFERENCE_SHA,checkpoint_interval=12,consecutive_labels=3,
        geometry='Closest of six phases per labeled reference cycle; exact equal distances erase and reset streak. First three equal checkpoint labels commit once.',
        strict='Existing StreamingReadout: full closure/divisor/unique-cycle gates and three consecutive qualified checkpoints.',
        continuation='Retain every12-update checkpoint through the original120/600 horizon; no model execution or stopping.',
        observer_access='Full896-dimensional states and twelve labeled reference vectors; no final label or future state enters either decision.',
        adaptation='Existing outcomes and prior retrospective observability were already seen. No training, parameter search, held-out or confirmatory claim.',
        limits=dict(threads=1,nice_at_least=10,rss_bytes=MAX_RSS,internal_seconds=MAX_SECONDS,
            external_seconds=EXTERNAL_SECONDS,total_output_bytes=MAX_OUTPUT,working_output_bytes=WORKING_OUTPUT,
            minimum_available_bytes=MIN_AVAILABLE),
        scope='Label visibility is distinct from state readiness for another write. No achieved call/time saving, safe stopping, population/all-phase reliability, causal-inference or language-channel claim.')


class Streak:
    def __init__(self):self.previous=None;self.count=0;self.decision=None;self.steps=0
    def push(self,step,label):
        require(type(step) is int and step==self.steps+12,'checkpoint must be the next twelve-update grid point')
        require(label in (*LABELS,None),'invalid checkpoint label')
        self.steps=step
        self.count=self.count+1 if label is not None and label==self.previous else (1 if label else 0)
        self.previous=label
        if self.count==3 and self.decision is None:self.decision=dict(steps=step,label=label)


def geometry(state,refs):
    require(state.shape==(896,) and refs.shape==(2,6,896) and state.dtype==refs.dtype==np.float64
            and np.isfinite(state).all() and np.isfinite(refs).all(),'invalid float64 geometry inputs')
    distances=np.linalg.norm(refs-state,axis=2).min(axis=1)
    require(np.isfinite(distances).all(),'nonfinite geometry distance')
    label=None if distances[0]==distances[1] else LABELS[int(distances[1]<distances[0])]
    return label,distances.tolist()


def observe(states,refs,check=lambda:None):
    """Only current/past states reach the observers; state zero is never pushed."""
    require(states.ndim==2 and states.shape[1:]==(896,) and len(states)>=13 and (len(states)-1)%12==0
            and states.dtype==np.float64 and np.isfinite(states).all(),'invalid complete trajectory')
    strict=StreamingReadout(refs); simple=Streak(); checkpoints=[]
    try:
        for step,state in enumerate(states[1:],1):
            check(); item=strict.push(state)
            if item is None:continue
            label,distances=geometry(state,refs);simple.push(step,label)
            checkpoints.append(dict(steps=step,geometry_label=label,geometry_label_distances=distances,
                geometry_absolute_margin=abs(distances[0]-distances[1]),strict_label=item['qualified_label'],
                maximum_closure_residual=item['maximum_closure_residual'],reference_matches=item['reference_matches']))
    except BaseException as error:
        error.partial_observation=dict(geometry_decision=simple.decision,strict_decision=strict.decision,
            checkpoints=checkpoints,consumed_updates=strict.steps)
        raise
    return dict(geometry_decision=simple.decision,strict_decision=strict.decision,checkpoints=checkpoints)


def qualified(outcome):
    matches=outcome['reference_matches']
    return matches[0] if outcome['cycle_gate_passed'] and len(matches)==1 else None


def score_decision(decision,final,checkpoints):
    require(final in (*LABELS,None),'invalid final label')
    if decision is None:
        return dict(outcome='no_decision',final_unknown=final is None,later_geometry_opposite_steps=[],
            later_geometry_tie_steps=[],later_opposite_qualified_steps=[],later_unqualified_steps=[])
    require(decision['label'] in LABELS and any(c['steps']==decision['steps'] for c in checkpoints),'decision outside observed grid')
    later=[c for c in checkpoints if c['steps']>decision['steps']];label=decision['label']
    return dict(outcome='unknown_final' if final is None else ('correct' if label==final else 'wrong'),
        final_unknown=final is None,
        later_geometry_opposite_steps=[c['steps'] for c in later if c['geometry_label'] not in (None,label)],
        later_geometry_tie_steps=[c['steps'] for c in later if c['geometry_label'] is None],
        later_opposite_qualified_steps=[c['steps'] for c in later if c['strict_label'] not in (None,label)],
        later_unqualified_steps=[c['steps'] for c in later if c['strict_label'] is None])


def summarize(rows):
    require(rows and all(r['status']=='completed' for r in rows),'summary requires every completed planned row')
    result=dict(planned_rows=len(rows),qualified_finals=sum(r['final_label'] is not None for r in rows),
        unknown_finals=sum(r['final_label'] is None for r in rows))
    for method in ('geometry','strict'):
        values=[r[method+'_score'] for r in rows];q=result['qualified_finals']
        correct=sum(v['outcome']=='correct' for v in values);wrong=sum(v['outcome']=='wrong' for v in values)
        no_known=sum(v['outcome']=='no_decision' and not v['final_unknown'] for v in values)
        accepted=correct+wrong
        result[method]=dict(correct=correct,wrong=wrong,no_decision_qualified=no_known,
            qualified_denominator=q,correct_fraction_all_qualified=correct/q if q else None,
            accepted_qualified=accepted,correct_fraction_accepted_qualified=correct/accepted if accepted else None,
            accepted_unknown=sum(v['outcome']=='unknown_final' for v in values),
            no_decision_unknown=sum(v['outcome']=='no_decision' and v['final_unknown'] for v in values),
            rows_with_later_geometry_opposite=sum(bool(v['later_geometry_opposite_steps']) for v in values),
            rows_with_later_opposite_qualified=sum(bool(v['later_opposite_qualified_steps']) for v in values),
            rows_with_later_unqualified=sum(bool(v['later_unqualified_steps']) for v in values))
    return result


class Evidence:
    def __init__(self,data_root,check=lambda:None):self.root=Path(data_root);self.items={};self.check=check
    def add(self,item):
        p=Path(item['path']);p=p if p.is_absolute() else self.root/p;p=p.absolute()
        require(not p.is_symlink() and p.is_file(),'missing or symlink evidence: '+str(p))
        normalized=dict(path=str(p),bytes=item['bytes'],sha256=item['sha256'])
        if str(p) in self.items:require(self.items[str(p)]==normalized,'conflicting evidence binding')
        self.check();actual=binding(p);require(actual==normalized,'evidence hash/size differs: '+str(p))
        self.items[str(p)]=actual;return p
    def read(self,path):
        p=Path(path).absolute();require(str(p) in self.items,'unbound input: '+str(p));self.add(self.items[str(p)])
        return json.loads(p.read_text())
    def add_all(self,items):
        for item in items:self.add(item)
    def recheck(self):
        for item in list(self.items.values()):self.add(item)


def close(a,b,message):require(np.isclose(a,b,rtol=1e-12,atol=1e-20),message)


def validate_final(actual,saved):
    require(actual['cycle_gate_passed']==saved['cycle_gate_passed'] and actual['reference_matches']==saved['reference_matches'],
        'saved final classification differs')
    close(actual['maximum_l2_residual'],saved['maximum_l2_residual'],'saved final closure differs')
    require(len(saved['proper_divisors'])==3 and len(saved['reference_distances'])==2,'saved final gate inventory differs')
    for a,b in zip(actual['proper_divisors'],saved['proper_divisors'],strict=True):
        require(a['divisor']==b['divisor'] and a['excluded']==b['excluded'],'saved divisor gate differs')
        close(a['maximum_l2_gap'],b['maximum_l2_gap'],'saved divisor distance differs')
    for a,b in zip(actual['reference_distances'],saved['reference_distances'],strict=True):
        require(a['label']==b['label'],'saved reference order differs');close(a['distance'],b['distance'],'saved reference distance differs')


def validate_saved_strict(observation,prior):
    grid=prior['horizon_diagnostics']['grid'];actual=observation['checkpoints']
    require(len(grid)==len(actual),'saved strict grid incomplete')
    for a,b in zip(actual,grid,strict=True):
        require(a['steps']==b['steps'] and a['strict_label']==b['qualified_label'] and a['reference_matches']==b['reference_matches'],
            'saved strict checkpoint differs')
        close(a['maximum_closure_residual'],b['maximum_closure_residual'],'saved checkpoint closure differs')
    require(observation['strict_decision']==prior['shadow']['decision']==prior['proposed_policy']['decision'],
        'saved strict decision differs')


def load_cohort(name,count,steps,root,evidence):
    receipt_path=Path(root)/'docs/receipts'/f'{name}.json';evidence.add(binding(receipt_path));receipt=evidence.read(receipt_path)
    evidence.add_all(receipt['bindings'])
    audits=[Path(v['path']) if Path(v['path']).is_absolute() else evidence.root/v['path']
        for v in receipt['bindings'] if Path(v['path']).name=='complete-audit.json']
    require(len(audits)==1,'missing or ambiguous complete audit');audit=evidence.read(audits[0])['audit'];base=audits[0].parent
    require(len(audit['rows'])==count,'incomplete planned cohort inventory')
    if steps==600:
        require(receipt['status']=='audit_passed' and receipt['verified_rows']==count
            and audit['status']=='audit_passed' and audit['verified_rows']==count,'unverified primary cohort')
        evidence.add_all(audit['bindings'])
        # Historical audit inventories repeat the same reference record. Evidence.add
        # validates every occurrence; only identical normalized bindings may coalesce.
        refs={str(evidence.add(v)) for v in audit['bindings'] if v['sha256']==REFERENCE_SHA}
        require(len(refs)==1,'missing or ambiguous reference binding');refpath=Path(next(iter(refs)))
    else:
        require(receipt['independent_audit_status']=='complete_audit_passed' and audit['complete_audit_passed']
            and audit['status']=='complete_audit_passed','unverified recovery cohort')
        require(audit['reference_binding']['sha256']==REFERENCE_SHA,'recovery reference differs')
        refpath=evidence.add(audit['reference_binding']);evidence.add_all(audit['input_context']['bindings'])
        evidence.add(audit['final_results']['binding'])
        for row in audit['rows']:evidence.add_all(row['bindings'])
    with np.load(refpath,allow_pickle=False) as saved:refs=saved['states'].copy()
    require(refs.shape==(2,6,896) and refs.dtype==np.float64 and np.isfinite(refs).all(),'invalid reference cycles')
    rows=[]
    for i,prior in enumerate(audit['rows']):
        identity=prior['packet_index'] if steps==600 else prior['trial_id']
        require(identity==i and prior['status']=='verified','misordered or unchecked cohort row')
        if steps==120:require(prior['integrity_passed'] and prior['producer_agreement'],'unchecked recovery row')
        path=base/'run'/(f'trial-{i:02d}-readback-final-600.npz' if steps==600 else f'trial-{i:02d}-final-120.npz')
        require(str(path.absolute()) in evidence.items,'unbound complete trajectory')
        rows.append(dict(row_id=i,prior=prior,path=path))
    return refs,rows


def analyze_row(item,steps,refs,check=lambda:None):
    data=gates.load_trajectory(item['path'],steps,0)
    require(float(data['alpha'])==0.07293697809562831 and float(data['log10_temperature'])==-0.15395151820007266,'trajectory controls differ')
    try:
        observation=observe(data['states'],refs,check)
        final=gates.recompute(data['states'][1:],refs,0);validate_final(final,item['prior']['outcome'])
        if steps==600:validate_saved_strict(observation,item['prior'])
    except BaseException as error:
        error.partial_row=dict(row_id=item['row_id'],status='failed',updates=steps,
            trajectory=binding(item['path']),error=f'{type(error).__name__}: {error}',
            partial_observation=getattr(error,'partial_observation',locals().get('observation')))
        raise
    final_label=qualified(final);final.pop('classification',None);row=dict(row_id=item['row_id'],status='completed',updates=steps,
        stratum=item['prior'].get('stratum',item['prior'].get('kind','line')),
        trajectory=binding(item['path']),final_label=final_label,final_qualification=final,**observation)
    for method in ('geometry','strict'):row[method+'_score']=score_decision(observation[method+'_decision'],final_label,observation['checkpoints'])
    d=row['geometry_decision'];c=next((c for c in row['checkpoints'] if d and c['steps']==d['steps']),None)
    row['at_geometry_decision']=None if c is None else dict(strict_checkpoint_label=c['strict_label'],
        strict_policy_already_decided=row['strict_decision'] is not None and row['strict_decision']['steps']<=d['steps'],
        warning='Geometric label commitment does not certify that the carried state is ready for another write.')
    return row


class Writer:
    def __init__(self,path):self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False)
    def size(self):return sum(p.stat().st_size for p in self.path.iterdir() if p.is_file())
    def json(self,name,data,final=False):
        blob=(json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
        require(self.size()+len(blob)<=(MAX_OUTPUT if final else WORKING_OUTPUT),'output byte cap')
        with (self.path/name).open('xb') as f:f.write(blob)


def source_bindings(root):
    root=Path(root);commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();items=[]
    paths=[Path(__file__),Path(gates.__file__),Path(streaming_readout.__file__),
        root/'docs/causal-geometry-readout.md',root/'tests/test_causal_geometry.py',root/'pyproject.toml',root/'uv.lock']
    paths += [root/'docs/receipts'/f'{name}.json' for name,_,_ in COHORTS]
    for path in paths:
        p=path.resolve();item=binding(p);relative=p.relative_to(root.resolve()).as_posix()
        raw=subprocess.check_output(['git','-C',str(root),'show',commit+':'+relative])
        require(len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],'analysis source is not frozen: '+relative)
        items.append(dict(**item,relative_path=relative))
    return dict(commit=commit,files=items,numpy_version=np.__version__)


def admission():
    available=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:'))
    require(available>=MIN_AVAILABLE,'less than8GiB memory available')
    current=os.getpriority(os.PRIO_PROCESS,0)
    if current<10:os.nice(10-current)
    return dict(utc=utc(),available_bytes=available,nice=os.getpriority(os.PRIO_PROCESS,0),load_average=list(os.getloadavg()))


def run(root,data_root,output,loader=load_cohort,row_analyzer=analyze_row,clock=time.monotonic,
        rss_reader=rss,admit=admission,freeze=source_bindings,install_signal=True):
    writer=Writer(output);started=clock();old=None
    report=dict(status='running',utc=utc(),plan=plan(),cohorts=[dict(name=n,steps=s,status='not_started',
        rows=[dict(row_id=i,status='not_started') for i in range(c)]) for n,c,s in COHORTS],
        additional_model_calls=0,actual_saved_model_calls=0)
    def check():
        if clock()-started>=MAX_SECONDS:raise TimeoutError('120second analysis cap')
        if rss_reader()>=MAX_RSS:raise MemoryError('1GiB RSS cap')
    evidence=Evidence(data_root,check)
    try:
        if install_signal:
            def stop(signum,frame):raise TimeoutError('120second wall-time signal')
            old=signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
        report['admission']=admit();report['source']=freeze(root)
        evidence.add_all(report['source'].get('files',[]));writer.json('design.json',plan())
        writer.json('source.json',report['source'])
        for cohort,(name,count,steps) in zip(report['cohorts'],COHORTS,strict=True):
            cohort['status']='running';refs,items=loader(name,count,steps,root,evidence)
            require(len(items)==count,'loader omitted a planned row')
            for i,item in enumerate(items):
                require(item['row_id']==i,'loader reordered planned rows');cohort['rows'][i]['status']='running'
                try:
                    check();row=row_analyzer(item,steps,refs,check)
                    require(row['row_id']==i and row['status']=='completed','row analyzer identity/completion differs')
                    writer.json(f'{name}-row-{i:02d}.json',row);cohort['rows'][i]=row
                    check()
                except BaseException as error:
                    partial=getattr(error,'partial_row',None)
                    if partial is not None:
                        writer.json(f'{name}-row-{i:02d}.json',partial,final=True);cohort['rows'][i]=partial
                    raise
            cohort['summary']=summarize(cohort['rows']);cohort['status']='completed'
        evidence.recheck();check();report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        for cohort in report['cohorts']:
            if cohort['status']=='running':
                cohort['status']='failed'
                for row in cohort['rows']:
                    if row['status']=='running':row.update(status='failed',error=report['error'])
    finally:
        if install_signal:
            signal.setitimer(signal.ITIMER_REAL,0)
            if old is not None:signal.signal(signal.SIGALRM,old)
        report.update(elapsed_seconds=clock()-started,peak_rss_bytes=rss_reader(),bindings=list(evidence.items.values()))
        # Completed row bodies are saved once; the summary retains exact shard bindings.
        for cohort in report['cohorts']:
            for i,row in enumerate(cohort['rows']):
                path=writer.path/f'{cohort["name"]}-row-{i:02d}.json'
                if path.exists():cohort['rows'][i]=dict(row_id=i,status=row['status'],artifact=binding(path))
        writer.json('result.json',report,final=True)
        writer.json('manifest.json',dict(files=[binding(p) for p in sorted(writer.path.iterdir()) if p.is_file()]),final=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/data2/shrimp-memory-data'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();result=run(Path(__file__).resolve().parents[1],args.data_root,args.output_dir)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
