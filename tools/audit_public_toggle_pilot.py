#!/usr/bin/env python3
"""Independent compact saved-evidence audit for two fixed public map pilots.

Numerical/cache primitives come from an independent auditor, never a producer.
Successful-update probabilities are hash-only in this compact format: their
values are not claimed independently recovered. Every saved state, token,
diagnostic/failure array, cache and final row remains part of the audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile
import traceback

for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np
import audit_public_repair_half_pulse as primitive

ROOT=Path(__file__).resolve().parents[1]
TOGGLE='public-toggle-pilot'
PATH='public-path-comparison'
PROFILES=(TOGGLE,PATH)
H=dict(alpha=0.07293697809562831,log10_temperature=-0.15395151820007266)
E0=dict(alpha=H['alpha'],log10_temperature=-0.1535)
E1=dict(alpha=0.07299785589927693,log10_temperature=H['log10_temperature'])
RAW_CAP=512*2**20
PACKAGE_CAP=ARCHIVE_CAP=2**30
TRANSFER_CAP=4*2**30
P_PINS=dict(cycle_packet='86da50426fbef108d7f779e1bdb0573aa0926f35c765f835548b4b59df924332',
            cycle_states='57c5ba33135673f7935483fe10f2a7bf6c3a28a9696c4a937c1f1fb875569584',
            source_audit='ede01c781a2109c99d4813596750db6c78bd89bfc2fc7a8c51f06aacdf998309')


def require(ok,label):
    if not bool(ok):raise ValueError(label)


def same(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes(order='C')==b.tobytes(order='C')


def strict_json(text):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key '+key);result[key]=value
        return result
    def invalid(value):raise ValueError('nonfinite JSON constant '+value)
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def binding(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'missing/symlink '+str(path))
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(2**20):digest.update(chunk)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest.hexdigest())


def controls(fraction):return {k:H[k]+fraction*(E0[k]-H[k]) for k in H}


def control_bits(ct):
    return {k:dict(value=v,hex=float(v).hex(),bits=int(np.asarray(v,dtype=np.float64).view(np.uint64))) for k,v in ct.items()}


def schedule(profile):
    """Independent literal schedule; no state-dependent control selection."""
    require(profile in PROFILES,'fixed public profile required')
    if profile==TOGGLE:
        specs=[('baseline-hold',[(H,120),(H,240)],[0,1]),
               ('baseline-write0',[(E0,120),(H,240)],[0,0]),
               ('baseline-write1',[(E1,120),(H,240)],[1,1])]
        da=E1['alpha']-H['alpha'];dt=E0['log10_temperature']-H['log10_temperature']
        for sa,st in ((-1,-1),(-1,1),(1,-1),(1,1)):
            name=('m' if sa<0 else 'p')+('m' if st<0 else 'p')
            corner=dict(alpha=H['alpha']+sa*da,log10_temperature=H['log10_temperature']+st*dt)
            for dwell in (12,36):specs.append((f'diagonal-{name}-{dwell}',[(corner,dwell),(H,240)],None))
    else:
        specs=[('direct-short',[(controls(.78125),240),(controls(.75),240)],None),
               ('direct-long',[(controls(.78125),960),(controls(.75),240)],None),
               ('staircase',[(controls(f),240) for f in (.7578125,.765625,.7734375,.78125,.75)],None)]
    groups=[];stages=[];calls=[]
    for gid,(name,legs,expected) in enumerate(specs):
        group=dict(group_id=gid,name=name,stage_ids=[],row_ids=[2*gid,2*gid+1],expected_baseline_pair=expected)
        for lid,(ct,n) in enumerate(legs):
            sid=len(stages);stage_name=('pulse','hold')[lid] if profile==TOGGLE else f'leg-{lid}'
            stage=dict(stage_id=sid,group_id=gid,group=name,local_stage_id=lid,name=stage_name,controls=ct.copy(),steps=n)
            stages.append(stage);group['stage_ids'].append(sid)
            for category,step,cached in [('prefix',0,None),('comparison',0,True),('comparison',0,False)]+[('update',i,True) for i in range(1,n+1)]:
                calls.append(dict(call=len(calls)+1,group_id=gid,group=name,stage_id=sid,stage=stage_name,category=category,step=step,cached=cached))
        groups.append(group)
    require(len(calls)==(3258 if profile==TOGGLE else 2907),'fixed total calls')
    return groups,stages,calls


def classify_pair(bits):
    require(type(bits) is list and len(bits)==2 and all(x is None or type(x) is int and x in (0,1) for x in bits),'ordered neutral pair')
    if None in bits:return 'unresolved'
    return {(0,1):'identity',(1,0):'toggle',(0,0):'reset0',(1,1):'reset1'}[tuple(bits)]


def separation(left,right,threshold=1e-4):
    primitive.valid_matrix(left,(6,896),'left cycle');primitive.valid_matrix(right,(6,896),'right cycle')
    rows=[]
    for shift in range(6):
        values=np.linalg.norm(left-np.roll(right,-shift,axis=0),axis=1)
        require(np.isfinite(values).all(),'finite cyclic distances')
        rows.append(dict(shift=shift,phase_l2_distances=values.tolist(),maximum_l2_distance=float(values.max())))
    best=min(rows,key=lambda r:r['maximum_l2_distance'])
    return dict(passed=best['maximum_l2_distance']>threshold,
        minimum_over_shifts_of_maximum_l2_distance=best['maximum_l2_distance'],best_shift=best['shift'],
        required_separation=threshold,all_shifts=rows)


def metric_match(saved,actual,label):
    if isinstance(actual,dict):
        require(type(saved) is dict and set(saved)==set(actual),label+' fields')
        for key,value in actual.items():metric_match(saved[key],value,label+'.'+key)
    elif isinstance(actual,list):
        require(type(saved) is list and len(saved)==len(actual),label+' length')
        for i,(a,b) in enumerate(zip(saved,actual,strict=True)):metric_match(a,b,f'{label}[{i}]')
    elif type(actual) is float:
        require(type(saved) in (int,float) and math.isfinite(saved) and math.isclose(saved,actual,rel_tol=1e-12,abs_tol=1e-18),label+' numeric value')
    else:require(type(saved) is type(actual) and saved==actual,label+' exact value')


def stage_metrics(states,refs):
    """P6 qualification and pair distance; no stability or branch inference."""
    require(states.ndim==3 and states.shape[1:]==(2,896) and len(states)>=13,'stage tail availability')
    reads=[primitive.read_cycle(states[1:,lane],refs) for lane in (0,1)]
    gap=separation(states[-6:,0],states[-6:,1])
    qualified=all(r['cycle_gate_passed'] for r in reads)
    return dict(readouts=reads,pair_separation=gap,both_primitive_six=qualified,
                qualified_distinct=qualified and gap['passed'],
                qualified_same=qualified and gap['minimum_over_shifts_of_maximum_l2_distance']<=1e-6)


def pair_status(metrics):
    if not metrics['both_primitive_six']:return 'unresolved'
    distance=metrics['pair_separation']['minimum_over_shifts_of_maximum_l2_distance']
    return 'distinct' if distance>1e-4 else 'same_within_tolerance' if distance<=1e-6 else 'intermediate'


def p_references(paths):
    """Verify the three literal source pins without importing the producer."""
    require(set(paths)==set(P_PINS),'three original P input paths')
    records={key:binding(path) for key,path in paths.items()}
    for key,record in records.items():require(record['sha256']==P_PINS[key],'fixed P pin '+key)
    packet=strict_json(Path(paths['cycle_packet']).read_text())
    prior=strict_json(Path(paths['source_audit']).read_text())
    require(packet['controls']==controls(.75) and prior['status']=='passed','qualified original P source')
    arrays=primitive.npz(paths['cycle_states'])
    cycles,terminal=arrays['source_cycle_states'],arrays['source_terminal_states']
    primitive.valid_matrix(cycles,(6,2,896),'P source cycles')
    primitive.valid_matrix(terminal,(2,896),'P terminal pair')
    require(same(terminal,cycles[-1]),'literal original terminal pair')
    return np.concatenate((cycles[-1:],cycles[:-1]),axis=0).transpose(1,0,2).copy(),records


class Evidence:
    def __init__(self,run,profile,commit=None,source_root=ROOT,synthetic=False):
        self.run=Path(run).resolve();self.profile=profile;self.commit=commit
        self.source_root=Path(source_root).resolve();self.synthetic=synthetic
        self.bindings={};self.assertions=0;self.partial=[]
        self.groups,self.stages,self.calls=schedule(profile)

    def check(self,value,label):self.assertions+=1;require(value,label)

    def record(self,path,expected=None):
        value=binding(path)
        if expected is not None:self.check((value['bytes'],value['sha256'])==(expected['bytes'],expected['sha256']),'binding '+str(path))
        old=self.bindings.get(value['path']);self.check(old is None or old==value,'conflicting repeated provenance')
        self.bindings[value['path']]=value;return value

    def bound(self,record,base=None):
        base=self.run if base is None else Path(base)
        name=Path(record['path']);self.check(not name.is_absolute() and '..' not in name.parts,'safe artifact path')
        path=base/name;self.check(path.resolve().is_relative_to(base.resolve()),'artifact path inside root')
        parent=path
        while parent!=base:self.check(not parent.is_symlink(),'symlink artifact component');parent=parent.parent
        self.record(path,record);return path

    def json(self,name,base=None):
        path=(self.run if base is None else Path(base))/name;self.record(path)
        return strict_json(path.read_text())

    def index(self,records):
        result={}
        for row in records:
            name=row['path'];self.check(type(name) is str and name not in result,'unique inventory path')
            self.check(type(row['bytes']) is int and row['bytes']>=0,'integer artifact size')
            result[name]=(row['bytes'],row['sha256'])
        return result

    def inventory(self):
        manifest=self.json('manifest.json');index=self.index(manifest['files'])
        self.check(set(index)=={p.name for p in self.run.iterdir() if p.is_file() and p.name!='manifest.json'},'complete raw manifest')
        for row in manifest['files']:self.bound(row)
        self.check(all(not p.is_symlink() and (p.is_file() or p.name=='return-shards') for p in self.run.iterdir()),'raw directory inventory')
        self.raw_bytes=sum(p.stat().st_size for p in self.run.iterdir() if p.is_file())
        self.check(self.raw_bytes<=RAW_CAP,'raw artifact cap')
        self.result=self.json('result.json');self.design=self.json('design.json');self.events=[strict_json(x) for x in (self.run/'events.jsonl').read_text().splitlines()]
        self.check(self.result['profile']==self.design['profile']==self.profile,'fixed result/design profile')
        self.check(self.result['status'] in ('completed','preflight_passed','instrument_failed','execution_failed','incomplete','failed'),'terminal status')
        self.check(len(self.result['groups'])==len(self.groups) and len(self.result['stages'])==len(self.stages),'fixed group/stage counts')

    # Remaining adapters are kept local to this new auditor; no producer code is imported.

    def fields(self,saved,want,label):
        for key,value in want.items():
            self.check(key in saved and type(saved[key]) is type(value) and saved[key]==value,label+' '+key)

    def provenance(self):
        source=self.json('source.json');self.check(source==self.result['source'],'source summary')
        if self.synthetic:
            self.check(source.get('synthetic') is True or source.get('fixture') is True,'explicit synthetic source')
        else:
            self.check(source['source_commit']==self.commit and len(self.commit)==40,'frozen source commit')
            self.check(not source.get('working_tree_status'),'frozen source is clean')
            tracked=subprocess.check_output(['git','-C',str(self.source_root),'ls-tree','-r','--name-only',self.commit],text=True).splitlines()
            wanted={'run_public_toggle_pilot.py','audit_public_toggle_pilot.py','test_public_toggle_pilot_audit.py',
                    'audit_public_repair_half_pulse.py','run_public_mechanism.py','run_public_control_continuation.py',
                    'run_public_batch_writes.py','run_public_command_memory.py','validate_qwen_map.py','pyproject.toml','uv.lock'}
            wanted|={'run_public_path_comparison.py','public-path-comparison.md'} if self.profile==PATH else {'test_public_toggle_pilot.py','public-toggle-pilot.md'}
            seen=set();self.source_members=[]
            for record in source['files']:
                name=record['path'];self.check(Path(name).name==name and name not in seen,'unique source basename');seen.add(name)
                relative=record.get('repository_path')
                matches=([relative] if relative in tracked and Path(relative).name==name else []) if relative is not None else [p for p in tracked if Path(p).name==name]
                self.check(len(matches)==1,'unambiguous committed source '+name)
                data=subprocess.check_output(['git','-C',str(self.source_root),'show',self.commit+':'+matches[0]])
                self.check((len(data),hashlib.sha256(data).hexdigest())==(record['bytes'],record['sha256']),'committed source bytes '+name)
                self.source_members.append(dict(path=matches[0],bytes=len(data),sha256=record['sha256'],commit=self.commit))
                if name==Path(__file__).name:self.check(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==record['sha256'],'executing frozen checker')
                if name==Path(primitive.__file__).name:self.check(hashlib.sha256(Path(primitive.__file__).read_bytes()).hexdigest()==record['sha256'],'executing frozen numerical helper')
            self.check(wanted<=seen,'complete runtime/checker source inventory')
        self.record(Path(__file__));self.record(Path(primitive.__file__))
        if not (self.run/'qualified-inputs.json').exists():
            self.check(self.result['model_calls']==0 and self.result['status'] not in ('completed','preflight_passed'),'inputs required before work')
            self.refs=None;return
        qualified=self.json('qualified-inputs.json')
        if self.synthetic:
            self.check(qualified.get('synthetic') is True or qualified.get('fixture') is True,'explicit synthetic inputs')
            self.refs=primitive.npz(self.run/'reference-cycles.npz')['states'];self.record(self.run/'reference-cycles.npz')
        elif self.profile==TOGGLE:
            temporary=dict(audited_source_commit=self.commit,external_bindings=[])
            self.refs=primitive.references(self.run,temporary)
            for item in temporary['external_bindings']:self.record(item['path'],item)
            self.check(qualified['references']['references']['sha256']==primitive.REFERENCE_HASH,'original H references pin')
        else:
            self.check(set(qualified['bindings'])==set(P_PINS),'three original P input pins')
            paths={}
            for key,sha in P_PINS.items():
                item=qualified['bindings'][key];self.check(item['sha256']==sha,'fixed P pin '+key)
                paths[key]=primitive.resolve_external(item);self.record(paths[key],item)
            self.refs,_=p_references(paths)
            self.check(qualified['original_history_order']==[0,1],'original P history order')
        primitive.valid_matrix(self.refs,(2,6,896),'reference cycles')
        self.initial=self.refs[:,0].copy()
        self.check(same(np.load(self.run/'reference-starts.npy',allow_pickle=False),self.initial),'literal ordered starts')
        self.record(self.run/'reference-starts.npy')
        if (self.run/'reference-cycles.npz').exists():
            saved=primitive.npz(self.run/'reference-cycles.npz');self.check(set(saved)=={'states'} and same(saved['states'],self.refs),'saved literal references');self.record(self.run/'reference-cycles.npz')
        if self.result['model_calls']:
            model=self.json('model-inputs.json');self.check(set(model)==set(primitive.INPUT_HASHES),'model metadata inventory')
            for key,sha in primitive.INPUT_HASHES.items():self.check(model[key]['sha256']==sha,'pinned model '+key)
            if not self.synthetic:
                ready=self.json('backend-readiness.json')
                self.check(len(ready['norms'])==len(set(ready['norms']))==49 and ready['rotaries']==['model.rotary_emb'],'native64 patches')
                self.check(ready['torch_threads']==ready['torch_interop_threads']==1 and ready['mode']=='native64','native64 one thread')
                self.check(all(ready['thread_environment'].get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'four numerical thread limits')

    def ledger(self):
        result=self.result;slots=result['calls'];self.check(len(slots)==len(self.calls),'complete planned ledger')
        planned=self.json('planned-calls.json');self.check(len(planned)==len(slots),'frozen planned ledger length')
        active=[];stopped=False
        for slot,want,frozen in zip(slots,self.calls,planned,strict=True):
            self.fields(slot,want,'call identity');self.fields(frozen,want,'planned call identity')
            self.check(frozen['status']=='not_started','planned calls contain no outcomes')
            self.check(slot['status'] in ('not_started','started','returned','failed','incomplete'),'call terminal category')
            if slot['status']=='not_started':stopped=True
            else:self.check(not stopped,'no call after an unstarted slot');active.append(slot)
        self.check(type(result['model_calls']) is int and result['model_calls']==len(active),'attempted call count')
        counts={k:sum(s['category']==k for s in active) for k in ('prefix','comparison','update')}
        self.check(result['call_counts']==counts,'call categories')
        starts=[e for e in self.events if e['event']=='model_call_started'];ends=[e for e in self.events if e['event']=='model_call_completed']
        self.check(len(starts)==len(active) and len({e['call'] for e in ends})==len(ends),'unique invocation events')
        for slot,event in zip(active,starts,strict=True):self.fields(event,self.calls[slot['call']-1],'call start event')
        self.check(len(ends)==len(active),'every attempt has a retained completion/failure')
        for event,slot in zip(ends,active,strict=True):self.fields(event,slot,'call completion event')
        if result['status']=='completed':self.check(len(active)==len(self.calls) and all(s['status']=='returned' for s in slots),'completed full schedule')
        elapsed=result['elapsed_seconds'];rss=result['peak_rss_bytes']
        self.check(type(elapsed) in (float,int) and math.isfinite(elapsed) and elapsed>=0 and type(rss) is int and rss>=0,'measured resources')
        self.check((elapsed<3600 and rss<8*2**30) or result['status']=='incomplete','resource breach is not scientific completion')
        frozen_stages=self.json('planned-stages.json');frozen_groups=self.json('planned-groups.json')
        self.check(len(frozen_stages)==len(self.stages) and len(frozen_groups)==len(self.groups),'frozen plan cardinality')
        for frozen,want in zip(frozen_stages,self.stages,strict=True):
            self.fields(frozen,want,'frozen stage');self.check(frozen['status']=='not_started' and frozen['initial'] is None and not frozen['blocks'],'frozen stage contains no work')
        for frozen,want in zip(frozen_groups,self.groups,strict=True):
            self.fields(frozen,{k:want[k] for k in ('group_id','name','stage_ids')},'frozen group')
            self.check(frozen['status']=='not_started','frozen group contains no outcome')
        self.check(len(result['rows'])==2*len(self.groups),'all final rows retained')
        if self.profile==TOGGLE:
            self.check(self.design['schedule']==dict(groups=11,stages=22,terminal_rows=22,batch_size=2,batch_updates=3192,
                row_updates=6384,prefix_calls=22,comparison_calls=44,total_calls=3258,within_group_carries=11,between_group_resets=10),'fixed toggle cost/carry design')
            self.check(self.design['readout']==dict(hold_steps=240,closure_indices=[235,240],previous_indices=[229,234],
                closure_tolerance=1e-8,proper_divisors=[1,2,3],divisor_separation=1e-4,reference_match_tolerance=1e-6,toggle_pair=[1,0]),'fixed toggle readout gates')
            self.check(self.design['limits']==dict(model_calls=3258,internal_seconds=3600,rss_bytes=8*2**30,vm_bytes=16*2**30,
                raw_bytes=RAW_CAP,final_reserve_bytes=32*2**20,threads=1),'fixed toggle resource bounds')
        else:
            self.check(self.design['calls']==2907 and self.design['updates']==2880 and self.design['initializations']==3,'fixed matched path cost')
        self.active=active

    def stage(self,saved,want,initial):
        self.fields(saved,want,'fixed stage');self.check(saved['nominal_controls']==want['controls'],'nominal controls')
        if 'control_bits' in saved:self.check(saved['control_bits']==control_bits(want['controls']),'exact binary64 control identity')
        sid=want['stage_id'];status=saved['status']
        self.check(status in ('not_started','completed','failed','incomplete','instrument_failed','execution_failed'),'stage status')
        slots=[s for s in self.active if s['stage_id']==sid]
        if saved['initial'] is None:
            self.check(not slots and saved['completed_steps']==0 and not saved['blocks'],'unrun stage contains no work')
            return None,dict(stage_id=sid,status=status,completed_updates=0,readouts=None)
        self.check(initial is not None and saved['initial']['path']==f'stage-{sid:02d}-initial.npy','stage literal input')
        actual=primitive.batch_state(self.run,saved['initial']);self.check(same(actual,initial),'exact carry/reset')
        blocks=[];n=0
        for item in saved['blocks']:
            block=primitive.npz(self.bound(item));n+=len(block['states'])
            self.check(item['path']==f'stage-{sid:02d}-block-{n:03d}.npz','contiguous block filename');blocks.append(block)
        arrays=primitive.reconstruct(actual,blocks,want['controls'])
        self.check(arrays['updates']==n==saved['completed_steps']<=want['steps'],'saved update count')
        self.check(n<=saved['returned_steps']<=saved['attempted_steps']<=min(want['steps'],n+1),'attempted/returned/carried counts')
        if saved['trajectory'] is not None:
            self.check(saved['trajectory']['path']==f'stage-{sid:02d}-trajectory.json','trajectory filename')
            trajectory=strict_json(self.bound(saved['trajectory']).read_text())
            self.check(trajectory==dict(stage_id=sid,initial=saved['initial'],blocks=saved['blocks'],completed_steps=n,
                state_shape=[n+1,2,896],status=status,controls=want['controls'],format='initial plus nonoverlapping post-update blocks'),'trajectory manifest')
        else:self.check(not slots and n==0,'executed stage trajectory required')
        cache,_,_=primitive.audit_cache_stage(self.run,saved,slots,actual,37)
        for slot in slots:primitive.inspect_call(self.run,slot,arrays,cache,37)
        if any(s['category']=='update' for s in slots):self.check(cache.get('diagnostic',{}).get('passed') is True,'no update after cache/full diagnostic failure')
        if status=='completed':self.check(n==want['steps']==saved['attempted_steps']==saved['returned_steps'],'completed stage length')
        metrics=stage_metrics(arrays['states'],self.refs) if n>=12 else None
        if self.profile==PATH and status=='completed':
            observation=self.json(f'stage-{sid:02d}-observation.json')
            self.fields(observation,dict(stage_id=sid,controls=want['controls'],off_P=want['controls']!=controls(.75)), 'path observation')
            fraction=[.78125,.75,.78125,.75,.7578125,.765625,.7734375,.78125,.75][sid]
            self.check(observation['path_fraction']==fraction,'fixed path fraction')
            for saved_read,read in zip(observation['readouts'],metrics['readouts'],strict=True):primitive.neutral_match(saved_read,read)
            gap=metrics['pair_separation'];metric_match(observation['pair_distance'],gap['minimum_over_shifts_of_maximum_l2_distance'],'path pair distance')
            metric_match(observation['pair_distances_by_shift'],[r['maximum_l2_distance'] for r in gap['all_shifts']],'all path cyclic shifts')
            self.check(observation['both_primitive_period_six']==metrics['both_primitive_six'] and observation['pair_status']==pair_status(metrics),'path finite classification')
        return arrays,dict(stage_id=sid,group_id=want['group_id'],status=status,controls=want['controls'],completed_updates=n,
                          update_seconds=float(arrays['step_seconds'].sum()),cache=cache,metrics=metrics)

    def numerical(self):
        if self.refs is None:
            self.stage_reports=[];self.row_reports=[];self.group_reports=[]
            for gid,group in enumerate(self.result['groups']):
                self.check(group['closed'] is True and group['status']=='not_run' and group['model_calls']==0,'missing-input groups explicitly unrun')
                self.check(self.json(f'group-{gid:02d}.json')==group,'unrun marker binding')
                self.group_reports.append(dict(group_id=gid,name=group['name'],status='not_run',mapping='unavailable',decoded_pair=[None,None],model_calls=0))
            for rid,row in enumerate(self.result['rows']):
                self.check(row['status']=='not_run' and row['decoded_bit'] is None and row['readout'] is None,'missing-input row unavailable')
                self.check(self.json(f'row-{rid:02d}-completed.json')==row,'unrun row binding');self.row_reports.append(row)
            return
        self.stage_reports=[];self.row_reports=[];self.group_reports=[];self.array_stages={}
        starts=[e for e in self.events if e['event']=='group_started']
        started_groups=[]
        for want,saved in zip(self.groups,self.result['groups'],strict=True):
            self.fields(saved,{k:want[k] for k in ('group_id','name','stage_ids')},'fixed group')
            gid=want['group_id'];previous=self.initial;stage_outputs=[]
            for sid in want['stage_ids']:
                arrays,report=self.stage(self.result['stages'][sid],self.stages[sid],previous)
                self.stage_reports.append(report);stage_outputs.append(arrays);self.array_stages[sid]=arrays
                previous=None if arrays is None else arrays['states'][-1]
            attempted=[s for s in self.active if s['group_id']==gid]
            if saved.get('initial_state_sha256') is not None:
                self.check(saved['initial_state_sha256']==primitive.array_digest(self.initial),'literal reset group fingerprint');started_groups.append(gid)
            completed=saved['status']=='completed';last=stage_outputs[-1]
            if completed:
                self.check(all(self.result['stages'][sid]['status']=='completed' for sid in want['stage_ids']),'completed group stages')
                self.check(saved['terminal_state_sha256']==primitive.array_digest(last['states'][-1]),'group terminal fingerprint')
            reads=[]
            for lane in (0,1):
                rid=2*gid+lane;row=self.result['rows'][rid]
                self.fields(row,dict(row_id=rid,group_id=gid,lane=lane,source_reference_index=lane),'ordered original-history row')
                self.check(row['source_reference_phase']==0,'declared reference phase at start')
                if self.profile==PATH:
                    self.check(row['original_history_index']==lane and row['reference_label_scope']=='Original P-history indices; not H memory bits. decoded_bit is a legacy numeric-index field.','P history identity scope')
                if completed:
                    self.check(row['status']=='completed','completed final row')
                    recomputed=primitive.read_cycle(last['states'][1:,lane],self.refs)
                    primitive.neutral_match(row['readout'],recomputed)
                    self.check('all_reference_alignments' in row['readout'],'all six reference alignments retained')
                    # This inherited alignment record carries a descriptive
                    # separation flag at 1e-3; neutral matching still uses 1e-6.
                    metric_match(row['readout']['all_reference_alignments'],[separation(last['states'][-6:,lane],ref,1e-3) for ref in self.refs],'all reference alignments')
                    self.check(row['decoded_bit']==recomputed['decoded_bit'],'neutral decoded row')
                    if self.profile==PATH:self.check(row['decoded_reference_index']==recomputed['decoded_bit'],'final P history identity')
                    reads.append(recomputed['decoded_bit'])
                    self.row_reports.append(dict(row_id=rid,group_id=gid,lane=lane,source_reference_index=lane,status='completed',decoded_bit=recomputed['decoded_bit'],readout=recomputed))
                else:
                    self.check(row['status']!='completed' and row['decoded_bit'] is None and row['readout'] is None,'partial/unrun row not a scientific null')
                    reads.append(None);self.row_reports.append(dict(row_id=rid,group_id=gid,lane=lane,source_reference_index=lane,status=row['status'],decoded_bit=None,readout=None))
                self.check(self.json(f'row-{rid:02d}-completed.json')==row,'immutable row snapshot')
                if self.profile==PATH:
                    self.row_reports[-1].update(original_history_index=lane,decoded_reference_index=self.row_reports[-1]['decoded_bit'],reference_label_scope=row['reference_label_scope'])
            mapping=classify_pair(reads) if completed else 'unavailable'
            if self.profile==TOGGLE:self.check(saved['mapping']==mapping and saved['decoded_pair']==reads,'simultaneous two-lane mapping')
            else:
                self.check(saved['mapping']=='not_applicable','P histories are not toggle bits')
                if completed:self.check(saved['decoded_pair']==reads,'final original P-history pair')
                observations=[self.json(f'stage-{sid:02d}-observation.json') for sid in want['stage_ids'] if self.result['stages'][sid]['status']=='completed']
                self.check(saved['observations']==observations,'closed group observation copies')
            marker=self.json(f'group-{gid:02d}.json');self.check(marker==saved and marker['closed'] is True,'immutable closed marker')
            self.check(marker['model_calls']==len(attempted),'closed marker attempted calls')
            summary=self.json(f'group-{gid:02d}-summary.json')
            self.check(summary['stages']==[self.result['stages'][sid] for sid in want['stage_ids']] and summary['calls']==[s for s in self.result['calls'] if s['group_id']==gid],'closed stage/call summary')
            self.check(summary['rows']==[self.result['rows'][2*gid+i] for i in (0,1)],'closed row summary')
            expected={f'row-{2*gid+i:02d}-completed.json' for i in (0,1)}|{f'group-{gid:02d}-summary.json'}
            for sid in want['stage_ids']:
                stage=self.result['stages'][sid]
                for key in ('initial','trajectory','scalar_prefix_cache','expanded_prefix_cache'):
                    if stage.get(key):expected.add(stage[key]['path'])
                expected|={x['path'] for key in ('blocks','failed_outputs') for x in stage[key]}
                if self.profile==PATH and stage['status']=='completed':expected.add(f'stage-{sid:02d}-observation.json')
            expected|={s['artifact']['path'] for s in attempted if s.get('artifact')}
            index=self.index(marker['artifacts']);self.check(set(index)==expected,'exact disjoint closed artifact inventory')
            for record in marker['artifacts']:self.bound(record)
            self.group_reports.append(dict(group_id=gid,name=want['name'],status=saved['status'],decoded_pair=reads,mapping=mapping if self.profile==TOGGLE else 'not_applicable',model_calls=len(attempted)))
        self.check([e['group_id'] for e in starts]==started_groups and self.result['initializations']==len(started_groups),'literal reset event count')
        for event in starts:self.check(event['initial_sha256']==primitive.array_digest(self.initial),'literal reset event bytes')
        if self.profile==TOGGLE and self.result['status']=='completed':
            baselines=all(self.group_reports[i]['decoded_pair']==self.groups[i]['expected_baseline_pair'] for i in range(3))
            toggles=[g['group_id'] for g in self.group_reports[3:] if g['mapping']=='toggle']
            outcome='baseline_unqualified' if not baselines else 'toggle_observed' if toggles else 'toggle_unresolved' if any(g['mapping']=='unresolved' for g in self.group_reports[3:]) else 'no_toggle_observed'
            self.check(self.result['baseline_qualified']==baselines and self.result['toggle_candidate_group_ids']==toggles and self.result['outcome']==outcome,'baseline and probe interpretation remain separate')

    def archive(self,path,record,expected):
        bound=self.record(path,record);self.check(bound['bytes']<=ARCHIVE_CAP,'archive cap')
        seen=set()
        with tarfile.open(path,'r:*') as stream:
            for member in stream:
                self.check(member.isfile() and member.name in expected and member.name not in seen,'safe unique archive member')
                self.check(member.size==expected[member.name][0],'archive member size')
                h=hashlib.sha256();payload=stream.extractfile(member)
                while block:=payload.read(2**20):h.update(block)
                self.check(h.hexdigest()==expected[member.name][1],'archive member hash');seen.add(member.name)
        self.check(seen==set(expected),'complete archive coverage')

    def paired_closure(self,own_manifest,own_entries,completion):
        """Check both return proofs; never inspect the other job's scientific rows."""
        job=self.run.parent
        pair_binding=self.json('pair-context.json',job)
        pair_path=Path(pair_binding['path'])
        self.check(pair_path.is_absolute() and pair_path.resolve()==pair_path,'canonical pair manifest path')
        self.record(pair_path,pair_binding);pair=strict_json(pair_path.read_text())
        self.check(pair['profile']=='public-core-pair' and len(pair['jobs'])==2,'fixed shared pair')
        self.check([x['profile'] for x in pair['jobs']]==list(PROFILES),'fixed two public profiles')
        for key in ('name','local','remote','checkout','slot'):
            self.check(len({x[key] for x in pair['jobs']})==2,'distinct pair '+key)
        self.check(all(x['head']==self.commit for x in pair['jobs']),'paired frozen source')
        own=[x for x in pair['jobs'] if Path(x['local'])==job and x['profile']==self.profile]
        self.check(len(own)==1,'own pair identity')
        maintenance=self.json('maintenance-closeout.json',job)
        self.fields(maintenance,dict(profile='public-core-pair',status='cleared',active=False,
            validation_id=pair['pair_id'],pair_manifest=pair_binding,all_human_inputs_untouched=True),'shared maintenance closure')
        self.check(len(maintenance['jobs'])==2,'both pair proofs retained')
        mapping={};required={'local-return-verification.json','transfer-closeout.json','final-manifest.json',
            'terminal-process-check.json','finished-utc.txt','exit-status.txt','transfer-account.json'}
        for member,proof in zip(pair['jobs'],maintenance['jobs'],strict=True):
            local=Path(member['local'])
            self.check(local.is_absolute() and local.resolve()==local and local.name==member['name'],'canonical member identity')
            self.check(self.json('pair-context.json',local)==pair_binding,'identical pair contexts')
            self.fields(proof,dict(id=member['name'],profile=member['profile'],local=str(local),
                no_owned_numerical_processes=True,transfer_reservation_closed=True),'pair proof identity')
            proof_bindings={}
            for record in proof['bindings']:
                p=Path(record['path'])
                self.check(p.parent==local and p.name in required and p.name not in proof_bindings,'exact closure binding inventory')
                proof_bindings[p.name]=self.record(p,record)
            self.check(set(proof_bindings)==required,'all seven closure proofs')
            manifest=strict_json((local/'final-manifest.json').read_text());inventory=self.index(manifest['files'])
            digest=proof_bindings['final-manifest.json']['sha256'];mapping[member['name']]=digest
            self.check(proof['manifest_sha256']==digest and proof['files']==len(inventory),'pair manifest identity/count')
            self.check(sum(size for size,_ in inventory.values())<=PACKAGE_CAP and manifest['archive']['bytes']<=ARCHIVE_CAP,'pair package/archive cap')
            for name in ('terminal-process-check.json','finished-utc.txt','exit-status.txt'):
                self.check(inventory.get(name)==(proof_bindings[name]['bytes'],proof_bindings[name]['sha256']),'terminal evidence is returned inventory')
            terminal=strict_json((local/'terminal-process-check.json').read_text())
            self.check(terminal['no_owned_numerical_processes'] is True and terminal['remaining_numerical_processes']==[],'both owned numerical processes ended')
            code=int((local/'exit-status.txt').read_text())
            self.check(type(proof['exit_status']) is int and proof['exit_status']==code,'pair exit status')
            verified=strict_json((local/'local-return-verification.json').read_text())
            closed=strict_json((local/'transfer-closeout.json').read_text())
            self.check(verified['status']=='verified' and verified['manifest_sha256']==digest and verified['files']==len(inventory) and verified['archive']==manifest['archive'],'both inventories verified')
            self.fields(closed,dict(id=member['name'],state='complete',local_return_verified=True,
                reserved_bytes=TRANSFER_CAP,manifest_sha256=digest,verified_manifest_entries=len(inventory)),'both reservations closed')
            account=strict_json((local/'transfer-account.json').read_text())
            charges=[x['bytes'] for x in account['attempts']]
            self.check(account['reserved_bytes']==TRANSFER_CAP and all(type(x) is int and x>=0 for x in charges),'paired cumulative transfer charges')
            overhead=account['overhead_reserve']
            self.check(type(overhead) is int and overhead>=0 and sum(charges)+overhead<=TRANSFER_CAP and proof['charged_attempt_bytes']==sum(charges),'paired transfer reserves respected')
            if member==own[0]:
                self.check(digest==own_manifest['sha256'] and len(inventory)==own_entries,'own return in shared gate')
                expected='returned_pending_scientific_audit' if code==0 else 'failed_run_evidence_returned'
                self.fields(completion,dict(status=expected,exit_status=code,files=own_entries,transfer_reservation_closed=True),'completion after shared release')
                self.check((code==0)==(self.result['status'] in ('completed','preflight_passed')),'exit agrees with scientific/execution status')
        gate=maintenance['gate']
        self.check(gate['active'] is False and gate['validation_id']==pair['pair_id'] and gate['return_manifest_sha256_by_job']==mapping,'shared gate binds both manifests')
        close=completion['maintenance']
        self.fields(close,dict(status='cleared',validation_id=pair['pair_id'],pair_manifest=pair_binding),'completion shared gate')
        shared=Path(close['pair_closeout']['path']);self.record(shared,close['pair_closeout'])
        self.check(strict_json(shared.read_text())==maintenance,'shared and job closeout identity')
        self.pair_return=dict(pair_id=pair['pair_id'],pair_manifest=pair_binding,manifest_sha256_by_job=mapping,
            both_reservations_closed=True,both_owned_processes_absent=True,maintenance_released=True)

    def transport(self):
        job=self.run.parent
        manifest=self.json('final-manifest.json',job);mb=self.record(job/'final-manifest.json')
        self.check(mb['bytes']<=4*2**20,'final manifest cap')
        index=self.index(manifest['files']);self.check(sum(v[0] for v in index.values())<=PACKAGE_CAP,'complete package cap')
        for item in manifest['files']:self.bound(item,job)
        self.check({'run/'+p.name for p in self.run.iterdir() if p.is_file()}<=set(index),'all numerical evidence returned')
        transferred=self.index(manifest['transferred_files']);self.check(all(index.get(k)==v for k,v in transferred.items()),'transferred subset')
        self.check(type(manifest['already_local_files']) is int and manifest['already_local_files']==len(set(index)-set(transferred)),'verified local complement')
        self.archive(self.bound(manifest['archive'],job),manifest['archive'],transferred)
        returned_bytes=mb['bytes']+manifest['archive']['bytes']
        for gid,group in enumerate(self.result['groups']):
            folder=self.run/'return-shards'/f'group-{gid:02d}'
            if not folder.exists():continue
            shard=self.json('manifest.json',folder);sb=self.record(folder/'manifest.json')
            self.check(sb['bytes']<=2**20 and shard['group_id']==gid,'incremental manifest identity/cap')
            expected=self.index(group['artifacts']);marker=self.record(self.run/f'group-{gid:02d}.json')
            expected[f'group-{gid:02d}.json']=(marker['bytes'],marker['sha256'])
            self.check(self.index(shard['files'])==expected,'immutable shard inventory')
            returned_bytes+=sb['bytes']
            if (folder/'verified.json').exists():
                self.archive(folder/'archive.tar',shard['archive'],expected);returned_bytes+=shard['archive']['bytes']
                verification=self.json('verified.json',folder)
                self.check(verification['manifest_sha256']==sb['sha256'] and verification['files']==len(expected),'verified shard binding')
            else:
                raw=self.record(folder/'archive.tar') if (folder/'archive.tar').exists() else None
                if raw:returned_bytes+=raw['bytes']
                self.partial.append(dict(group_id=gid,status='unverified_shard_retained',manifest=sb,archive=raw))
        verified=self.json('local-return-verification.json',job);closed=self.json('transfer-closeout.json',job)
        completion=self.json('completion.json',job);account=self.json('transfer-account.json',job)
        self.check(verified['status']=='verified' and verified['files']==len(index) and verified['manifest_sha256']==mb['sha256'] and verified['archive']==manifest['archive'],'complete local verification')
        self.check(closed['state']=='complete' and closed['local_return_verified'] is True and closed['verified_manifest_entries']==len(index) and closed['manifest_sha256']==mb['sha256'],'closed task transfer reservation')
        self.check(completion['transfer_reservation_closed'] is True,'completion transfer release')
        self.check(account['reserved_bytes']==TRANSFER_CAP and all(type(x['bytes']) is int and x['bytes']>=0 for x in account['attempts']),'fixed cumulative transfer account')
        charged=sum(x['bytes'] for x in account['attempts'])
        self.check(charged>=returned_bytes and type(account['overhead_reserve']) is int and account['overhead_reserve']>=0 and charged+account['overhead_reserve']<=TRANSFER_CAP,'all returned attempts charged within reserve')
        if not self.synthetic:
            self.check((job/'source-commit.txt').read_text().strip()==self.commit,'returned frozen source commit')
            self.record(job/'source-commit.txt')
            self.paired_closure(mb,len(index),completion)
        self.return_entries=len(index);self.return_status='synthetic_transport_verified' if self.synthetic else 'verified'

    def execute(self,with_transport=True):
        self.inventory();self.provenance();self.ledger();self.numerical()
        self.return_entries=0;self.return_status='synthetic_not_applicable'
        self.check(with_transport or self.synthetic,'production cannot bypass closed return')
        if with_transport:self.transport()
        # Deduplicate only identical normalized path/bytes/hash evidence.
        report=dict(status='audit_passed',integrity_status='passed',profile=self.profile,synthetic=self.synthetic,
            audited_source_commit=self.commit,scientific_status=self.result['status'],outcome=self.result.get('outcome'),
            model_calls=self.result['model_calls'],auditor_model_calls=0,network_calls=0,assertion_count=self.assertions,
            elapsed_seconds=self.result['elapsed_seconds'],peak_rss_bytes=self.result['peak_rss_bytes'],actual_raw_bytes=self.raw_bytes,
            groups=self.group_reports,rows=self.row_reports,stages=self.stage_reports,
            baseline_qualified=self.result.get('baseline_qualified'),toggle_candidate_group_ids=self.result.get('toggle_candidate_group_ids'),
            return_status=self.return_status,returned_manifest_entries=self.return_entries,
            transfer_reservation_closed=with_transport and not self.synthetic,partial_evidence=self.partial,
            bindings=list(self.bindings.values()),verified_source_members=getattr(self,'source_members',[]),
            auditor=binding(__file__),numeric_helper=binding(primitive.__file__),
            paired_return=getattr(self,'pair_return',None),
            scope='Fixed public finite-state pilot. All saved states/tokens/times, cache/diagnostic/failure arrays and final rows checked. Successful-update probability values were not retained and cannot be reconstructed from their hashes. No general toggle reliability, other-phase, minimum-duration, stability, branch-identity, causal-hysteresis or language claim.')
        if self.profile==PATH:
            report['path_comparisons']=[dict(group_id=g['group_id'],arm=g['name'],
                approach_stage_id=g['stage_ids'][-2],return_stage_id=g['stage_ids'][-1],
                final_original_history_pair=self.group_reports[g['group_id']]['decoded_pair']) for g in self.groups]
        return report


def audit(run,profile,commit=None,source_root=ROOT,*,synthetic=False,with_transport=True):
    return Evidence(run,profile,commit,source_root,synthetic).execute(with_transport)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--profile',choices=PROFILES,required=True)
    parser.add_argument('--source-commit',required=True);parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv);args.output_dir.mkdir(exist_ok=False)
    try:report=audit(args.run_dir,args.profile,args.source_commit,args.source_root);code=0
    except Exception as error:
        report=dict(status='audit_failed',integrity_status='failed',profile=args.profile,synthetic=False,
            error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),auditor_model_calls=0,network_calls=0);code=1
    report['utc']=subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
    report['auditor']=binding(__file__);report['numeric_helper']=binding(primitive.__file__)
    (args.output_dir/'independent-review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','profile','error') if k in report},indent=2))
    return code


if __name__=='__main__':raise SystemExit(main())

