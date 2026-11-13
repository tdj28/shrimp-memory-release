#!/usr/bin/env python3
"""Independent saved-data checker for public loops and stateful gate chains.

The numerical/cache routines are from an independent auditor, never the producer.
No model import, generation or inference is performed.
"""
import argparse,ast,json,subprocess,traceback
from collections import Counter
from pathlib import Path
import numpy as np
import audit_public_repair_half_pulse as a

ROOT=Path(__file__).resolve().parents[1]
HOLD=a.HOLD
ENDPOINTS=a.ENDPOINTS
SOURCES=['tools/'+n for n in ('run_public_repair_handoff.py','run_public_control_continuation.py','run_public_control_tolerance.py','run_public_batch_writes.py','run_public_command_memory.py','transparent_command_contract.py','check_p6_writes.py','check_p6_recovery.py','check_p6_hold.py','validate_qwen_map.py','benchmark_qwen_batch.py','run_public_half_pulse.py','validate_qwen_batch_trajectory.py','run_public_mechanism.py')]+['docs/hysteresis-and-logic-protocol.md','tests/test_public_mechanism.py','pyproject.toml','uv.lock']
R=a.require


def schedule(mode,dwell,middle_fraction=.5):
    R(mode in ('and','or','hysteresis') and dwell in (240,480),'fixed mode/dwell')
    R(mode=='hysteresis' or dwell==240,'gate dwell')
    R(middle_fraction in (.25,.5,.75) and (mode=='hysteresis' or middle_fraction==.5),'loop fraction')
    groups=[];stages=[];calls=[]
    cases=[{'direction':b} for b in (0,1)] if mode=='hysteresis' else [{'b':b,'c':c} for b in (0,1) for c in (0,1)]
    for gid,case in enumerate(cases):
        name=f'loop-{case["direction"]}' if mode=='hysteresis' else f'{mode}-b{case["b"]}-c{case["c"]}'
        groups.append(dict(group_id=gid,name=name,case=case,stage_ids=list(range(4*gid,4*gid+4))))
        if mode=='hysteresis':
            e=ENDPOINTS[case['direction']];controls=[{k:HOLD[k]+f*(e[k]-HOLD[k]) for k in HOLD} for f in (middle_fraction,1.,middle_fraction,0.)];steps=[dwell]*4
        else:
            # A and decoded state never participate in this control selection.
            def c(bit):return (ENDPOINTS[0] if bit==0 else HOLD) if mode=='and' else (HOLD if bit==0 else ENDPOINTS[1])
            controls=[c(case['b']),HOLD,c(case['c']),HOLD];steps=[120,240,120,240]
        for local,(ct,n) in enumerate(zip(controls,steps,strict=True)):
            sid=len(stages);stage=dict(stage_id=sid,group_id=gid,group=name,local_stage_id=local,name=f'leg-{local}',controls=ct,steps=n)
            stages.append(stage)
            for cat,step,cached in [('prefix',0,None),('comparison',0,True),('comparison',0,False)]+[('update',i,True) for i in range(1,n+1)]:
                calls.append(dict(call=len(calls)+1,group_id=gid,group=name,stage_id=sid,stage=stage['name'],category=cat,step=step,cached=cached))
    return groups,stages,calls


def recurrence(states):
    """Finite tail recurrence; no stability/chaos inference from a failed gate."""
    R(states.ndim==2 and states.dtype==np.float64 and np.isfinite(states).all(),'finite tail')
    rows=[];qualified=None
    for p in range(1,min(64,len(states)//2)+1):
        tail=states[-p:];old=states[-2*p:-p];res=float(np.linalg.norm(tail-old,axis=1).max())
        divisors=[dict(period=d,maximum_gap=float(np.linalg.norm(tail-np.roll(tail,d,axis=0),axis=1).max())) for d in range(1,p) if p%d==0]
        passed=res<=1e-8 and all(x['maximum_gap']>1e-4 for x in divisors)
        rows.append(dict(period=p,maximum_phase_residual=res,proper_divisors=divisors,passed=passed))
        if passed and qualified is None:qualified=p
    return dict(qualified_primitive_period=qualified,tests=rows)


def cycle_gap(x,y,p):
    R(type(p) is int and p>0 and len(x)>=p and len(y)>=p,'cycle gap inputs')
    return min(float(np.linalg.norm(x[-p:]-np.roll(y[-p:],s,axis=0),axis=1).max()) for s in range(p))


def verify_return(base,report,commit):
    outer=base.parent;manifest=json.loads((outer/'final-manifest.json').read_text());names=set()
    for r in manifest['files']:
        R(r['path'] not in names,'duplicate final member');names.add(r['path']);report['bindings'].append(a.binding(a.checked(outer,r)))
    report['bindings'] += [a.binding(outer/'final-manifest.json'),a.binding(a.checked(outer,manifest['archive']))]
    R(manifest['archive']['bytes']<=768*1024**2,'archive cap')
    v=json.loads((outer/'local-return-verification.json').read_text());c=json.loads((outer/'transfer-closeout.json').read_text());done=json.loads((outer/'completion.json').read_text())
    R(v['status']=='verified' and v['manifest_sha256']==a.binding(outer/'final-manifest.json')['sha256'] and v['archive']==manifest['archive'],'return hash verification')
    R(c['state']=='complete' and c['local_return_verified'] is True and c['manifest_sha256']==v['manifest_sha256'] and c['verified_manifest_entries']==len(names),'closed reservation')
    R(done['transfer_reservation_closed'] is True and done['files']==len(names),'completion return')
    R((outer/'source-commit.txt').read_text().strip()==commit,'returned source')
    report['verified_return_inventory_files']=len(names)


def source(base,result,commit,mode,dwell):
    src=json.loads((base/'source.json').read_text());R(src==result['source'] and src['source_commit']==commit,'source identity')
    R(src['arguments']['mode']==mode and int(src['arguments']['dwell'])==dwell,'source mode/dwell')
    R(len(src['files'])==len(SOURCES),'source inventory')
    for rec,path in zip(src['files'],SOURCES,strict=True):
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+path]);R(rec['path']==Path(path).name and len(raw)==rec['bytes'] and a.sha(raw)==rec['sha256'],'frozen Git source '+path)
    model=json.loads((base/'model-inputs.json').read_text());R(set(model)==set(a.INPUT_HASHES),'model provenance inventory')
    for k,v in a.INPUT_HASHES.items():R(model[k]['sha256']==v,'model provenance '+k)
    tree=ast.parse(subprocess.check_output(['git','-C',str(ROOT),'show',commit+':tools/validate_qwen_map.py'],text=True))
    prefix=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREFIX_IDS' for t in n.targets))
    return len(prefix)


def audit(base,commit,mode,dwell,report):
    fraction=report.get('middle_fraction',.5)
    groups,expected,calls=schedule(mode,dwell,fraction);verify_return(base,report,commit)
    manifest=json.loads((base/'manifest.json').read_text());names=set()
    for rec in manifest['files']:
        R(rec['path'] not in names,'duplicate producer member');names.add(rec['path']);report['bindings'].append(a.binding(a.checked(base,rec)))
    R(names=={p.name for p in base.iterdir() if p.is_file() and p.name!='manifest.json'},'complete raw inventory')
    report['bindings'].append(a.binding(base/'manifest.json'))
    result=json.loads((base/'result.json').read_text());R(result['mode']==mode and result['dwell']==dwell and result['profile']=='public-mechanism-'+mode,'profile')
    report['producer_status']=result['status'];report['retained_error']=result.get('error')
    # Incomplete attempts are preserved and explicitly left uninterpreted.
    R(result['status']=='completed','incomplete producer retained; no complete-gate claim')
    R(len(result['groups'])==len(groups) and len(result['stages'])==len(expected) and len(result['calls'])==len(calls),'full planned inventory')
    design=json.loads((base/'design.json').read_text());R(design['calls']==len(calls) and design['updates']==sum(x['steps'] for x in expected),'design counts')
    R(design.get('middle_fraction',.5)==fraction and float(result['source']['arguments'].get('middle_fraction',.5))==fraction,'frozen loop fraction')
    R(design['limits']==dict(seconds=3600,rss=8*1024**3,raw=512*1024**2) and design['reference_phases']==[0,0] and design['batch_initial_bits']==[0,1],'design bounds/initial states')
    R(result['model_calls']==len(calls) and result['call_counts']==dict(Counter(x['category'] for x in calls)),'actual call counts')
    R(0<result['elapsed_seconds']<=3600 and result['peak_rss_bytes']<=8*1024**3,'resource bounds')
    for filename,items,identity in [('planned-stages.json',expected,'stage'),('planned-groups.json',groups,'group')]:
        frozen=json.loads((base/filename).read_text());R(len(frozen)==len(items),'frozen '+identity+' inventory')
        for saved,want in zip(frozen,items,strict=True):
            R(all(saved[k]==v for k,v in want.items()) and saved['status']=='not_started','frozen '+identity+' identity')
            if identity=='stage':R(saved['initial'] is None and not saved['blocks'] and saved['completed_steps']==0,'frozen stage contains outcomes')
            else:R(saved['initial_state_sha256'] is None and not saved['observations'],'frozen group contains outcomes')
    R(sum(p.stat().st_size for p in base.iterdir() if p.is_file())<=512*1024**2,'raw byte cap')
    prefix=source(base,result,commit,mode,dwell);refs=a.references(base,report)
    R(refs is not None,'reference evidence');initial=np.stack([refs[0,0],refs[1,0]])
    R(a.same_bytes(np.load(base/'reference-starts.npy',allow_pickle=False),initial),'reference start bytes')
    events=[json.loads(x) for x in (base/'events.jsonl').read_text().splitlines()]
    frozen=[e for e in events if e['event']=='inputs_frozen'];groupstarts=[e for e in events if e['event']=='group_started']
    R(len(frozen)==1 and frozen[0]['initial_sha256']==a.array_digest(initial),'frozen initial event')
    R(len(groupstarts)==result['initializations']==len(groups),'group reset count')
    for gid,e in enumerate(groupstarts):R(e['group_id']==gid and e['initial_sha256']==a.array_digest(initial),'group reset event')
    starts=[x for x in events if x['event']=='model_call_started'];ends=[x for x in events if x['event']=='model_call_completed']
    R(len(starts)==len(ends)==len(calls),'event coverage');R(len({x['call'] for x in ends})==len(calls),'unique completed calls')
    for slot,want,event,end in zip(result['calls'],calls,starts,ends,strict=True):
        R(all(type(slot[k]) is type(v) and slot[k]==v and event[k]==v for k,v in want.items()),'call identity')
        R(slot['status']=='returned' and all(end.get(k)==v for k,v in slot.items()),'completed event/slot')
    R(json.loads((base/'planned-calls.json').read_text())==[dict(**c,status='not_started') for c in calls],'planned calls no outcomes')
    arrays={};stage_reports=[]
    for sid,(s,want) in enumerate(zip(result['stages'],expected,strict=True)):
        R(all(s[k]==v for k,v in want.items()) and s['status']=='completed','fixed stage identity')
        R(s['nominal_controls']==want['controls'] and s['historical_comparison'] is None and s['historical_comparison_scope']=='not_applicable_new_mechanism','explicit stage scope')
        previous=initial if s['local_stage_id']==0 else arrays[sid-1]['states'][-1]
        x=a.batch_state(base,s['initial']);R(a.same_bytes(x,previous),'exact state carry/reset boundary')
        blocks=[a.npz(a.checked(base,r)) for r in s['blocks']];arr=a.reconstruct(x,blocks,want['controls']);arrays[sid]=arr
        n=want['steps'];R(arr['updates']==n==s['completed_steps']==s['attempted_steps']==s['returned_steps'],'full stage counts')
        slots=[c for c in result['calls'] if c['stage_id']==sid]
        cache,_,_=a.audit_cache_stage(base,s,slots,x,prefix)
        R(cache['diagnostic']['passed'],'cache comparison failed')
        for slot in slots:a.inspect_call(base,slot,arr,cache,prefix)
        observation=json.loads((base/f'stage-{sid:02d}-observation.json').read_text())
        R(observation['stage_id']==sid and observation['controls']==want['controls'] and observation['off_hold']==(want['controls']!=HOLD),'observation controls')
        reads=[];periods=[]
        for i in (0,1):
            read=a.read_cycle(arr['states'][1:,i],refs);a.neutral_match(observation['readouts'][i],read);reads.append(read);periods.append(recurrence(arr['states'][1:,i]))
        ps=[p['qualified_primitive_period'] for p in periods]
        gap=cycle_gap(arr['states'][1:,0],arr['states'][1:,1],ps[0]) if ps[0] is not None and ps[0]==ps[1] else None
        stage_reports.append(dict(stage_id=sid,controls=want['controls'],steps=n,readouts=reads,recurrence=periods,aligned_interhistory_gap=gap,diagnostics_passed=True))
    truth=[];pairs=[]
    for gid,(g,want) in enumerate(zip(result['groups'],groups,strict=True)):
        R(all(g[k]==v for k,v in want.items()) and g['status']=='completed' and g['diagnostics_passed'] is True,'group contract')
        R(g==json.loads((base/f'group-{gid:02d}.json').read_text()),'group shard')
        R(g['initial_state_sha256']==a.array_digest(initial) and g['terminal_state_sha256']==a.array_digest(arrays[want['stage_ids'][-1]]['states'][-1]),'group initial/final bytes')
        R(g['observations']==[json.loads((base/f'stage-{sid:02d}-observation.json').read_text()) for sid in want['stage_ids']],'group observation inventory')
        for rec in g['artifacts']:a.checked(base,rec)
        if mode in ('and','or'):
            b,c=want['case']['b'],want['case']['c']
            for bit in (0,1):
                first=(bit & b) if mode=='and' else (bit | b);final=(first & c) if mode=='and' else (first | c)
                actual=[stage_reports[4*gid+offset]['readouts'][bit]['decoded_bit'] for offset in (1,3)]
                truth.append(dict(a=bit,b=b,c=c,first_expected=first,final_expected=final,first_actual=actual[0],final_actual=actual[1],passed=actual==[first,final]))
        else:
            left,right=arrays[4*gid]['states'],arrays[4*gid+2]['states'];rows=[]
            for bit in (0,1):
                p=stage_reports[4*gid]['recurrence'][bit]['qualified_primitive_period'];q=stage_reports[4*gid+2]['recurrence'][bit]['qualified_primitive_period']
                rows.append(dict(initial_bit=bit,outward_period=p,return_period=q,aligned_gap=cycle_gap(left[1:,bit],right[1:,bit],p) if p is not None and p==q else None))
            pairs.append(dict(direction=want['case']['direction'],middle_control_history_comparisons=rows))
    report.update(status='passed',all_cases_preserved=True,stage_reports=stage_reports,truth_table=truth,hysteresis_path_comparisons=pairs,model_calls=result['model_calls'],elapsed_seconds=result['elapsed_seconds'],peak_rss_bytes=result['peak_rss_bytes'],complete_truth_table_passed=all(x['passed'] for x in truth) if truth else None,scope='Exploratory finite-tail recurrence and strict reference readout. No stability, bifurcation, universal gate, or causal hysteresis proof.')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--mode',choices=['and','or','hysteresis'],required=True);p.add_argument('--dwell',type=int,choices=[240,480],default=240);p.add_argument('--output',type=Path,required=True);p.add_argument('--middle-fraction',type=float,choices=[.25,.5,.75],default=.5);args=p.parse_args()
    R(not args.output.exists(),'audit output already exists')
    report=dict(status='failed',audited_source_commit=args.commit,mode=args.mode,dwell=args.dwell,middle_fraction=args.middle_fraction,bindings=[],external_bindings=[],model_replay_calls=0,auditor_source=a.binding(__file__),numeric_helper=a.binding(a.__file__))
    try:audit(args.run,args.commit,args.mode,args.dwell,report)
    except Exception as e:report.update(error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    return 0 if report['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
