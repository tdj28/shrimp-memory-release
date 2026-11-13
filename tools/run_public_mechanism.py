#!/usr/bin/env python3
"""Bounded public continuation loops and stateful truth-table chains."""
import argparse, copy, functools, json, signal, subprocess, time, traceback
from pathlib import Path
import run_public_repair_handoff as engine
import numpy as np
prior, memory, reference, batch = engine.prior, engine.memory, engine.reference, engine.batch
ROOT=Path(__file__).resolve().parents[1]
MODES=('hysteresis','and','or')
MAX_SECONDS=3600
MAX_RSS=8*1024**3
MAX_OUTPUT=512*1024**2
WORKING_OUTPUT=MAX_OUTPUT-16*1024**2
HOLD=memory.writes.HOLD
ENDPOINTS=[engine.endpoint_controls(b,False) for b in (0,1)]


def control(input_bit,mode):
    if type(input_bit) is not int or input_bit not in (0,1) or mode not in ('and','or'):
        raise ValueError('explicit gate input and mode required')
    # This function never receives the stored first input or any decoded output.
    return (ENDPOINTS[0] if input_bit==0 else HOLD).copy() if mode=='and' else (HOLD if input_bit==0 else ENDPOINTS[1]).copy()


def plans(mode,dwell=240,middle_fraction=.5):
    if mode not in MODES or dwell not in (240,480):raise ValueError('unsupported fixed design')
    if mode!='hysteresis' and dwell!=240:raise ValueError('gate dwell fixed')
    if middle_fraction not in (.25,.5,.75) or (mode!='hysteresis' and middle_fraction!=.5):raise ValueError('unsupported loop fraction')
    groups=[];stages=[];calls=[]
    cases=[dict(direction=b) for b in (0,1)] if mode=='hysteresis' else [dict(b=b,c=c) for b in (0,1) for c in (0,1)]
    for gid,case in enumerate(cases):
        name=f'loop-{case["direction"]}' if mode=='hysteresis' else f'{mode}-b{case["b"]}-c{case["c"]}'
        g=dict(group_id=gid,name=name,case=case,stage_ids=[],status='not_started',initial_state_sha256=None,observations=[])
        if mode=='hysteresis':
            e=ENDPOINTS[case['direction']]
            controls=[{k:HOLD[k]+f*(e[k]-HOLD[k]) for k in HOLD} for f in (middle_fraction,1.,middle_fraction,0.)]
            ns=[dwell]*4
        else:
            controls=[control(case['b'],mode),HOLD.copy(),control(case['c'],mode),HOLD.copy()];ns=[120,240,120,240]
        for lid,(ct,n) in enumerate(zip(controls,ns,strict=True)):
            st=copy.deepcopy(prior.stage_plan(prior.WritePlan(0))[0]);sid=len(stages)
            st.update(stage_id=sid,group_id=gid,group=name,local_stage_id=lid,name=f'leg-{lid}',controls=ct,steps=n,
                nominal_controls=ct.copy(),historical_comparison_scope='not_applicable_new_mechanism',historical_comparison=None)
            stages.append(st);g['stage_ids'].append(sid)
            for cat,step,cached in [('prefix',0,None),('comparison',0,True),('comparison',0,False)]+[('update',i,True) for i in range(1,n+1)]:
                calls.append(dict(call=len(calls)+1,group_id=gid,group=name,stage_id=sid,stage=st['name'],category=cat,step=step,cached=cached,status='not_started'))
        groups.append(g)
    return groups,stages,calls


def design(mode,dwell,middle_fraction=.5):
    g,s,c=plans(mode,dwell,middle_fraction)
    return dict(profile='public-mechanism-'+mode,mode=mode,dwell=dwell,middle_fraction=middle_fraction,groups=g,stages=s,
        reference_phases=[0,0],batch_initial_bits=[0,1],reference_initializations=len(g),
        calls=len(c),updates=sum(x['steps'] for x in s),limits=dict(seconds=MAX_SECONDS,rss=MAX_RSS,raw=MAX_OUTPUT),
        scope='Exploratory finite-precision pilot. No quasistatic hysteresis or general gate claim from a single dwell/phase. All intermediate observations and failures retained.')


class Recorder(memory.Recorder):
    def admit(self,size,final=False):
        if self.size()+size>(MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):raise RuntimeError('raw budget')


class Budget:
    def __init__(self,recorder,mode,dwell,slots,planner=plans):
        self.recorder=recorder;self.slots=slots;self.calls=0;self.counts=dict(prefix=0,comparison=0,update=0);self.stage_id=None;self.max_rss=MAX_RSS
        _,self.stages,self.expected=planner(mode,dwell)
        if self.expected!=slots:raise ValueError('call contract')
    def check(self):
        if self.recorder.clock()-self.recorder.started>=MAX_SECONDS:raise TimeoutError('wall cap')
        if self.recorder.rss()>=MAX_RSS:raise MemoryError('RSS cap')
        if self.recorder.size()>WORKING_OUTPUT:raise RuntimeError('raw cap')
    def begin(self,stage):
        e=self.stages[stage['stage_id']]
        if any(stage[k]!=e[k] for k in ('name','group_id','group','local_stage_id','steps','controls')):raise ValueError('stage contract')
        self.stage_id=stage['stage_id']
    def claim(self,label,category='prefix'):
        self.check();self.recorder.admit(prior.FAILED_RETURN_BYTES+16384)
        if self.calls>=len(self.expected):raise RuntimeError('call cap')
        slot=self.slots[self.calls];e=self.expected[self.calls]
        if slot['category']!=category or slot['stage_id']!=self.stage_id or any(slot[k]!=e[k] for k in ('call','group_id','stage_id','step','cached')):raise ValueError('call contract')
        self.calls+=1;self.counts[category]+=1;slot.update(status='started',label=label);self.recorder.event('model_call_started',**slot);return slot


def run(args,backend_factory=prior.Backend,qualifier=memory.qualify_inputs,clock=time.monotonic,rss=reference.rss_bytes,install_signals=True,plan_factory=plans,design_factory=design,extra_sources=()):
    fraction=getattr(args,'middle_fraction',.5)
    if plan_factory is plans:plan_factory=functools.partial(plans,middle_fraction=fraction)
    if design_factory is design:design_factory=functools.partial(design,middle_fraction=fraction)
    groups,stages,slots=plan_factory(args.mode,args.dwell);rec=Recorder(args.output_dir,clock,rss);budget=Budget(rec,args.mode,args.dwell,slots,plan_factory)
    result=dict(schema_version=1,profile='public-mechanism-'+args.mode,mode=args.mode,dwell=args.dwell,status='failed',groups=groups,stages=stages,calls=slots,initializations=0)
    state=None;handlers={}
    if install_signals:
        def stop(signum,frame):raise InterruptedError(f'signal {signum}')
        for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,stop)
        signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    try:
        rec.json('design.json',design_factory(args.mode,args.dwell));rec.json('planned-groups.json',groups);rec.json('planned-stages.json',stages);rec.json('planned-calls.json',slots)
        modules=(engine,engine.continuation,engine.shared,prior,memory,engine.contract,memory.writes,memory.recovery,memory.hold_test,reference,batch,prior.historical_source,prior.trajectory_source)
        paths=[Path(m.__file__) for m in modules]+[Path(__file__),ROOT/'docs/hysteresis-and-logic-protocol.md',ROOT/'tests/test_public_mechanism.py',ROOT/'pyproject.toml',ROOT/'uv.lock']+list(extra_sources)
        result['source']=dict(utc=reference.utc_now(),source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),files=[memory.binding(p) for p in paths],arguments={k:str(v) for k,v in vars(args).items()})
        rec.json('source.json',result['source']);refs,prov=qualifier(args);engine.shared.check_references(refs);rec.json('qualified-inputs.json',prov)
        initial=np.stack([refs[0,0],refs[1,0]]);state=initial.copy();rec.state('reference-starts.npy',state)
        rec.event('inputs_frozen',initial_sha256=prior.array_hash(initial));budget.check();start=clock();backend=backend_factory(args,rec,budget);result['cold_initialization_seconds']=clock()-start
        if hasattr(backend,'bindings'):rec.json('model-inputs.json',backend.bindings)
        for g in groups:
            state=initial.copy();result['initializations']+=1;g.update(status='running',initial_state_sha256=prior.array_hash(state));rec.event('group_started',group_id=g['group_id'],initial_sha256=prior.array_hash(state))
            for sid in g['stage_ids']:
                engine.execute_stage(backend,state,stages[sid],rec,budget)
                saved,_,_=prior.reconstruct(rec.directory,stages[sid])
                # Off-hold reference matching is descriptive only, never a branch-existence gate.
                reads=[memory.neutral_readout(saved[1:,i],refs) for i in (0,1)]
                obs=dict(stage_id=sid,controls=stages[sid]['controls'],readouts=reads,off_hold=stages[sid]['controls']!=HOLD)
                g['observations'].append(obs);rec.json(f'stage-{sid:02d}-observation.json',obs)
            g.update(status='completed',terminal_state_sha256=prior.array_hash(state),diagnostics_passed=all(stages[i]['cache_comparison']['passed'] for i in g['stage_ids']))
            artifacts=engine.shared.immutable_group_artifacts(rec.directory,{**g,'row_ids':[]},stages,slots)
            extra=[memory.binding(rec.directory/f'stage-{sid:02d}-observation.json') for sid in g['stage_ids']]
            g['artifacts']=artifacts+extra;rec.json(f'group-{g["group_id"]:02d}.json',g);rec.event('group_finished',group_id=g['group_id'],status=g['status'])
        if budget.calls!=len(slots):raise ValueError('incomplete call schedule')
        result['status']='completed'
    except BaseException as exc:
        result.update(status=memory.error_status(exc),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        for g in groups:
            if g['status']=='running':g['status']=result['status']
    finally:
        if install_signals:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        rec.finalizing=True
        result.update(model_calls=budget.calls,call_counts=budget.counts,elapsed_seconds=clock()-rec.started,peak_rss_bytes=rss(),terminal_state_sha256=prior.array_hash(state) if state is not None else None)
        rec.json('result.json',result,final=True);rec.event('finished',status=result['status']);rec.stream.close()
        rec.json('manifest.json',dict(files=[memory.binding(p) for p in sorted(rec.directory.iterdir()) if p.is_file() and p.name!='manifest.json']),final=True)
        for sig,h in handlers.items():signal.signal(sig,h)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=MODES,required=True);p.add_argument('--dwell',type=int,choices=(240,480),default=240);p.add_argument('--middle-fraction',type=float,choices=(.25,.5,.75),default=.5);p.add_argument('--preflight-only',action='store_true')
    for name in ('cache-dir','vectors','references','reference-manifest','source-hold-results','recovery-results','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('references-sha256','reference-manifest-sha256','recovery-results-sha256'):p.add_argument('--'+name,required=True)
    args=p.parse_args()
    if args.preflight_only:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        try:
            refs,prov=memory.qualify_inputs(args);engine.shared.check_references(refs);out=dict(status='passed',design=design(args.mode,args.dwell,args.middle_fraction),provenance=prov,model_calls=0)
        except Exception as e:out=dict(status='failed',error=str(e),traceback=traceback.format_exc(),model_calls=0)
        (args.output_dir/'input-preflight.json').write_text(json.dumps(out,indent=2)+'\n');return 0 if out['status']=='passed' else 1
    return 0 if run(args)['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
