#!/usr/bin/env python3
"""One fixed public write to two numerical states, with saved scalar comparison."""
import argparse
import copy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')

import numpy as np
import benchmark_qwen_batch as batch
import run_public_command_memory as memory
import run_public_half_pulse as historical_source
from validate_qwen_batch_trajectory import array_hash, cache_hash, precision
import validate_qwen_batch_trajectory as trajectory_source

reference, contract = memory.reference, memory.contract
MESSAGES = ('Write bit 1 to register R0.', 'Write bit 1 to register R0.')
STARTS, OLD_TRIALS = (0, 1), (2, 3)
HISTORICAL_SOURCE = 'f0c757ed2c7142bd2dd5cb78a796f69f7f49f4c4'
MAX_CALLS, MAX_UPDATES = 366, 360
MAX_SECONDS, EXTERNAL_SECONDS, GRACE = 600, 780, 30
MAX_RSS, MAX_VM = 8*1024**3, 16*1024**3
MAX_OUTPUT, FINAL_RESERVE = 32*1024**2, 2*1024**2
WORKING_OUTPUT = MAX_OUTPUT-FINAL_RESERVE
ARCHIVE_CAP, TRANSFER_RESERVATION, SOURCE_UPLOAD_CAP = 48*1024**2, 64*1024**2, 8*1024**2
FAILED_RETURN_BYTES = 2473984
DIAGNOSTIC_TOLERANCE, TRAJECTORY_TOLERANCE = 1e-12, 1e-10
BLOCK = 12
PRECISION_SCOPE = ('Existing native64 normalization/rotary patches with fresh per-call observations, '
    'CPU math SDPA and one thread; two stage-specific cached/full batch checks. '
    'No new derivative-convergence test or global precision proof.')


@dataclass(frozen=True)
class WritePlan:
    requested_bit: int = 1

    def __post_init__(self):
        if type(self.requested_bit) is not int or self.requested_bit not in (0,1):
            raise ValueError('only the explicit public write0 or write1 plan is supported')

    @property
    def messages(self):
        text='Write bit 0 to register R0.' if self.requested_bit==0 else 'Write bit 1 to register R0.'
        return (text,text)

    @property
    def historical_trials(self):
        return (0,1) if self.requested_bit==0 else (2,3)


def validate_plan(plan=None):
    plan=WritePlan() if plan is None else plan
    if type(plan) is not WritePlan: raise ValueError('require an explicit immutable WritePlan')
    plan.__post_init__()
    return plan


def actions(plan=None):
    plan=validate_plan(plan)
    values = tuple(contract.parse_command(text) for text in plan.messages)
    for text, value in zip(plan.messages, values, strict=True): contract.verify_action(text, value)
    if values[0] != values[1] or [s.steps for s in values[0].stages] != [120,240]:
        raise ValueError('public actions differ from the fixed common schedule')
    return values


def stage_plan(plan=None):
    return [dict(stage_id=i, name=s.name, controls=dict(alpha=s.alpha,log10_temperature=s.log10_temperature),
        steps=s.steps, status='not_started', attempted_steps=0, returned_steps=0, completed_steps=0,
        blocks=[], initial=None, trajectory=None, cache_comparison=None, historical_comparison=None,
        failed_outputs=[]) for i,s in enumerate(actions(plan)[0].stages)]


def row_plan(plan=None):
    plan=validate_plan(plan)
    return [dict(row_id=i, message=text, source_reference_index=i, source_reference_label=('ad','td')[i],
        source_phase=0, historical_trial_id=plan.historical_trials[i], status='not_started', parser_outcome='not_started',
        action=None, dispatch_count=0, completed_updates=0, readout=None, historical_comparison=None)
        for i,text in enumerate(plan.messages)]


def call_plan(plan=None):
    rows=[]
    for stage in stage_plan(plan):
        for category,step,cached in [('prefix',0,None),('comparison',0,True),('comparison',0,False)]+[
                ('update',step,True) for step in range(1,stage['steps']+1)]:
            rows.append(dict(call=len(rows)+1,stage_id=stage['stage_id'],stage=stage['name'],
                category=category,step=step,cached=cached,status='not_started'))
    return rows


def artifact_envelope():
    parts=dict(state_bytes=362*2*896*8, token_time_bytes=360*24,
        cache_arrays_bytes=2*3*24*2*2*37*64*8,
        diagnostic_array_bytes=4*(3*2*896+2*151936)*8,
        container_allowance_bytes=512*1024, metadata_allowance_bytes=2*1024**2)
    total=sum(parts.values())
    if total != 28627392 or total+FAILED_RETURN_BYTES+16384 >= WORKING_OUTPUT:
        raise ValueError('planned and failed-return evidence do not fit working allowance')
    return dict(parts=parts,estimated_uncompressed_envelope_bytes=total,
        failed_return_reserved_bytes=FAILED_RETURN_BYTES,
        final_partial_block_bytes=12*(2*896*8+24),
        scope='Each state saved once; actual admission checks remain mandatory. No compression assumption.')


def derive_design(plan=None):
    plan=validate_plan(plan)
    stages=stage_plan(plan); calls=call_plan(plan)
    if len(calls)!=MAX_CALLS or sum(s['steps'] for s in stages)!=MAX_UPDATES:
        raise ValueError('fixed public schedule changed')
    return dict(schema_version=1,evidence='adapted exploratory batch-write engineering comparison',
        requested_bit=plan.requested_bit,messages=list(plan.messages),row_order=['ad phase zero','td phase zero'],
        historical_trial_ids=list(plan.historical_trials),
        adaptation=('Reverse-control write0 test after the fixed write1 batch passed; no change to prior outcomes.'
                    if plan.requested_bit==0 else 'Initial fixed write1 batch comparison.'),
        schedule=dict(batch_size=2,batch_updates=360,row_updates=720,prefix_calls=2,comparison_calls=4,
                      total_calls=366,endpoint_steps=120,hold_steps=240,reset_count=0),
        limits=dict(model_calls=MAX_CALLS,internal_seconds=MAX_SECONDS,external_seconds=EXTERNAL_SECONDS,
            termination_grace_seconds=GRACE,threads=1,peak_rss_bytes=MAX_RSS,external_vm_bytes=MAX_VM,
            working_raw_bytes=WORKING_OUTPUT,finalization_reserve_bytes=FINAL_RESERVE,total_raw_bytes=MAX_OUTPUT,
            archive_bytes=ARCHIVE_CAP,transfer_reservation_bytes=TRANSFER_RESERVATION,source_upload_bytes=SOURCE_UPLOAD_CAP),
        map=dict(model=reference.MODEL,revision=reference.REVISION,prefix_ids=reference.PREFIX_IDS,
                 hold=memory.writes.HOLD,endpoints=memory.writes.ENDPOINTS),
        diagnostic_tolerance=DIAGNOSTIC_TOLERANCE,trajectory_tolerance=TRAJECTORY_TOLERANCE,
        artifact_envelope=artifact_envelope(),precision_scope=PRECISION_SCOPE,
        evaluation='Neutral terminal gates and saved scalar comparison; no desired label enters a numerical update.',
        stopping='Execute the fixed schedule after finite diagnostic or trajectory disagreement. Abort on structural, '
                 'cache mutation, precision, execution or resource errors; no retry, reset, row removal or extension.',
        timing_scope='Measure absolute elapsed and update-forward times. Historical times are descriptive; '
                    '366 versus 732 is a prescribed call comparison, not a measured runtime speedup.',
        scope='Two states receive the same public write. No independent addressing, arbitrary register bank, '
              'all-phase reliability, semantic language success or hidden communication claim.')


class Recorder(memory.Recorder):
    def admit(self,size,final=False):
        if self.size()+size > (MAX_OUTPUT if final or self.finalizing else WORKING_OUTPUT):
            raise RuntimeError('raw artifact limit reached')


class Budget:
    def __init__(self,recorder,slots,plan=None):
        self.plan=validate_plan(plan)
        self.recorder,self.slots,self.calls=recorder,slots,0
        self.counts=dict(prefix=0,comparison=0,update=0)
        self.stage_id=None; self.max_rss=MAX_RSS

    def check(self):
        if self.recorder.clock()-self.recorder.started >= MAX_SECONDS: raise TimeoutError('600-second internal cap')
        if self.recorder.rss() >= MAX_RSS: raise MemoryError('8 GiB RSS cap')
        if self.recorder.size()>WORKING_OUTPUT: raise RuntimeError('working raw artifact limit reached')

    def begin(self,stage):
        expected=stage_plan(self.plan)[stage['stage_id']]
        if any(stage[k]!=expected[k] for k in ('name','controls','steps')):
            raise ValueError('stage differs from fixed public action')
        if self.calls>=MAX_CALLS or self.slots[self.calls]['stage_id']!=stage['stage_id']:
            raise ValueError('stage order differs from fixed call schedule')
        self.stage_id=stage['stage_id']

    def claim(self,label,category='prefix'):
        self.check()
        # Also reserve before prefixes. The stage cache snapshot has separate admission.
        self.recorder.admit(FAILED_RETURN_BYTES+16384)
        if self.calls>=MAX_CALLS: raise RuntimeError('model-call cap reached')
        slot=self.slots[self.calls]
        if slot['stage_id']!=self.stage_id or slot['category']!=category:
            raise RuntimeError('model call differs from fixed schedule')
        self.calls+=1; self.counts[category]+=1; slot.update(status='started',label=label)
        self.recorder.event('model_call_started',**slot)
        return slot


class Backend(batch.QwenBackend):
    """Stage-aware adapter; historical backend/module constants remain unchanged."""
    def __init__(self,args,recorder,budget):
        # Keep the original pinned vector, never recover it by dividing steering.
        reference.require_hash(args.vectors,reference.VECTOR_SHA256)
        with np.load(args.vectors,allow_pickle=False) as source: vector=source['philosophy'].copy()
        if vector.shape!=(896,) or vector.dtype!=np.float64 or not np.isfinite(vector).all():
            raise ValueError('invalid original steering vector')
        super().__init__(args,budget,recorder)
        self.vector=self.torch.tensor(vector,dtype=self.torch.float64)

    def prepare(self,controls):
        self.observations.clear()
        memory.writes.set_controls(self.loop,self.vector,controls)
        if not precision(self.observations): raise ValueError('prefix lacks fresh native64 observations')


def qualify_historical(args,refs,receipt_path=None,plan=None):
    plan=validate_plan(WritePlan(getattr(args,'requested_bit',1)) if plan is None else plan)
    if hasattr(args,'requested_bit') and (type(args.requested_bit) is not int or args.requested_bit!=plan.requested_bit):
        raise ValueError('historical qualifier plan differs from explicit requested bit')
    root=Path(__file__).resolve().parents[1]
    receipt_path=Path(receipt_path) if receipt_path else root/'docs/receipts/p6-writes.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt['independent_audit_status']!='audit_passed' or receipt['experiment_source_commit']!=HISTORICAL_SOURCE:
        raise ValueError('historical receipt differs from audited fixed source')
    bindings=[]; loaded={}
    for name in ('results.json','metadata.json'):
        matches=[b for b in receipt['bindings'] if b['path'].endswith('/run/'+name)]
        if len(matches)!=1: raise ValueError('historical result binding missing or ambiguous')
        path=historical_source.require_bound(args.historical_run,{**matches[0],'path':name})
        loaded[name]=json.loads(path.read_text()); bindings.append(memory.binding(path))
    result,meta=loaded['results.json'],loaded['metadata.json']
    if result['status']!='passed' or meta['source_commit']!=HISTORICAL_SOURCE:
        raise ValueError('historical run did not complete from declared source')
    if (meta['model']!=reference.MODEL or meta['revision']!=reference.REVISION
            or meta['prefix_ids']!=reference.PREFIX_IDS or meta['hold']!=memory.writes.HOLD
            or meta['candidate_endpoints']!=memory.writes.ENDPOINTS or meta['threads']!=1):
        raise ValueError('historical map inputs differ')
    for name,digest in (('weights',reference.WEIGHTS_SHA256),('config',memory.hold_test.CONFIG_SHA256),
                         ('vectors',reference.VECTOR_SHA256)):
        if meta[name]['sha256']!=digest: raise ValueError('historical model input differs: '+name)
    for key in ('references_sha256','reference_manifest_sha256','recovery_results_sha256'):
        if meta['arguments'][key]!=getattr(args,key): raise ValueError('historical prerequisite differs: '+key)
    for module in (reference,memory.hold_test,memory.recovery):
        path=Path(module.__file__)
        if meta['source_hashes'][path.name]!=reference.sha256(path): raise ValueError('historical mathematical source differs')
    histories=[]; controls=[]
    for index,old_id in enumerate(plan.historical_trials):
        row=result['rows'][old_id]
        if (row['trial_id']!=old_id or row['source_index']!=index or row['phase']!=0
                or row['status']!='completed' or row['completed_steps']!=360):
            raise ValueError('historical row identity/completion differs')
        previous=refs[index,0]; stages=[]; stage_bindings=[]
        for old,declared in zip(row['stages'],actions(plan)[index].stages,strict=True):
            controls_expected=dict(alpha=declared.alpha,log10_temperature=declared.log10_temperature)
            if (old['name']!=declared.name or old['controls']!=controls_expected
                    or old['steps']!=declared.steps or old['completed_steps']!=declared.steps
                    or old['status']!='completed' or not old['cache_comparison']['passed']):
                raise ValueError('historical stage controls/completion differs')
            path=historical_source.require_bound(args.historical_run,old['trajectory'])
            bindings.append(memory.binding(path)); stage_bindings.append(memory.binding(path))
            with np.load(path,allow_pickle=False) as data:
                states,ids,seconds=data['states'].copy(),data['token_ids'].copy(),data['step_seconds'].copy()
                if (states.shape!=(declared.steps+1,896) or states.dtype!=np.float64 or not np.isfinite(states).all()
                        or ids.shape!=(declared.steps,) or ids.dtype!=np.int64 or np.any(ids<0) or np.any(ids>=batch.VOCAB)
                        or seconds.shape!=ids.shape or seconds.dtype!=np.float64 or not np.isfinite(seconds).all()
                        or np.any(seconds<=0) or int(data['first_state_index'])!=0
                        or float(data['alpha'])!=declared.alpha or float(data['log10_temperature'])!=declared.log10_temperature
                        or states[0].tobytes()!=previous.tobytes()):
                    raise ValueError('historical array contract/continuity differs')
            previous=states[-1].copy(); stages.append(dict(states=states,token_ids=ids))
        neutral=memory.neutral_readout(stages[1]['states'][1:],refs)
        if neutral['qualified_label'] is None: raise ValueError('historical hold is unqualified')
        histories.append(stages); controls.append(dict(trial_id=old_id,source_index=index,model_calls=366,
            trajectories=stage_bindings,neutral_readout=neutral))
    return histories,dict(receipt=memory.binding(receipt_path),source_commit=HISTORICAL_SOURCE,bindings=bindings,
        requested_bit=plan.requested_bit,messages=list(plan.messages),historical_trial_ids=list(plan.historical_trials),
        controls=controls,timing_scope='Historical scalar trajectories; not contemporaneous speed controls.')


class BlockArchive:
    def __init__(self,recorder,stage,state):
        self.recorder,self.stage=recorder,stage
        self.pending=[]; self.tokens=[]; self.seconds=[]; self.saved=0
        self.all_states=[state.copy()]; self.all_tokens=[]
        if stage['initial'] is None:
            stage['initial']=recorder.state(f'stage-{stage["stage_id"]:02d}-initial.npy',state)
        else:
            path=historical_source.require_bound(recorder.directory,stage['initial'])
            saved=np.load(path,allow_pickle=False)
            if saved.dtype!=state.dtype or saved.shape!=state.shape or saved.tobytes()!=state.tobytes():
                raise ValueError('frozen initial differs from carried state')

    def append(self,state,tokens,seconds):
        self.pending.append(state.copy()); self.tokens.append(tokens.copy()); self.seconds.append(seconds)
        self.all_states.append(state.copy()); self.all_tokens.append(tokens.copy())
        self.stage['completed_steps']+=1

    def flush(self,final=False):
        if not self.pending: return
        if not final and len(self.pending)<BLOCK: return
        end=self.saved+len(self.pending)
        name=f'stage-{self.stage["stage_id"]:02d}-block-{end:03d}.npz'
        record=self.recorder.npz(name,final=final,states=np.stack(self.pending),
            token_ids=np.stack(self.tokens),step_seconds=np.asarray(self.seconds,dtype=np.float64),
            first_state_index=np.asarray(self.saved+1,dtype=np.int64),last_state_index=np.asarray(end,dtype=np.int64),
            **{k:np.asarray(v,dtype=np.float64) for k,v in self.stage['controls'].items()})
        self.stage['blocks'].append(record); self.saved=end
        self.pending=[]; self.tokens=[]; self.seconds=[]
        self.recorder.event('block_saved',_final=final,stage_id=self.stage['stage_id'],last_state_index=end,artifact=record)

    def finalize(self):
        self.flush(final=True)
        report=dict(stage_id=self.stage['stage_id'],initial=self.stage['initial'],blocks=self.stage['blocks'],
            completed_steps=self.stage['completed_steps'],state_shape=[self.stage['completed_steps']+1,2,896],
            status=self.stage['status'],controls=self.stage['controls'],format='initial plus nonoverlapping post-update blocks')
        self.stage['trajectory']=self.recorder.json(f'stage-{self.stage["stage_id"]:02d}-trajectory.json',report,final=True)


def reconstruct(directory,stage):
    """Actual immutable file contract, also used by lightweight integration tests."""
    directory=Path(directory)
    path=historical_source.require_bound(directory,stage['initial']); initial=np.load(path,allow_pickle=False)
    if initial.shape!=(2,896) or initial.dtype!=np.float64 or not np.isfinite(initial).all():
        raise ValueError('invalid stage initial state')
    states=[initial[None]]; tokens=[]; seconds=[]; cursor=0
    for record in stage['blocks']:
        path=historical_source.require_bound(directory,record)
        with np.load(path,allow_pickle=False) as data:
            y,ids,t=data['states'],data['token_ids'],data['step_seconds']; n=len(y)
            if (not 1<=n<=BLOCK or y.shape!=(n,2,896) or y.dtype!=np.float64 or not np.isfinite(y).all()
                    or ids.shape!=(n,2) or ids.dtype!=np.int64 or np.any(ids<0) or np.any(ids>=batch.VOCAB)
                    or t.shape!=(n,) or t.dtype!=np.float64 or not np.isfinite(t).all() or np.any(t<=0)
                    or data['first_state_index'].shape!=() or data['first_state_index'].dtype!=np.int64
                    or data['last_state_index'].shape!=() or data['last_state_index'].dtype!=np.int64
                    or int(data['first_state_index'])!=cursor+1 or int(data['last_state_index'])!=cursor+n
                    or any(float(data[k])!=v for k,v in stage['controls'].items())):
                raise ValueError('invalid/gapped block contract')
            states.append(y.copy()); tokens.append(ids.copy()); seconds.append(t.copy()); cursor+=n
    if cursor!=stage['completed_steps']: raise ValueError('block count differs from completed steps')
    return np.concatenate(states),np.concatenate(tokens) if tokens else np.empty((0,2),dtype=np.int64),\
        np.concatenate(seconds) if seconds else np.empty(0,dtype=np.float64)


def compare_history(states,tokens,histories,stage_id,plan=None):
    plan=validate_plan(plan)
    rows=[]
    for i in range(2):
        old=histories[i][stage_id]; count=len(states)
        if old['states'].shape[0]<count: raise ValueError('historical trajectory too short')
        gap=float(np.max(np.abs(states[:,i]-old['states'][:count])))
        mismatches=np.flatnonzero(tokens[:,i]!=old['token_ids'][:count-1]).tolist()
        rows.append(dict(row_id=i,historical_trial_id=plan.historical_trials[i],compared_states=count,compared_updates=count-1,
            maximum_coordinate_gap=gap,coordinate_tolerance=TRAJECTORY_TOLERANCE,
            token_mismatch_update_indices=[n+1 for n in mismatches],token_mismatch_count=len(mismatches),
            state_bytes_equal=states[:,i].tobytes()==old['states'][:count].tobytes(),
            token_bytes_equal=tokens[:,i].tobytes()==old['token_ids'][:count-1].tobytes(),
            passed=gap<=TRAJECTORY_TOLERANCE and not mismatches))
    return dict(rows=rows,passed=all(r['passed'] for r in rows))


def same_bytes(left,right):
    return set(left)==set(right) and all(left[k].dtype==right[k].dtype and left[k].shape==right[k].shape
        and left[k].tobytes()==right[k].tobytes() for k in left)


def check_return(y,p,seconds,extra):
    return (y.shape==(2,896) and p.shape==(2,batch.VOCAB) and y.dtype==p.dtype==np.float64
        and np.isfinite(y).all() and np.isfinite(p).all() and np.all(p>=0) and np.all(p<=1)
        and np.allclose(p.sum(-1),1.,rtol=0,atol=DIAGNOSTIC_TOLERANCE)
        and math.isfinite(seconds) and seconds>0 and extra['observed_output_dtype']=='torch.float64'
        and extra['observed_probability_dtype']=='torch.float64')


def evaluate(backend,state,stage,recorder,budget,prefix,archive):
    planned=budget.slots[budget.calls]
    category,cached=planned['category'],planned['cached']
    slot=budget.claim(f"{stage['name']}-{category}-{planned['step']}-{cached}",category)
    slot['forward_attempted']=False
    original=state.copy(); arrays=dict(input=original)
    if category=='update': stage['attempted_steps']+=1
    returned=None
    try:
        before=backend.cache_arrays()
        slot['forward_attempted']=True
        y,p,seconds,extra=backend.forward(original.copy(),cached)
        returned=(y,p); arrays.update(output=y,probabilities=p)
        if category=='update': stage['returned_steps']+=1
        arrays['input_after']=extra['input_after']
        slot.update(seconds=seconds if math.isfinite(seconds) else None,
            input_sha256=array_hash(original),output_sha256=array_hash(y),probabilities_sha256=array_hash(p),
            dtype_observations=copy.deepcopy(backend.observations),
            output_dtype=extra['observed_output_dtype'],probability_dtype=extra['observed_probability_dtype'])
        after=backend.cache_arrays()
        slot.update(cache_before_sha256=cache_hash(before),cache_after_sha256=cache_hash(after),
            cache_length_before=int(before['length']),cache_length_after=int(after['length']),
            cache_unchanged=bool(batch.valid_cache(before) and batch.valid_cache(after)
                and same_bytes(before,prefix) and same_bytes(after,prefix)),
            input_unchanged=bool(extra['input_after'].shape==original.shape and extra['input_after'].dtype==original.dtype
                and extra['input_after'].tobytes()==original.tobytes()),
            precision_passed=bool(precision(backend.observations)),outputs_valid=bool(check_return(y,p,seconds,extra)))
        slot['expanded_cache_valid']=True
        if cached:
            expanded={**extra['used_cache_before'],'length':np.asarray(len(reference.PREFIX_IDS),dtype=np.int64)}
            expected={k:v if k=='length' else np.repeat(v,2,axis=0) for k,v in prefix.items()}
            slot.update(expanded_cache_sha256=cache_hash(expanded),used_cache_length_after=int(extra['used_cache_length_after']),
                expanded_cache_valid=bool(batch.valid_cache(expanded,2) and same_bytes(expanded,expected)
                    and int(extra['used_cache_length_after'])==len(reference.PREFIX_IDS)+1))
            if category=='comparison':
                stage['expanded_prefix_cache']=recorder.npz(f'stage-{stage["stage_id"]:02d}-expanded-prefix.npz',**expanded)
        if not all(slot[k] for k in ('outputs_valid','input_unchanged','cache_unchanged','expanded_cache_valid','precision_passed')):
            raise ValueError('finite/native64/input/cache structural gate failed')
        slot['top_tokens']=p.argmax(-1).tolist()
        if category=='update':
            # Record actual admitted mutation before resource checks and I/O.
            archive.append(y,np.asarray(slot['top_tokens'],dtype=np.int64),seconds); state[:]=y
            slot['state_carried']=True
        budget.check(); slot['status']='returned'
    except BaseException as exc:
        slot.update(status=memory.error_status(exc),error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        with batch.preserve_call_files():
            if category=='comparison' or slot['status']!='returned':
                filename=f'call-{slot["call"]:03d}-'+('diagnostic' if category=='comparison' else 'failed')+'.npz'
                slot['artifact']=recorder.npz(filename,**arrays)
                if category=='update': stage['failed_outputs'].append(slot['artifact'])
            recorder.event('model_call_completed',_final=slot['status']!='returned',**slot)
            if category=='update': archive.flush()
    return returned


def run(args,backend_factory=Backend,qualifier=memory.qualify_inputs,historical_qualifier=qualify_historical,
        clock=time.monotonic,rss=reference.rss_bytes,install_signals=True):
    plan=validate_plan(WritePlan(getattr(args,'requested_bit',1)))
    recorder=Recorder(args.output_dir,clock,rss)
    rows,stages,slots=row_plan(plan),stage_plan(plan),call_plan(plan); budget=Budget(recorder,slots,plan)
    result=dict(schema_version=1,status='failed',rows=rows,stages=stages,calls=slots,reset_count=0,
        evidence='adapted exploratory batch-write engineering comparison',numerical_equivalence_passed=None,
        requested_bit=plan.requested_bit,timing_scope=derive_design(plan)['timing_scope'],precision_scope=PRECISION_SCOPE)
    backend=state=None; old_handlers={}
    if install_signals:
        def stop(signum,frame):
            if signum==signal.SIGALRM: raise TimeoutError('declared internal deadline reached')
            raise InterruptedError(f'received signal {signum}')
        old_handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
        for s in old_handlers: signal.signal(s,stop)
        signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    try:
        recorder.json('design.json',derive_design(plan)); recorder.json('planned-rows.json',rows)
        root=Path(__file__).resolve().parents[1]
        paths=[Path(m.__file__) for m in (memory,contract,memory.writes,memory.recovery,memory.hold_test,reference,
                                        batch,historical_source,trajectory_source)]
        paths += [Path(__file__),root/'docs/public-batch-writes.md',root/'docs/receipts/p6-writes.json',
                  root/'docs/receipts/p6-write-path-design.json',root/'pyproject.toml',root/'uv.lock']
        if plan.requested_bit==0: paths += [root/'docs/public-batch-zero.md',root/'docs/receipts/public-batch-writes.json']
        result['source']=dict(utc=reference.utc_now(),source_commit=subprocess.check_output(
            ['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),files=[memory.binding(p) for p in paths],
            arguments={k:str(v) for k,v in vars(args).items()})
        recorder.json('source.json',result['source'])
        refs,provenance=qualifier(args)
        if refs.shape!=(2,6,896) or refs.dtype!=np.float64 or not np.isfinite(refs).all():
            raise ValueError('invalid qualified reference cycles')
        recorder.json('qualified-inputs.json',provenance)
        histories,historical=historical_qualifier(args,refs,plan=plan)
        recorder.json('historical-inputs.json',historical)
        state=refs[:,0].copy()
        stages[0]['initial']=recorder.state('stage-00-initial.npy',state)
        for row,action in zip(rows,actions(plan),strict=True):
            row['action']={**asdict(action),'stages':[asdict(s) for s in action.stages]}
            row['parser_outcome']='accepted'
            def dispatch(parsed):
                contract.verify_action(row['message'],parsed); row['dispatch_count']+=1
            contract.dispatch_command(row['message'],dispatch,prepared_action=action)
        recorder.json('frozen-rows.json',rows)
        recorder.event('inputs_frozen',historical_source=historical.get('source_commit'),initial_sha256=array_hash(state))
        budget.check(); started=clock(); backend=backend_factory(args,recorder,budget)
        result['cold_initialization_seconds']=clock()-started
        if hasattr(backend,'bindings'): recorder.json('model-inputs.json',backend.bindings)
        for row in rows: row['status']='running'
        for stage in stages:
            archive=None; prefix=None; stage['status']='running'
            try:
                budget.begin(stage); archive=BlockArchive(recorder,stage,state)
                try:
                    backend.prepare(stage['controls'])
                    if not precision(backend.observations): raise ValueError('prefix native64 observations missing')
                    slot=slots[budget.calls-1]; slot['dtype_observations']=copy.deepcopy(backend.observations)
                    prefix=backend.cache_arrays()
                    if not batch.valid_cache(prefix): raise ValueError('invalid scalar prefix cache')
                    stage['scalar_prefix_cache']=recorder.npz(f'stage-{stage["stage_id"]:02d}-scalar-prefix.npz',**prefix)
                    slot['status']='returned'; budget.check()
                except BaseException as exc:
                    if budget.calls and slots[budget.calls-1]['category']=='prefix':
                        slots[budget.calls-1].update(status=memory.error_status(exc),error=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    if budget.calls and slots[budget.calls-1]['category']=='prefix':
                        recorder.event('model_call_completed',_final=True,**slots[budget.calls-1])
                diagnostics=[evaluate(backend,state,stage,recorder,budget,prefix,archive) for _ in range(2)]
                stage['cache_comparison']=batch.compare(*diagnostics)
                for _ in range(stage['steps']): evaluate(backend,state,stage,recorder,budget,prefix,archive)
                stage['status']='completed'
            except BaseException as exc:
                stage.update(status=memory.error_status(exc),error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                with batch.preserve_call_files():
                    if prefix is not None:
                        try:
                            after=backend.cache_arrays()
                            stage['final_scalar_cache_sha256']=cache_hash(after)
                            stage['final_scalar_cache_unchanged']=same_bytes(prefix,after)
                            if not stage['final_scalar_cache_unchanged']:
                                stage.update(status='failed',final_cache_error='final scalar cache changed')
                        except BaseException as exc:
                            stage.update(status='failed',final_scalar_cache_unchanged=False,
                                final_cache_error=f'{type(exc).__name__}: {exc}')
                    if archive is not None:
                        archive.finalize()
                        saved,ids,_=reconstruct(recorder.directory,stage)
                        stage['historical_comparison']=compare_history(saved,ids,histories,stage['stage_id'],plan)
                        if stage['status']=='completed' and stage['name']=='hold':
                            for i,row in enumerate(rows): row['readout']=memory.neutral_readout(saved[1:,i],refs)
                    recorder.event('stage_finished',_final=True,stage_id=stage['stage_id'],status=stage['status'],
                        completed_steps=stage['completed_steps'])
            if stage['status']!='completed': raise ValueError('stage failed final cache gate')
        if budget.calls!=MAX_CALLS or budget.counts!=dict(prefix=2,comparison=4,update=360):
            raise ValueError('completed schedule count differs')
        for i,row in enumerate(rows):
            comparisons=[s['historical_comparison']['rows'][i] for s in stages]
            old_label=historical['controls'][i]['neutral_readout']['qualified_label']
            row['historical_comparison']=dict(compared_states=361,compared_updates=360,
                maximum_coordinate_gap=max(c['maximum_coordinate_gap'] for c in comparisons),
                token_mismatch_count=sum(c['token_mismatch_count'] for c in comparisons),
                state_bytes_equal=all(c['state_bytes_equal'] for c in comparisons),
                token_bytes_equal=all(c['token_bytes_equal'] for c in comparisons),
                final_labels_equal=row['readout']['qualified_label']==old_label,
                historical_qualified_label=old_label,
                passed=all(c['passed'] for c in comparisons) and row['readout']['qualified_label']==old_label)
            row['status']='completed'
        result['numerical_equivalence_passed']=all(s['cache_comparison']['passed'] for s in stages) and all(
            r['historical_comparison']['passed'] for r in rows)
        result['status']='completed'
    except BaseException as exc:
        result.update(status=memory.error_status(exc),error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        for row in rows:
            if row['status']=='running': row.update(status=memory.error_status(exc),error=result['error'])
    finally:
        if install_signals:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig in old_handlers: signal.signal(sig,signal.SIG_IGN)
        recorder.finalizing=True
        try:
            for row in rows:
                row['completed_updates']=sum(s['completed_steps'] for s in stages)
                recorder.json(f'row-{row["row_id"]:02d}.json',row,final=True)
            result.update(model_calls=budget.calls,call_counts=budget.counts,
                completed_batch_updates=sum(s['completed_steps'] for s in stages),
                completed_row_updates=2*sum(s['completed_steps'] for s in stages),
                elapsed_seconds=clock()-recorder.started,peak_rss_bytes=rss())
            result['terminal_state_sha256']=array_hash(state) if state is not None else None
            result['resource_limits_passed']=(budget.calls<=MAX_CALLS and result['elapsed_seconds']<MAX_SECONDS
                and result['peak_rss_bytes']<MAX_RSS and recorder.size()<=MAX_OUTPUT)
            recorder.json('result.json',result,final=True)
            recorder.event('finished',status=result['status'],model_calls=budget.calls)
            recorder.stream.close()
            recorder.json('manifest.json',dict(files=[memory.binding(p) for p in sorted(recorder.directory.iterdir())
                if p.is_file() and p.name!='manifest.json']),final=True)
        finally:
            for sig,handler in old_handlers.items(): signal.signal(sig,handler)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requested-bit',type=int,choices=(0,1),default=1)
    for name in ('cache-dir','vectors','references','reference-manifest','source-hold-results',
                 'recovery-results','historical-run','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('references-sha256','reference-manifest-sha256','recovery-results-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    for digest in (args.references_sha256,args.reference_manifest_sha256,args.recovery_results_sha256):
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            parser.error('input hashes must be lowercase SHA-256 digests')
    return 0 if run(args)['status']=='completed' else 1


if __name__=='__main__': raise SystemExit(main())
