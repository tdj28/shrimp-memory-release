"""Compact, counted native-float64 state/temperature derivatives for E01.

Each cycle action saves every realized state, tangent and control. The prefix
cache depends on alpha only. Temperature tangents enter every feedback update.
"""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import numpy as np
import validate_qwen_map as instrument
from run_public_periodic_corrector import fresh_precision,qualify_model

ALPHA=.07293697809562831
H=-.15395151820007266
E0=-.1535
P=-.15361287955001818
N=-.1535987696062659
DELTA=E0-H
DIM=896
CAPS=dict(ordinary=8720,jvp_cycles=1200,augmented=200,seconds=43200,
          rss_bytes=28000000000,raw_bytes=2*2**30)

class LimitError(RuntimeError):pass
class InstrumentError(RuntimeError):pass

def finite(a,shape):
    a=np.asarray(a)
    if a.dtype!=np.float64 or a.shape!=shape or not np.isfinite(a).all():
        raise ValueError('finite float64 shape required: '+str(shape))
    return a.copy()

class Record:
    def __init__(self,path):
        self.directory=Path(path);self.directory.mkdir(parents=True,exist_ok=False)
        self.started=time.monotonic();self.stream=(self.directory/'events.jsonl').open('x')
        self.assigned=set();self.serial=0;self.closed=0;self.bytes=0;self.finalizing=False
        self.raw_limit=CAPS['raw_bytes']
    def event(self,event,**kw):
        row=dict(event=event,elapsed_seconds=time.monotonic()-self.started,**kw)
        line=json.dumps(row,allow_nan=False)+'\n';self.check(len(line.encode()))
        self.stream.write(line);self.stream.flush();self.bytes+=len(line.encode())
    def json(self,name,obj):
        if Path(name).name!=name:raise ValueError('flat artifact name required')
        raw=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
        self.check(len(raw))
        target=self.directory/name;temporary=self.directory/('.partial-'+name)
        if target.exists():raise FileExistsError(target)
        with temporary.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.link(temporary,target);temporary.unlink()
        self.bytes+=len(raw)
    def check(self,n):
        if self.bytes+n>self.raw_limit-(0 if self.finalizing else 16*2**20):raise LimitError('raw evidence cap')
    def arrays(self,label,**arrays):
        self.serial+=1;name=f'{self.serial:05d}-{label}.npz'
        self.check(sum(np.asarray(v).nbytes for v in arrays.values())+16384)
        with (self.directory/name).open('xb') as f:np.savez(f,**arrays)
        self.bytes+=(self.directory/name).stat().st_size
        return name
    def checkpoint(self,label,summary):
        self.closed+=1;name=f'checkpoint-{self.closed:04d}.json'
        rows=[]
        for p in sorted(self.directory.iterdir()):
            if p.is_file() and p.name not in self.assigned and p.name not in ('events.jsonl','result.json','manifest.json') and not p.name.startswith('checkpoint-'):
                rows.append(dict(path=p.name,bytes=p.stat().st_size,sha256=instrument.sha256(p)))
        self.json(name,dict(profile='public-e01',checkpoint_id=self.closed,closed=True,
                           label=label,summary=summary,artifacts=rows))
        self.assigned.update(r['path'] for r in rows)
        self.event('checkpoint_closed',file=name,label=label)

class Counts:
    def __init__(self,rec,*,prior=None,geometry_deadline_unix=None,campaign_deadline_unix=None):
        self.recorder=rec;self.ordinary=0;self.jvp_cycles=0;self.augmented=0;self.calls=0
        self.prior=dict(prior or {})
        self.geometry_deadline_unix=geometry_deadline_unix
        self.campaign_deadline_unix=campaign_deadline_unix
    def remaining_seconds(self):
        remaining=CAPS['seconds']-self.prior.get('elapsed_seconds',0)-(time.monotonic()-self.recorder.started)
        for deadline in (self.geometry_deadline_unix,self.campaign_deadline_unix):
            if deadline is not None:remaining=min(remaining,deadline-time.time())
        return remaining
    def check(self):
        if self.remaining_seconds()<=0:raise TimeoutError('original geometry/case wall deadline or cumulative time cap')
        if instrument.rss_bytes()>=CAPS['rss_bytes']:raise MemoryError('geometry RSS cap')
        self.recorder.check(0)
    def claim(self,label):
        self.claim_kind('ordinary');self.calls+=1
    def claim_kind(self,kind):
        self.check()
        if self.prior.get(kind,0)+getattr(self,kind)>=CAPS[kind]:raise LimitError(kind+' cumulative cap')
        setattr(self,kind,getattr(self,kind)+1)
    def report(self):
        return dict(ordinary=self.ordinary,jvp_cycles=self.jvp_cycles,augmented=self.augmented,
                    model_invocations=self.calls,elapsed_seconds=time.monotonic()-self.recorder.started,
                    peak_rss_bytes=instrument.rss_bytes())
    def case_report(self,fresh=None):
        out=dict(self.report() if fresh is None else fresh)
        for key in ('ordinary','jvp_cycles','augmented','model_invocations','elapsed_seconds'):
            out[key]+=self.prior.get(key,0)
        out['peak_rss_bytes']=max(out['peak_rss_bytes'],self.prior.get('peak_rss_bytes',0))
        return out

class Map:
    def __init__(self,args,rec,counts):
        for k in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY'):os.environ[k]='1'
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
        self.torch=torch;self.rec=rec;self.counts=counts
        snap,model_inputs=qualify_model(args);rec.json('model-inputs.json',model_inputs)
        tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
        ids=tok.apply_chat_template(instrument.MESSAGES,add_generation_prompt=True)
        ids=list(ids['input_ids'] if hasattr(ids,'keys') else ids)
        if ids!=instrument.PREFIX_IDS:raise InstrumentError('prefix differs')
        model=AutoModelForCausalLM.from_pretrained(snap,local_files_only=True,trust_remote_code=False,
            dtype=torch.float64,attn_implementation='sdpa').cpu().eval().requires_grad_(False)
        if any(p.dtype!=torch.float64 or p.device.type!='cpu' for p in model.parameters()):
            raise InstrumentError('model precision/device')
        self.observations={};patches=instrument.patch_native_float64(model,self.observations)
        with np.load(args.vectors,allow_pickle=False) as a:vec=a['philosophy'].copy()
        self.vector=torch.tensor(finite(vec,(DIM,)))
        self.loop=instrument.FeedbackMap(model,self.vector,ALPHA,P,counts,rec)
        self.loop.mode='native64';self.alpha=None;self.set_alpha(ALPHA)
        rec.json('backend-readiness.json',dict(patches=patches,precision=self.observations,
            threads=[torch.get_num_threads(),torch.get_num_interop_threads()],
            versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','torch','transformers')},
            thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
    def set_alpha(self,alpha):
        if self.alpha==alpha:return
        self.counts.check();self.alpha=float(alpha)
        self.loop.steering=self.alpha*self.vector.reshape(1,1,-1)
        self.observations.clear();self.loop.prepare()
        if not fresh_precision(self.observations):raise InstrumentError('prefix precision')
        self.rec.event('alpha_selected',alpha=self.alpha)
    def cycle(self,x,logtemp,*,v=None,dlogtemp=0.,label='cycle',steps=6,cached=True):
        self.last_cycle_file=None
        import torch.autograd.forward_ad as fw
        from torch.nn.attention import sdpa_kernel,SDPBackend
        if not 1<=steps<=13:raise ValueError('cycle evidence reservation supports 1..13 steps')
        # Reserve the complete action, events and a full failed probability vector
        # before producing evidence. The terminal reserve is separate.
        self.rec.check(4*2**20)
        torch=self.torch;x=finite(x,(DIM,));differentiated=v is not None
        if differentiated:
            v=finite(v,(DIM,));self.counts.claim_kind('jvp_cycles')
            if steps!=6:raise ValueError('JVP counter unit is six steps')
        elif dlogtemp!=0.:raise ValueError('temperature tangent needs state tangent')
        traces=[x.copy()];tangents=[] if v is None else [v.copy()];tops=[];precision=[]
        started=time.monotonic();failed=None;received={}
        try:
            for i in range(steps):
                self.counts.check()
                if not differentiated:self.counts.claim_kind('ordinary')
                self.counts.calls+=1;self.observations.clear()
                with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                    if differentiated:
                        with fw.dual_level():
                            # Explicit float64 scalars: default torch scalar dtype is float32.
                            t=fw.make_dual(torch.tensor(float(logtemp),dtype=torch.float64),torch.tensor(float(dlogtemp),dtype=torch.float64))
                            self.loop.temperature=torch.pow(10.,t)
                            y,pr=self.loop.forward(fw.make_dual(torch.tensor(x),torch.tensor(v)),cached=cached)
                            uy=fw.unpack_dual(y);y,j=uy.primal,uy.tangent;pr=fw.unpack_dual(pr).primal
                            if j is None:raise InstrumentError('lost mixed tangent')
                            v=j.detach().numpy().copy()
                    else:
                        self.loop.temperature=10.**float(logtemp);y,pr=self.loop.forward(torch.tensor(x),cached=cached)
                    received=dict(received_state=y.detach().numpy().copy(),received_probabilities=pr.detach().numpy().copy())
                    if differentiated:received['received_tangent']=v.copy()
                    if not fresh_precision(self.observations):raise InstrumentError('effective precision')
                    if not bool(torch.isfinite(pr).all()):raise InstrumentError('nonfinite probabilities')
                    values,ids=torch.topk(pr,5);tops.append(np.stack((ids.detach().numpy(),values.detach().numpy())))
                    precision.append(copy.deepcopy(self.observations))
                    x=finite(y.detach().numpy(),(DIM,));traces.append(x.copy())
                    if differentiated:tangents.append(finite(v,(DIM,)))
                self.counts.check()
        except BaseException as e:
            failed=dict(type=type(e).__name__,message=str(e))
            if received:self.rec.arrays(label+'-failed-return',**received)
            raise
        finally:
            self.loop.temperature=10.**float(logtemp)
            name=self.rec.arrays(label,states=np.array(traces),tangents=np.array(tangents),
                alpha=np.array(self.alpha),logtemp=np.array(logtemp),dlogtemp=np.array(dlogtemp),top5=np.array(tops))
            self.last_cycle_file=name
            self.rec.event('cycle',file=name,label=label,steps_completed=len(traces)-1,differentiated=differentiated,
                cached=cached,seconds=time.monotonic()-started,failed=failed,precision=precision)
        return np.array(traces),None if not differentiated else np.array(tangents)

class Scaled:
    def __init__(self,model,scale):
        if not np.isfinite(scale) or scale<=0:raise ValueError('positive scale')
        self.model=model;self.scale=float(scale)
    def unpack(self,y):return y[:-1]*self.scale,H+float(y[-1])*DELTA
    def pack(self,x,t):return np.r_[x/self.scale,(t-H)/DELTA]
    def residual(self,y):
        x,t=self.unpack(y);tr,_=self.model.cycle(x,t,label='residual')
        return (tr[-1]-x)/self.scale
    def action(self,y,d):
        x,t=self.unpack(y);tr,j=self.model.cycle(x,t,v=d[:-1]*self.scale,dlogtemp=d[-1]*DELTA,label='mixed-jvp')
        return (j[-1]-d[:-1]*self.scale)/self.scale
    def av(self,y,v,label='state-jvp'):
        x,t=self.unpack(y);_,j=self.model.cycle(x,t,v=v,label=label)
        return j[-1]

class Fold:
    def __init__(self,scaled,c,h=1e-3):
        self.s=scaled;self.c=finite(c,(DIM,));self.h=h
        if abs(np.linalg.norm(self.c)-1)>1e-10:raise ValueError('unit fixed normalization covector')
    def residual(self,z):
        y=z[:DIM+1];v=z[DIM+1:];x,t=self.s.unpack(y)
        tr,j=self.s.model.cycle(x,t,v=v,label='fold-residual')
        return np.r_[(tr[-1]-x)/self.s.scale,j[-1]-v,self.c@v-1.]
    def action(self,z,d):
        self.s.model.counts.claim_kind('augmented')
        y=z[:DIM+1];v=z[DIM+1:];dy=d[:DIM+1];dv=d[DIM+1:]
        model=self.s.model;input_file=None
        if hasattr(model,'rec'):
            input_file=model.rec.arrays('fold-action-input',z=z,d=d,augmented_action=np.array(model.counts.augmented))
        top=self.s.action(y,dy)
        norm=float(np.linalg.norm(dy))
        observation=dict(augmented_action=model.counts.augmented,h=float(self.h),norm=norm,norm_hex=norm.hex(),
            input_file=input_file,
            z_sha256=hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest(),
            d_sha256=hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest(),
            dy_sha256=hashlib.sha256(np.ascontiguousarray(dy).tobytes()).hexdigest(),
            y_sha256=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
            top_file=getattr(model,'last_cycle_file',None),plus_file=None,minus_file=None,
            vector_file=None,status='incomplete')
        if hasattr(model,'rec'):model.rec.event('fold_offset_reduction',**dict(observation,status='started'))
        try:
            if norm:
                try:a=self.s.av(y+self.h*dy/norm,v,'fold-fd-plus')
                finally:observation['plus_file']=getattr(model,'last_cycle_file',None)
                try:b=self.s.av(y-self.h*dy/norm,v,'fold-fd-minus')
                finally:observation['minus_file']=getattr(model,'last_cycle_file',None)
                mixed=(a-b)*(norm/(2*self.h))
            else:mixed=np.zeros(DIM)
            if np.linalg.norm(dv):
                try:mixed+=self.s.av(y,dv,'fold-vector')-dv
                finally:observation['vector_file']=getattr(model,'last_cycle_file',None)
            observation['status']='completed'
        finally:
            if hasattr(model,'rec'):model.rec.event('fold_offset_cycles',**observation)
        return np.r_[top,mixed,self.c@dv]
