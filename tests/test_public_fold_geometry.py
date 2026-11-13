import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import public_fold_map as fm
import run_public_fold_geometry as runner
import public_pseudoarclength as palc

class ToyCounts:
    def __init__(self):self.augmented=0
    def claim_kind(self,k):setattr(self,k,getattr(self,k,0)+1)

class ToyScaled:
    scale=2.
    def __init__(self):self.model=self;self.counts=ToyCounts();self.actions=0
    def unpack(self,y):return y[:-1]*self.scale,float(y[-1])
    def cycle(self,x,t,*,v,label):
        u=x/self.scale
        residual=-.5*u;residual[0]=u[0]**2+t
        deriv=.5*v;deriv[0]=(1+2*u[0])*v[0]
        return np.stack((x,x+self.scale*residual)),np.stack((v,deriv))
    def action(self,y,d):
        self.actions+=1
        out=-.5*d[:-1];out[0]=2*y[0]*d[0]+d[-1];return out
    def av(self,y,v,label='state'):
        self.actions+=1
        out=.5*v;out[0]=(1+2*y[0])*v[0];return out

def test_extended_action_matches_analytic_and_counts_all_subactions():
    s=ToyScaled();c=np.zeros(896);c[0]=1
    f=fm.Fold(s,c);rng=np.random.default_rng(83)
    z=rng.normal(size=1793)*.01;z[897]=1
    d=rng.normal(size=1793);d/=np.linalg.norm(d)
    expected=(f.residual(z+1e-5*d)-f.residual(z-1e-5*d))/(2e-5)
    got=f.action(z,d)
    np.testing.assert_allclose(got,expected,rtol=1e-7,atol=2e-10)
    assert s.counts.augmented==1 and s.actions==4

def test_extended_solver_full_blocks_and_literal_result_contract():
    # Small analogue of orbit/null/fixed normalization, same production helper API.
    def residual(z):
        x,t,v=z;return np.array([x*x+t,2*x*v,v-1])
    def action(z,d):
        x,t,v=z;dx,dt,dv=d;return np.array([2*x*dx+dt,2*v*dx+2*x*dv,dv])
    result=palc.solve_system(residual,action,np.array([.02,-.001,.9]),
        config=palc.NewtonConfig(residual_tolerance=1e-10),
        residual_blocks={'orbit':(0,1,1e-10),'null':(1,2,1e-10),'normalization':(2,3,1e-10)})
    assert result['status']=='converged'
    np.testing.assert_allclose(result['y'],[0,0,1],atol=1e-9)
    json.dumps(palc.serializable(result),allow_nan=False)
    bad=palc.solve_system(residual,action,np.array([0.,0.,0.]),
        config=palc.NewtonConfig(max_newton_iterations=0),
        residual_blocks={'orbit':(0,1,1e-10),'null':(1,2,1e-10),'normalization':(2,3,1e-10)})
    assert bad['status']=='unresolved' and not bad['gates']['passed']

def test_all_six_closure_and_primitive_gates():
    trace=np.zeros((13,896));trace[:,0]=np.arange(13)%6
    assert runner.cycle_gate(trace)['passed']
    trace[9,1]=2e-10
    assert not runner.cycle_gate(trace)['passed']
    trace[:]=0
    assert not runner.cycle_gate(trace)['passed']

def test_alignment_rolls_cycle_not_embedding_coordinates():
    cycle=np.arange(6*896,dtype=float).reshape(6,896)
    ref=np.roll(cycle,-4,axis=0)
    out,row=runner.align_cycle(cycle,ref)
    assert row['shift']==4
    np.testing.assert_array_equal(out,ref)

def test_checkpoint_contract_and_event_bytes(tmp_path):
    r=fm.Record(tmp_path/'run');r.arrays('a',states=np.ones((7,896)))
    r.event('finite',test=2)
    r.checkpoint('pilot',{'status':'passed'})
    q=json.loads((r.directory/'checkpoint-0001.json').read_text())
    assert q['profile']=='public-e01' and q['closed'] and q['checkpoint_id']==1
    assert len(q['artifacts'])==1
    assert r.bytes==sum(p.stat().st_size for p in r.directory.iterdir() if p.is_file())
    with pytest.raises(FileExistsError):r.json('checkpoint-0001.json',{})
    r.stream.close()

def test_counter_failure_preserves_category(tmp_path):
    r=fm.Record(tmp_path/'run');c=fm.Counts(r);c.jvp_cycles=fm.CAPS['jvp_cycles']
    with pytest.raises(fm.LimitError,match='jvp_cycles'):c.claim_kind('jvp_cycles')
    assert c.jvp_cycles==fm.CAPS['jvp_cycles']
    r.stream.close()

def test_temperature_scaling_includes_delta_over_r():
    class Model:
        def cycle(self,x,t,*,v=None,dlogtemp=0.,label=None):
            return np.stack((x,.3*x+t)),None if v is None else np.stack((v,.3*v+dlogtemp))
    s=fm.Scaled(Model(),.0027);y=np.r_[np.ones(896),.2];d=np.r_[np.ones(896)*.4,.8]
    got=s.action(y,d)
    np.testing.assert_allclose(got,-.7*d[:-1]+fm.DELTA*.8/.0027,atol=1e-13)

def test_continuation_turn_file_contract_and_state_interpolation():
    result=palc.continue_branch(lambda y:np.array([y[0]**2+y[1]]),
        lambda y,d:np.array([2*y[0]*d[0]+d[1]]),np.array([-.2,-.04]),np.array([0.,1.]),
        config=palc.PACConfig(step_sizes=(.075,.0375),max_accepted_steps=6),stop_on_turn=True)
    estimate=runner.turn_estimate(result)
    assert abs(estimate['log10_temperature']-fm.H)<1e-7
    assert abs(estimate['scaled_state_estimate'][0])<1e-4
    assert len(estimate['point_indices'])==2
    json.dumps(estimate,allow_nan=False)


def context_fixture(path):
    """Small synthetic acceptance contract, never a production approval."""
    path.mkdir();pterm=np.zeros((2,896));pterm[1,0]=2.
    np.savez(path/'inputs.npz',fold_states=np.zeros((2,896)),fold_alphas=np.array([.07,.08]),
        fold_logtemps=np.array([-.16,-.15]),p_terminal=pterm,p_cycles=np.repeat(pterm[None],6,axis=0),
        h_cycles=np.zeros((2,6,896)),state_scale=np.array(2.))
    def save(name,data):
        target=path/name;target.write_text(json.dumps(data,indent=2)+'\n');return runner._binding(target)
    packet=save('inputs.json',dict(arrays=runner._binding(path/'inputs.npz')))
    counts=dict(ordinary=139,jvp_cycles=83,augmented=16,model_invocations=637,
                elapsed_seconds=2234.5,peak_rss_bytes=10000000)
    result=save('prior-result.json',dict(profile='public-e01',status='completed',outcome='primary_fold_unresolved',
        counts=counts,caps=fm.CAPS,source_commit='a'*40))
    source=save('prior-source.json',dict(commit='a'*40))
    start_text='2026-10-30T03:33:14Z';(path/'case-started-utc.txt').write_text(start_text+'\n')
    start=datetime.fromisoformat(start_text.replace('Z','+00:00')).timestamp()
    audit=save('prior-audit.json',dict(profile='public-e01',status='audit_passed',integrity_status='passed',
        synthetic=False,scientific_status='completed',outcome='primary_fold_unresolved',pulse_eligible=False,
        pulse_admission=dict(eligible=False,reasons=['Synthetic finite nonconvergence fixture.']),
        return_status='manifest_verified',closed_verified_return=True,source_commit='a'*40,counts=counts,
        result=result,source=source,started_utc=runner._binding(path/'case-started-utc.txt'),
        input_packet=packet,input_arrays=runner._binding(path/'inputs.npz'),
        campaign_started_utc=start_text,campaign_deadline_unix=start+72000,raw_inventory=dict(raw_bytes=1000),
        fixture_scope='Approval/closure flags explicitly simulated for a model-free contract test.'))
    usage=dict(ordinary=139,jvp_cycles=83,augmented=16,model_invocations=637,
               raw_bytes=1000,package_bytes=2000,archive_bytes=1500,transfer_bytes=1700)
    limits={k:fm.CAPS[k] for k in ('ordinary','jvp_cycles','augmented','raw_bytes')}
    limits.update(package_bytes=3*2**30,archive_bytes=3*2**30,transfer_bytes=12*2**30)
    context=save('case-context.json',dict(profile='public-e01-case-context',amendment=runner.AMENDMENT,
        attempt_index=1,prior_audit=audit,prior_result=result,prior_source=source,
        started_utc=runner._binding(path/'case-started-utc.txt'),input_packet=packet,input_arrays=runner._binding(path/'inputs.npz'),
        prior_counts=counts,case_limits=limits,prior_usage=usage,remaining_limits={k:v-usage[k] for k,v in limits.items()}))
    args=runner.parser().parse_args(['--input-packet',str(path/'inputs.json'),'--input-packet-sha256',packet['sha256'],
        '--vectors',str(path/'no-model-vectors'),'--cache-dir',str(path/'no-model-cache'),'--output-dir',str(path/'run'),
        '--case-context',context['path'],'--case-context-sha256',context['sha256'],'--preflight-only'])
    return args,start


def reseal_context(args,mutate):
    value=json.loads(args.case_context.read_text());mutate(value)
    args.case_context.write_text(json.dumps(value,indent=2)+'\n')
    args.case_context_sha256=runner._binding(args.case_context)['sha256']


def test_context_actual_file_contract_preflight_without_model(tmp_path,monkeypatch):
    args,start=context_fixture(tmp_path/'case');monkeypatch.setattr(fm.time,'time',lambda:start+3000)
    monkeypatch.setattr(fm,'Map',lambda *a:pytest.fail('preflight loaded a model'))
    info=runner.case_context(args);assert info['prior_counts']['jvp_cycles']==83
    assert info['geometry_deadline_unix']==start+43200 and info['campaign_deadline_unix']==start+72000
    assert runner.run(args)==0
    result=json.loads((args.output_dir/'result.json').read_text())
    assert result['status']=='preflight_passed' and result['attempt_index']==1
    assert result['counts']['model_invocations']==0 and result['case_counts']['model_invocations']==637
    assert result['fold_equation_order']==['orbit','normalization','null']
    assert result['case_context']==runner._binding(args.case_context)
    assert result['raw_limit_bytes']==fm.CAPS['raw_bytes']-1000
    manifest=json.loads((args.output_dir/'manifest.json').read_text())
    assert 'qualified-case-context.json' in {x['path'] for x in manifest['files']}
    assert all(runner._binding(args.output_dir/x['path'])['sha256']==x['sha256'] for x in manifest['files'])


@pytest.mark.parametrize('field,value',[
    ('amendment','other'),('attempt_index',2),('prior_counts',{}),
    ('source_commit','b'*40),('geometry_deadline_unix',1e12)])
def test_context_rejects_conflicting_case_identity(tmp_path,field,value):
    args,start=context_fixture(tmp_path/'case');reseal_context(args,lambda d:d.update({field:value}))
    with pytest.raises(ValueError):runner.case_context(args,now=start+3000)


@pytest.mark.parametrize('key',['prior_audit','prior_result','prior_source','started_utc','input_packet','input_arrays'])
def test_context_rejects_mutated_bound_file(tmp_path,key):
    args,start=context_fixture(tmp_path/'case');doc=json.loads(args.case_context.read_text())
    with Path(doc[key]['path']).open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError,match='binding differs'):runner.case_context(args,now=start+3000)


@pytest.mark.parametrize('field,value',[('synthetic',True),('status','audit_failed'),('pulse_eligible',True),
    ('scientific_status','incomplete'),('outcome','geometry_qualified_return_for_audit'),('closed_verified_return',False)])
def test_context_refuses_unaccepted_or_ineligible_predecessor(tmp_path,field,value):
    args,start=context_fixture(tmp_path/'case');doc=json.loads(args.case_context.read_text());p=Path(doc['prior_audit']['path'])
    a=json.loads(p.read_text());a[field]=value;p.write_text(json.dumps(a))
    reseal_context(args,lambda d:d.update(prior_audit=runner._binding(p)))
    with pytest.raises(ValueError,match='prior audit'):runner.case_context(args,now=start+3000)


@pytest.mark.parametrize('key',['ordinary','jvp_cycles','augmented','raw_bytes','package_bytes','archive_bytes','transfer_bytes'])
def test_context_cannot_renew_any_case_envelope(tmp_path,key):
    args,start=context_fixture(tmp_path/'case')
    reseal_context(args,lambda d:d['remaining_limits'].__setitem__(key,d['remaining_limits'][key]+1))
    with pytest.raises(ValueError,match='limit'):runner.case_context(args,now=start+3000)


def test_context_original_deadline_and_paired_flags(tmp_path):
    args,start=context_fixture(tmp_path/'case')
    with pytest.raises(TimeoutError,match='deadline'):runner.case_context(args,now=start+43200)
    args.case_context_sha256=None
    with pytest.raises(ValueError,match='together'):runner.case_context(args,now=start+3000)
    args.case_context=None;assert runner.case_context(args) is None


def test_cumulative_counters_deadline_and_raw_reserve(tmp_path,monkeypatch):
    rec=fm.Record(tmp_path/'run');monkeypatch.setattr(fm.time,'time',lambda:100.)
    prior={k:fm.CAPS[k]-1 for k in ('ordinary','jvp_cycles','augmented')}
    counts=fm.Counts(rec,prior=prior,geometry_deadline_unix=101.,campaign_deadline_unix=200.)
    for k in prior:
        counts.claim_kind(k)
        with pytest.raises(fm.LimitError,match='cumulative'):counts.claim_kind(k)
        assert counts.case_report()[k]==fm.CAPS[k] and counts.report()[k]==1
    monkeypatch.setattr(fm.time,'time',lambda:101.)
    with pytest.raises(TimeoutError,match='original geometry'):counts.check()
    rec.raw_limit=16*2**20+100
    with pytest.raises(fm.LimitError,match='raw'):rec.check(101)
    rec.finalizing=True;rec.check(101);rec.stream.close()


def test_orthogonal_solver_order_qualifies_same_full_extended_equations():
    s=ToyScaled();c=np.zeros(896);c[0]=1.;f=fm.Fold(s,c)
    rng=np.random.default_rng(83);z=rng.normal(size=1793)*.01;z[0]=.02;z[896]=-.001;z[897]=.9
    config=palc.NewtonConfig(residual_tolerance=1e-10)
    original=palc.solve_system(f.residual,f.action,z.copy(),config=config,
        residual_blocks={'orbit':(0,896,1e-10),'null':(896,1792,1e-10),'normalization':(1792,1793,1e-10)})
    aligned=palc.solve_system(lambda x:runner.fold_solver_output(f.residual(x)),
        lambda x,d:runner.fold_solver_output(f.action(x,d)),z.copy(),config=config,
        residual_blocks={'orbit':(0,896,1e-10),'normalization':(896,897,1e-10),'null':(897,1793,1e-10)})
    assert original['status']=='unresolved' and original['reason']=='gmres_not_converged'
    assert aligned['status']=='converged' and np.linalg.norm(f.residual(aligned['y']))<=1e-10
    for iteration in aligned['iterations']:
        assert iteration['linear']['gmres_info']==0 and iteration['linear']['relative_residual']<=.001
    before=f.residual(z);after=runner.fold_solver_output(before)
    np.testing.assert_array_equal(after[:896],before[:896]);np.testing.assert_array_equal(after[897:],before[896:1792])
    assert after[896]==before[-1] and np.linalg.norm(after)==pytest.approx(np.linalg.norm(before),rel=2e-15)
    assert aligned['config']['gmres_restart']==10 and aligned['config']['max_linear_actions']==12


@pytest.mark.parametrize('fail_label',[None,'fold-fd-plus','fold-fd-minus'])
def test_recorded_offset_norm_precedes_calls_and_retains_partial_links(fail_label):
    class Recorded(ToyScaled):
        def __init__(self):super().__init__();self.rec=self;self.events=[];self.last_cycle_file=None
        def event(self,name,**kw):self.events.append((name,kw))
        def arrays(self,label,**arrays):self.saved=arrays;return label+'.npz'
        def action(self,y,d):self.last_cycle_file='top.npz';return super().action(y,d)
        def av(self,y,v,label='state'):
            assert self.events[-1][0]=='fold_offset_reduction'
            self.last_cycle_file=label+'.npz'
            if label==fail_label:raise TimeoutError('synthetic signed callback interruption')
            return super().av(y,v,label)
    s=Recorded();c=np.zeros(896);c[0]=1.;f=fm.Fold(s,c);rng=np.random.default_rng(87)
    z=rng.normal(size=1793)*.01;d=rng.normal(size=1793);d/=np.linalg.norm(d)
    if fail_label:
        with pytest.raises(TimeoutError):f.action(z,d)
    else:f.action(z,d)
    before,after=s.events
    assert before[0]=='fold_offset_reduction' and after[0]=='fold_offset_cycles'
    b,a=before[1],after[1]
    assert b['status']=='started' and b['norm']==float(np.linalg.norm(d[:897]))
    assert b['input_file']=='fold-action-input.npz'
    np.testing.assert_array_equal(s.saved['z'],z);np.testing.assert_array_equal(s.saved['d'],d)
    assert float.fromhex(b['norm_hex'])==b['norm'] and b['h']==1e-3
    assert b['top_file']=='top.npz' and b['plus_file'] is None and b['minus_file'] is None
    assert a['plus_file']=='fold-fd-plus.npz'
    assert a['minus_file']==(None if fail_label=='fold-fd-plus' else 'fold-fd-minus.npz')
    assert a['status']==('incomplete' if fail_label else 'completed')
