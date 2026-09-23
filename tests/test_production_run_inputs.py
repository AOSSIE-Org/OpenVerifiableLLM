from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from copy import deepcopy
from decimal import Decimal
import pytest
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.production_parents import validate_parents,public_initialization
from ovl_pipeline.supervision import rental_plan
from test_production_parents import parents
from test_rental_controller import intent
import production_run_inputs as m


def fixture():
    r,p=parents();rental=intent();w=rental['watchdog_intent']
    w['plan']=rental_plan({**w['plan']['input'],'maximum_seconds':86400,'allowance_usd':'9'})
    q={'compatible_environment_sha256':r['runtime']['compatible_environment_sha256'],
       'pilot_records':p['pilot_records'],'pilot_replays':p['pilot_replays']}
    initial={'record':p['initial_record'],'verification':p['initial_verification']}
    return r,p,rental,q,initial


def public_parents(p):
    result=deepcopy(p)
    projected=public_initialization({'record':p['initial_record'],'verification':p['initial_verification']})
    result.update(initial_record=projected['record'],initial_verification=projected['verification'])
    return result


def test_registration_freezes_actual_measurements_with_original_whole_rental_ceiling():
    r,p,rental,q,initial=fixture();now=rental['watchdog_intent']['plan']['input']['now_epoch']+100
    actual,basis=m.registration(r,q,initial,rental,now)
    assert actual['forecast_input']['schema']=='ovl.cost-forecast-input.v4'
    assert validate_parents(actual,**public_parents(p))['result']=='PASS'
    assert actual['pilots']['wikipedia']['record_sha256']==digest(q['pilot_records']['wikipedia'])
    inp=rental['watchdog_intent']['plan']['input']
    ceiling=sum(Decimal(inp[k]) for k in ('spent_usd','outstanding_usd','reserved_remaining_usd'))+Decimal(basis['original_maximum_rental_micro_usd'])/10**6
    forecast=Decimal(basis['forecast']['projected_total_micro_usd'])/10**6
    assert ceiling<=forecast<=ceiling+Decimal('.000003')
    assert 'not claimed measured' in basis['scope']


def test_construction_exposure_is_reallocated_inside_original_ceiling_and_expires():
    from production_run_coordinator import Run
    from types import SimpleNamespace
    r,p,rental,q,initial=fixture();plan=rental['watchdog_intent']['plan'];now=plan['input']['now_epoch']+100
    plain,old_basis=m.registration(r,q,initial,rental,now)
    actual,basis=m.registration(r,q,initial,rental,now,construction_seconds=120)
    assert validate_parents(actual,**public_parents(p))['result']=='PASS'
    assert abs(basis['forecast']['projected_total_micro_usd']-old_basis['forecast']['projected_total_micro_usd'])<=3
    assert Decimal(actual['forecast_input']['fixed_remaining_usd'])<Decimal(plain['forecast_input']['fixed_remaining_usd'])
    owner=Run.__new__(Run);owner.plan=plan;owner.qualified=q
    owner.selection={'timing':{'record_seconds':16000,'replay_seconds':16000,
                              'record_fixed_seconds':300,'replay_fixed_seconds':300,
                              'publication_policy':{'boundary_seconds':1}}}
    owner.health=SimpleNamespace(now=lambda:now+1)
    with pytest.raises(EvidenceError,match='elapsed rental exposure'):owner.forecast_window(plain)
    assert owner.forecast_window(actual)['result']=='FITS_OPERATING_LIMIT'
    owner.health.now=lambda:now+121
    with pytest.raises(EvidenceError,match='elapsed rental exposure'):owner.forecast_window(actual)
    assert rental['watchdog_intent']['plan']==plan


@pytest.mark.parametrize('seconds', [-1, True, 1501, 1.5])
def test_construction_exposure_is_finite_and_typed(seconds):
    r,_,rental,q,initial=fixture()
    with pytest.raises(EvidenceError):
        m.registration(r,q,initial,rental,rental['watchdog_intent']['plan']['input']['now_epoch'],construction_seconds=seconds)


@pytest.mark.parametrize('damage',['expired','wrong-recipe','wrong-code','insufficient-remainder'])
def test_registration_cannot_relabel_another_configuration_or_exceed_original_allowance(damage):
    r,p,rental,q,initial=fixture();plan=rental['watchdog_intent']['plan'];now=plan['input']['now_epoch']+100
    if damage=='expired':now=plan['external_terminate_epoch']
    elif damage=='wrong-recipe':r['recipe']={**r['recipe'],'seed':456}
    elif damage=='wrong-code':r['code_root']='f'*64
    else:now=plan['request_checkpoint_epoch']-1
    with pytest.raises(EvidenceError):m.registration(r,q,initial,rental,now)


@pytest.mark.parametrize('kind',['production-record','full-replay'])
def test_production_template_binds_public_parents_and_separates_private_key(tmp_path,kind):
    from test_production_run_job import selected
    from types import SimpleNamespace
    import production_run_job
    _,r,packet,bundle,policy,source,_,_=selected(tmp_path,kind)
    config=tmp_path/'config.json';write_json(config,{'explicit-offline-config-fixture':True})
    pp=tmp_path/'policy.json';sp=tmp_path/'source-policy.json';write_json(pp,policy);write_json(sp,source)
    base='/selected';profile={'remote_root':base}
    static=[{'path':base+'/prepared/'+p+'/stream.json','bytes':1,'sha256':r['coverage'][p]['stream_sha256']} for p in r['coverage']]
    job=m.production_job(kind,profile,config,static,packet,bundle,pp,sp,record_files=[] if kind=='full-replay' else None)
    job['deadline_epoch']=1000;job['argv']=['1000' if a==m.DEADLINE else a for a in job['argv']]
    remote=base+('-production-record' if kind=='production-record' else '-production-replay')
    assert production_run_job.command(job,r,packet,bundle,policy,source,SimpleNamespace(profile={'remote_root':remote}),kind)['--output']==remote
    assert not any('seed.key' in f['path'] for f in job['required_files'])


def test_generated_record_and_replay_consume_the_actual_delivered_anchor_tree(prepared,tmp_path,monkeypatch):
    from test_production_boundary_poll import fixture as published
    from ovl_pipeline.production_record import await_anchor
    from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy,verify_prefix
    from ovl_pipeline.production_anchoring import object_at
    import time
    hook,remote,out,health,starts,provider=published(prepared,tmp_path,monkeypatch)
    publisher=hook();assert publisher.poll()['index']==0
    a={k:Path(v) for k,v in publisher.arguments.items()};r=object_at(a['packet'],'registration.json')
    config=tmp_path/'runtime-config.json';write_json(config,{'synthetic':True})
    def selected(kind):
        job=m.production_job(kind,{'remote_root':'/selected'},config,[],a['packet'],a['registration-bundle'],
            a['production-policy'],a['source-policy'],record_files=[] if kind=='full-replay' else None)
        args=job['argv'][job['argv'].index('--')+1:];return dict(zip(args[::2],args[1::2]))
    record=selected('production-record');replay=selected('full-replay')
    anchor=remote/Path(record['--anchor-directory']).relative_to(record['--output'])
    policies=remote/Path(record['--progress-policies']).relative_to(record['--output'])
    replay_anchor=remote/Path(replay['--progress-directory']).relative_to(replay['--chain-directory'])
    assert replay_anchor==anchor
    envs=read_json(remote/'chain.json')['boundaries']
    assert await_anchor(r,digest(r),envs,anchor,policies,int(time.time())+30)['result']=='PASS'
    assert verify_prefix(r,digest(r),envs,replay_anchor,[ProgressPublisherPolicy(**p) for p in read_json(policies)],complete=False)['result']=='PASS'
    # Counterfactual: the old generated directory never sees the delivered files.
    now=[1]
    def advance(seconds):now[0]+=seconds
    with pytest.raises(EvidenceError,match='deadline reached'):
        await_anchor(r,digest(r),envs,remote/'public-anchors',policies,2,wall=lambda:now[0],sleep=advance)


from test_pipeline import prepared
