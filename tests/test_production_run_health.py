"""Mixed-phase journal recovery; explicit telemetry fixtures give no GPU credit."""
from pathlib import Path
import sys
from copy import deepcopy
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from production_run_health import ProductionRunHealth
from test_workload_health import Clock,SELECTION,activity
from test_external_watchdog import intent,NOW
from test_pilot_health import validation,BINDING as PILOT
from test_initialization_health import scan,BINDING as INITIAL
from test_sustained_health import transfer,BINDING as DOWNLOAD
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json
from ovl_pipeline.supervision import Journal

P='a'*64;I='b'*64;D='d'*64


def make(j,c,bindings=None,downloads=None):
    return ProductionRunHealth(j,intent(),'owned-pod',None,
        bindings if bindings is not None else {P:dict(PILOT),I:dict(INITIAL)},
        downloads if downloads is not None else {D:dict(DOWNLOAD)},
        wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})


def close(h,job,root):
    root.mkdir();status={'schema':'ovl.workload-job-exit.v1','job_sha256':job,'state':'EXITED','exit_code':0}
    write_json(root/'exit.json',status)
    h.exported_files(job,root,inventory(root,['exit.json']));h.job_exit(job,status)


def test_download_pilot_initializer_share_original_lifetime_and_recover_all_contracts(tmp_path):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=make(j,c);original=digest(h.plan)
        h.start_job({**SELECTION,'job_sha256':D,'kind':'setup'});c.advance(30)
        assert h.activity(D,transfer(1024**2));assert h.exported==NOW
        close(h,D,tmp_path/'download');assert not h.complete
        h.start_job({**SELECTION,'job_sha256':P});c.advance(30)
        assert h.activity(P,validation(1,20));assert h.activity(P,activity(2))
        close(h,P,tmp_path/'pilot');assert not h.complete
    with Journal(path).lease() as j:
        h=make(j,c);assert digest(h.plan)==original and len(h.jobs)==2
        assert h.exported==NOW+60 and not h.complete
        h.start_job({**SELECTION,'job_sha256':I});c.advance(30)
        assert not h.activity(I,scan());assert h.progress==NOW+60
        assert h.activity(I,scan(2,20));close(h,I,tmp_path/'initialization')
        assert h.exported==NOW+90 and not h.complete
    with Journal(path).lease() as j:
        h=make(j,c);assert all(v['finished'] for v in h.jobs.values())
        assert h.validations[P]==validation(1,20) and h.validations[I]==validation(2,20)
        assert not h.complete and digest(h.plan)==original


@pytest.mark.parametrize('changed',['pilot','initializer','download','missing-pilot'])
def test_adoption_needs_every_original_finished_or_active_contract(tmp_path,changed):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=make(j,c)
        for job,kind in [(D,'setup'),(P,'pilot'),(I,'pilot')]:
            h.start_job({**SELECTION,'job_sha256':job,'kind':kind});close(h,job,tmp_path/job)
    b={P:dict(PILOT),I:dict(INITIAL)};d={D:dict(DOWNLOAD)}
    if changed=='pilot':b[P]['documents']+=1
    elif changed=='initializer':b[I]['action']='verify'
    elif changed=='download':d[D]['bytes']+=1
    else:del b[P]
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):make(j,c,b,d)


@pytest.mark.parametrize('job,wrong',[(P,scan()),(I,validation()),(D,validation())])
def test_one_phase_cannot_borrow_another_activity_protocol(tmp_path,job,wrong):
    c=Clock()
    with Journal(tmp_path/'journal').lease() as j:
        h=make(j,c);h.start_job({**SELECTION,'job_sha256':job,'kind':'setup' if job==D else 'pilot'})
        before=len(j.events)
        with pytest.raises(EvidenceError):h.activity(job,deepcopy(wrong))
        assert len(j.events)==before and h.exported==NOW and not h.complete


def test_fake_initializer_event_cannot_be_adopted_as_pilot(tmp_path):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=make(j,c);h.start_job({**SELECTION,'job_sha256':P})
        j.append('decision',{'schema':'ovl.cost-activity-event.v2','kind':'initialization-scan',
            'observed_epoch':NOW,'detail':{'job_sha256':P,'observation':scan(1,20)},
            'advances_progress':True,'advances_export':False,'completes':False})
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):make(j,c)


def test_production_cannot_use_missing_registration_or_pilot_contract(tmp_path):
    c=Clock()
    with Journal(tmp_path/'journal').lease() as j:
        h=make(j,c,bindings={'e'*64:{'job_file':'untrusted'}},downloads={})
        before=len(j.events)
        with pytest.raises(EvidenceError):h.start_job({**SELECTION,'job_sha256':'e'*64,'kind':'production-record'})
        assert len(j.events)==before and not h.jobs


from test_pipeline import prepared


def test_real_production_retention_adopts_after_completed_setup_without_new_clock(prepared,tmp_path):
    from test_production_stage import configured,hooks,execute
    from test_workload_stage import intent as actual_intent
    data=configured(prepared,tmp_path);control,t,job,root,worker,r,bindings,out=data
    w=actual_intent();path=tmp_path/'run-health'
    with Journal(path).lease() as j:
        h=ProductionRunHealth(j,w,control.profile['pod_id'],r,bindings,{})
        h.start_job({**SELECTION,'pod_id':h.pod,'job_sha256':D,'kind':'setup'})
        close(h,D,tmp_path/'setup-retained');checkpoint_before=h.exported
        assert not h.complete
    with Journal(path).lease() as j:
        h=ProductionRunHealth(j,w,control.profile['pod_id'],r,bindings,{})
        assert h.exported==checkpoint_before
        cp,pub=hooks(tmp_path,t,root,r,h);result=execute(tmp_path,data,h,cp,pub)
        assert result['terminal']['exit_code']==0 and len(h.jobs)==2
        assert all(v['finished'] for v in h.jobs.values()) and not h.complete
    with Journal(path).lease() as j:
        h=ProductionRunHealth(j,w,control.profile['pod_id'],r,bindings,{})
        assert h.retained(root)[1]['terminal']==result['terminal'] and not h.complete


def registration_health(j,c,registration=None):
    return ProductionRunHealth(j,intent(),'owned-pod',registration or {'explicit-test-registration':True},{},{},
        wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})


def publication(h,stage='actions-run-observed',deadline=NOW+200):
    return {'schema':'ovl.publication-activity.v1','registration_sha256':h.registration_root,
        'boundary_sha256':h.registration_root,'stage':stage,'identity':{'run_id':123},'identity_sha256':digest({'run_id':123}),
        'deadline_epoch':deadline,'scope':'operator-publication-liveness-only-not-training-or-signer-verification'}


@pytest.mark.parametrize('stage',['actions-run-observed','checkpoint-privacy-review-verified','anchor-privacy-review-verified'])
def test_registration_progress_is_finite_and_replayed_without_export_credit(tmp_path,stage):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=registration_health(j,c);v=publication(h,stage);c.advance(30)
        assert h.registration_activity(v);assert h.progress==NOW+30 and h.exported==NOW and not h.complete
        c.advance(30);assert not h.registration_activity(v);assert h.progress==NOW+30
    with Journal(path).lease() as j:
        h=registration_health(j,c);assert not h.registration_activity(v);assert h.progress==NOW+30
        assert h.registration_activity(publication(h,'actions-step-01'));assert h.exported==NOW
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):registration_health(j,c,{'different-registration':True})


@pytest.mark.parametrize('damage',['deadline','expired','changed-root','overlap','changed-stage-identity','unknown-stage'])
def test_invalid_registration_progress_never_receives_credit(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'journal').lease() as j:
        h=registration_health(j,c);v=publication(h);assert h.registration_activity(v)
        if damage=='deadline':v=publication(h,'actions-step-01',NOW+201)
        elif damage=='expired':c.advance(201)
        elif damage=='changed-root':v['registration_sha256']='0'*64
        elif damage=='overlap':h.start_job({**SELECTION,'job_sha256':D,'kind':'setup'})
        elif damage=='changed-stage-identity':v['identity']={'run_id':999};v['identity_sha256']=digest(v['identity'])
        else:v['stage']='arbitrary-clock-tick'
        before=len(j.events)
        with pytest.raises(EvidenceError):h.registration_activity(v)
        assert len(j.events)==before and h.exported==NOW and not h.complete
