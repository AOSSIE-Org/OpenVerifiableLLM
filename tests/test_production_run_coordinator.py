"""One-rental orchestration with actual toy processes, explicit qualification doubles.

Provider/signature fixtures are test substitutes, never production admission.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
import time
from types import SimpleNamespace
import pytest
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,write_json
from production_run_coordinator import Run,restore_phase_bindings
import production_run_coordinator as m
from test_sustained_pilot_dispatch import fixture as development_fixture
from sustained_pilot_selection import DEADLINE


def configured(tmp_path,monkeypatch):
    _,plan,rental,control,remote,calls,controller,watchdog=development_fixture(tmp_path)
    plan={**plan,'schema':'ovl.sustained-pilot-plan.v2','prior_jobs':[],
          'completion_scope':'phase-only-under-production-run-coordinator'}
    # Explicit larger simulated rental permits the synthetic report's full
    # forecast; these are local control tests and allocate no provider resource.
    from ovl_pipeline.supervision import rental_plan,Journal
    from datetime import datetime,timezone
    w=rental['watchdog_intent'];w['plan']=rental_plan({**w['plan']['input'],'maximum_seconds':86400,'allowance_usd':'9'})
    stamp=datetime.fromtimestamp(w['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    w['payload']['terminateAfter']=rental['payload']['terminateAfter']=stamp
    controller=tmp_path/'full-controller'
    with Journal(controller).lease() as j:
        j.append('creation-intent',rental);j.append('creation-observed',{'id':control.profile['pod_id']})
    observed=read_json(watchdog);observed.update(intent_sha256=digest(w),plan_sha256=digest(w['plan']),external_terminate_epoch=w['plan']['external_terminate_epoch'])
    write_json(watchdog,observed)
    plan['rental_intent_sha256']=digest(rental)
    initial={**plan,'schema':'ovl.initialization-cycle-plan.v2'}
    selection={'object_store':str((tmp_path/'objects').resolve()),'schema':'ovl.production-run-selection.v1','rental_intent_sha256':digest(rental),
               'profile_sha256':digest(control.profile),'worker_sha256':file_hash(Path('scripts/pod_job_worker.py')),
               'phases':{'qualification':digest(plan),'initialization':digest(initial)},
               'timing':{'registration_seconds':120,'record_seconds':16000,'replay_seconds':16000,'export_seconds':60,
                         'record_fixed_seconds':300,'replay_fixed_seconds':300,
                         'checkpoint_policy':{},'publication_policy':{'boundary_seconds':1}}}
    def run():return Run(selection,digest(selection),rental,control,Path('scripts/pod_job_worker.py'),controller,
                         watchdog,tmp_path/'coordinator',tmp_path/'health.json',{}, {},sleep=lambda _:time.sleep(.01))
    # The setup/export process is actual. This control-flow fixture deliberately
    # does not claim that one toy setup job is a CUDA qualification cycle.
    monkeypatch.setattr(m.parents,'qualification',lambda *a:{'explicit-test-qualification-double':True})
    return run,plan,initial,selection,control,remote,calls


def test_actual_phase_is_retained_adopted_and_cannot_finish_rental(tmp_path,monkeypatch):
    factory,plan,initial,selection,control,remote,calls=configured(tmp_path,monkeypatch)
    with factory() as run:
        before=run.health.plan.copy();run.phase('qualification',plan,digest(plan),tmp_path,tmp_path/'qualification')
        assert not run.health.complete and len(run.health.jobs)==1
        n=len(calls);run.phase('qualification',plan,digest(plan),tmp_path,tmp_path/'qualification')
        assert len(calls)==n and run.health.plan==before
        with pytest.raises(EvidenceError,match='both complete'):run.finish()
    with factory() as run:
        assert not run.health.complete
        n=len(calls);run.phase('qualification',plan,digest(plan),tmp_path,tmp_path/'qualification')
        assert len(calls)==n


def test_failed_admission_after_real_phase_closes_paid_work_with_real_exit_exports(tmp_path,monkeypatch):
    factory,plan,_,_,_,_,calls=configured(tmp_path,monkeypatch)
    with pytest.raises(EvidenceError,match='injected admission failure'):
        with factory() as run:
            run.phase('qualification',plan,digest(plan),tmp_path,tmp_path/'qualification')
            raise EvidenceError('injected admission failure')
    assert read_json(tmp_path/'health.json')['complete'] is True
    final=tmp_path/'coordinator/failure'
    assert len(list(final.glob('*/exit.json')))==1
    assert read_json(final/'reason.json')['exception_class']=='EvidenceError'


def test_missing_public_registration_cannot_select_a_production_job(tmp_path,monkeypatch):
    factory,*_=configured(tmp_path,monkeypatch)
    with factory() as run:
        with pytest.raises(EvidenceError,match='authenticated public registration'):
            run.select_job({'kind':'production-record'},tmp_path/'production')
        with pytest.raises(EvidenceError,match='authenticated public registration'):
            run.stage('production-record',tmp_path/'absent','a'*64,[],None,tmp_path/'production',tmp_path/'objects')


def test_original_job_deadline_survives_restart_and_late_remaining_budget_refuses(tmp_path,monkeypatch):
    factory,plan,initial,selection,control,remote,calls=configured(tmp_path,monkeypatch)
    template=read_json(tmp_path/plan['stages'][0]['template_path'])
    template={**template,'kind':'production-record','deadline_epoch':DEADLINE}
    with factory() as run:
        run.authenticated={'explicit-test-admission-double':True};run.health.registration_root='a'*64
        path,root=run.select_job(template,tmp_path/'production')
        fixed=read_json(path)['deadline_epoch'];original=read_json(tmp_path/'production/job-selection.json')
        run.health.now=lambda:fixed+1
        assert run.select_job(template,tmp_path/'production')==(path,root)
        assert read_json(tmp_path/'production/job-selection.json')==original
        run.health.now=lambda:run.plan['request_checkpoint_epoch']-5
        observed=read_json(tmp_path/'watchdog.json');observed['observed_epoch']=run.health.now()
        write_json(tmp_path/'watchdog.json',observed)
        with pytest.raises(EvidenceError,match='complete record and full replay'):
            run.select_job(template,tmp_path/'another-production')
        assert not (tmp_path/'another-production/job-selection.json').exists()


def test_original_provider_identity_and_deadline_guards_remain_binding(tmp_path,monkeypatch):
    factory,plan,_,_,control,remote,calls=configured(tmp_path,monkeypatch)
    value=read_json(tmp_path/'watchdog.json');value['intent_sha256']='f'*64;write_json(tmp_path/'watchdog.json',value)
    with pytest.raises(EvidenceError,match='watchdog'):
        with factory():pass
    assert not calls


def test_restore_rejects_changed_derivation_before_journal_adoption(tmp_path,monkeypatch):
    factory,plan,*_=configured(tmp_path,monkeypatch)
    with factory() as run:run.phase('qualification',plan,digest(plan),tmp_path,tmp_path/'qualification')
    output=tmp_path/'qualification';restore_phase_bindings(plan,digest(plan),output,{}, {})
    path=output/'derived/first/selection.json';selected=read_json(path);selected['deadline_epoch']+=1;write_json(path,selected)
    with pytest.raises(EvidenceError,match='derivation'):
        restore_phase_bindings(plan,digest(plan),output,{}, {})


def registration_fixture(tmp_path,monkeypatch):
    from test_registration_dispatch import configured as signing
    from ovl_pipeline.production_anchoring import packet_objects
    source=tmp_path/'source-fixture';source.mkdir()
    packet,r,policy,provider,calls=signing(source,monkeypatch)
    r['forecast_input']['committed_future_usd']='61'
    write_json(packet/'registration.json',r)
    _,objects=packet_objects(packet)
    original=m.registration_publisher.request_commit
    def commit(request,registration,directory,**kw):
        revision=original(request,registration,directory,**kw)
        write_json(directory/'public-commit.json',{'revision':revision});return revision
    monkeypatch.setattr(m.registration_publisher,'request_commit',commit)
    return packet,r,policy,provider,objects


def test_registration_uses_actual_downloads_then_adopts_without_republication(tmp_path,monkeypatch):
    factory,plan,*_=configured(tmp_path,monkeypatch)
    packet,r,policy,provider,p=registration_fixture(tmp_path,monkeypatch)
    with factory() as run:
        run.phase('qualification',plan,digest(plan),tmp_path,tmp_path/'qualification')
        run.qualified={'pilot_records':p['pilot_records'],'pilot_replays':p['pilot_replays']}
        run.initial={'record':p['initial_record'],'verification':p['initial_verification']}
        args=(r,p['source'],packet/'source-statement.sigstore.json',policy,p['prepared'],tmp_path)
        first=run.register(*args)
        assert first['check']['result']=='PASS' and provider.commits==2
        original=read_json(run.output/'registration-deadline.json')
        assert run.register(*args)==first and provider.commits==2
        assert read_json(run.output/'registration-deadline.json')==original
        # A signature alone cannot replace the retained actual download bytes.
        first['bundle'].chmod(0o600);first['bundle'].write_bytes(b'altered public bundle')
        with pytest.raises(EvidenceError):run.register(*args)


@pytest.mark.parametrize('existing_export',[False,True])
def test_abort_retains_actual_terminal_production_bytes_and_partial_states(prepared,tmp_path,existing_export,monkeypatch):
    from test_production_health import bound
    from test_workload_stage import intent
    from ovl_pipeline.supervision import Journal
    from production_run_health import ProductionRunHealth
    import shutil
    control,transport,job,root,worker,r,bindings=bound(prepared,tmp_path)
    stage=tmp_path/'stage';stage.mkdir();shutil.copytree(tmp_path/'launch',stage/'launch')
    with Journal(tmp_path/'health-journal').lease() as journal:
        run=Run.__new__(Run);run.root='d'*64;run.control=control;run.output=tmp_path/'coordinator';run.output.mkdir()
        run.selection={'worker_sha256':worker,'timing':{'export_seconds':60}}
        run.health=ProductionRunHealth(journal,intent(),control.profile['pod_id'],r,bindings,{})
        run.health.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':control.profile['pod_id'],'kind':'production-record'})
        run.plan=run.health.plan;run.health_file=tmp_path/'health.json';run.sleep=lambda _:time.sleep(.01)
        run.active_stage=(bindings[root],job,root,stage,tmp_path/'objects')
        if existing_export:
            import production_retention
            original=production_retention.retain;limits=[];start=run.health.now()
            selection={'job_sha256':root};write_json(stage/'selection.json',selection)
            write_json(stage/'terminal-export-intent.json',{'schema':'ovl.production-stage-export.v1',
                'selection_sha256':digest(selection),'terminal':{'synthetic':'selected terminal'},
                'started_epoch':start,'deadline_epoch':start+60})
            def observed(*args,**kw):limits.append(args[7]);return original(*args,**kw)
            monkeypatch.setattr(production_retention,'retain',observed)
        run.abort('InjectedFixtureFailure')
        if existing_export:assert limits==[start+60]
        assert run.health.complete and read_json(run.health_file)['complete']
        proof=read_json(run.output/'failure/terminal/retention.json')
        receipt=read_json(Path(proof['roots'][1]['receipt_path']))
        assert (Path(receipt['files_directory'])/'recovery-partial/state.safetensors').read_bytes()==b'explicit incomplete sole recovery'


@pytest.mark.parametrize('damage',['rate','prior-exposure','numerical-time','publication-time'])
def test_forecast_cannot_omit_cost_or_complete_phase_windows(tmp_path,monkeypatch,damage):
    factory,plan,*_=configured(tmp_path,monkeypatch)
    packet,r,policy,provider,p=registration_fixture(tmp_path,monkeypatch)
    with factory() as run:
        run.qualified={k:p[k] for k in ('pilot_records','pilot_replays')}
        if damage=='rate':r['forecast_input']['hourly_usd']='0.01'
        elif damage=='prior-exposure':r['forecast_input']['committed_future_usd']='0'
        elif damage=='numerical-time':run.selection['timing']['replay_seconds']=1
        else:run.selection['timing']['publication_policy']['boundary_seconds']=1500
        with pytest.raises(EvidenceError):run.forecast_window(r)
    assert provider.commits==0

from test_pipeline import prepared


def test_journaled_stop_forbids_new_work_but_allows_existing_job_shutdown_adoption(tmp_path,monkeypatch):
    factory,plan,_,_,control,remote,calls=configured(tmp_path,monkeypatch)
    from ovl_pipeline.supervision import Journal
    with factory() as run:
        with Journal(run.controller).lease() as journal:journal.append('decision',{'action':'CHECKPOINT_AND_STOP'})
        write_json(run.controller/'stop-request.json',{'explicit-test-controller-stop':True})
        with pytest.raises(EvidenceError,match='requests stop'):run.guards()
        run.guards(starting=False)
        assert not calls and not run.health.complete
