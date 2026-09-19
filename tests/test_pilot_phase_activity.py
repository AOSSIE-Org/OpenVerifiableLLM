"""Bounded audited-phase liveness is never numerical replay or export evidence."""
import copy
import pytest
from test_workload_health import Clock,JOB,SELECTION,activity
from ovl_pipeline.canonical import EvidenceError,digest
from ovl_pipeline.supervision import Journal


def phase(count):
    return {'schema':'ovl.audited-pilot-phases.v1','process_instance':'e'*32,'pid':456,
            'completed':[{'phase':p,'report_sha256':digest(p)} for p in ('record','replay','resume')[:count]],
            'scope':'operator-supervision-only-not-training-verification'}


def test_completed_prefix_and_restart_never_turn_polling_into_progress_or_exports(tmp_path):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=c.health(j);h.start_job(SELECTION);initial=h.exported;c.advance(100)
        assert h.activity(JOB,phase(1));first=h.progress
        c.advance(60);assert not h.activity(JOB,phase(1));assert h.progress==first and h.exported==initial
    with Journal(path).lease() as j:
        h=c.health(j);assert not h.activity(JOB,phase(1));assert h.progress==first
        c.advance(50);assert h.activity(JOB,phase(3))  # observation may miss a completed intermediate phase
        final=h.progress;c.advance(30);assert not h.activity(JOB,phase(3));assert h.progress==final
        assert h.exported==initial and not h.complete and not h.jobs[JOB]['finished']


@pytest.mark.parametrize('damage',['empty','too-many','reordered','changed-prefix','regressed','pid','instance','bad-digest','duplicate-report','scope','unknown','extra'])
def test_changed_or_invented_phase_observation_never_renews_progress(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=c.health(j);h.start_job(SELECTION);h.activity(JOB,phase(2));prior=h.progress;c.advance(20)
        v=phase(3)
        if damage=='empty':v['completed']=[]
        elif damage=='too-many':v['completed'].append({'phase':'fourth','report_sha256':digest('fourth')})
        elif damage=='reordered':v['completed'].reverse()
        elif damage=='changed-prefix':v['completed'][0]['report_sha256']=digest('other')
        elif damage=='regressed':v=phase(1)
        elif damage=='pid':v['pid']+=1
        elif damage=='instance':v['process_instance']='f'*32
        elif damage=='bad-digest':v['completed'][-1]['report_sha256']='bad'
        elif damage=='duplicate-report':v['completed'][-1]['report_sha256']=v['completed'][0]['report_sha256']
        elif damage=='scope':v['scope']='verified-training'
        elif damage=='unknown':v['schema']='future'
        else:v['heartbeat']=True
        with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert h.progress==prior


@pytest.mark.parametrize('first',['numerical','phase','setup'])
def test_protocols_and_workload_kinds_cannot_be_mixed(tmp_path,first):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=c.health(j);h.start_job({**SELECTION,'kind':'setup' if first=='setup' else 'pilot'})
        if first=='numerical':h.activity(JOB,activity())
        elif first=='phase':h.activity(JOB,phase(1))
        with pytest.raises(EvidenceError):h.activity(JOB,activity() if first=='phase' else phase(2))


@pytest.mark.parametrize('observation_delay',[0,65])
def test_real_controller_still_stops_after_last_completed_phase_stalls(tmp_path,observation_delay):
    from datetime import datetime,timezone
    import workload_health as m
    from test_rental_controller import RentalFake,NOW
    from ovl_pipeline.supervision import rental_plan
    from ovl_pipeline.canonical import read_json
    f=RentalFake(tmp_path);f.healthy=False
    f.i['plan']=rental_plan({**f.i['plan']['input'],'maximum_seconds':1800,'checkpoint_grace_seconds':420})
    for payload in (f.i['payload'],f.value['payload']):
        payload['terminateAfter']=datetime.fromtimestamp(f.i['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    f.refresh()
    original=f.account
    with Journal(tmp_path/'health-journal').lease() as j:
        h=m.Health(j,f.i,'owned-pod',wall=lambda:f.now,clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
        h.start_job(SELECTION)
        def account():
            if f.alive:
                count=max(0,min(3,int((f.elapsed-observation_delay)//150)))
                if count:h.activity(JOB,phase(count))
                h.write(f.health)
            return original()
        f.account=account;f.run()
        # The real controller observes on its10second cadence.
        assert NOW+450+observation_delay <= h.progress < NOW+460+observation_delay
        assert h.exported==NOW and not h.complete
        stop=read_json(f.directory/'stop-request.json')
        assert h.progress+300 < stop['observed_epoch'] <= h.progress+330
        assert not f.alive and f.writes==1
        assert f.i['plan']['external_terminate_epoch']==NOW+1920
