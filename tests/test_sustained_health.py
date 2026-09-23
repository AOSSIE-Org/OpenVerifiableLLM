import pytest
from sustained_health import SustainedHealth
from test_workload_health import Clock,JOB,SELECTION,activity
from test_external_watchdog import intent,NOW
from ovl_pipeline.canonical import EvidenceError
from ovl_pipeline.supervision import Journal

BINDING={'schema':'ovl.public-input-binding.v1','plan_sha256':'c'*64,'bytes':3*1024**2}


def health(j,c,b=None):
    return SustainedHealth(j,intent(),'owned-pod',{},b or {JOB:dict(BINDING)},wall=lambda:c.now,
                           clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})


def transfer(n):
    return {'schema':'ovl.public-input-transfer.v1','process_instance':'b'*32,'pid':123,'plan_sha256':'c'*64,
            'total_bytes':BINDING['bytes'],'received_bytes':n,'scope':'operator-supervision-only-not-input-verification'}


def test_zero_and_repeated_bytes_do_not_refresh_progress_or_export_across_restart(tmp_path):
    c=Clock();path=tmp_path/'j'
    with Journal(path).lease() as j:
        h=health(j,c);h.start_job({**SELECTION,'kind':'setup'});c.advance(10)
        assert not h.activity(JOB,transfer(0));assert h.progress==NOW
        assert h.activity(JOB,transfer(1024**2));assert h.exported==NOW
    with Journal(path).lease() as j:
        h=health(j,c);assert not h.start_job({**SELECTION,'kind':'setup'});c.advance(100)
        assert not h.activity(JOB,transfer(1024**2));assert h.progress==NOW+10
        assert h.activity(JOB,transfer(3*1024**2));assert not h.complete and h.exported==NOW


@pytest.mark.parametrize('damage',['plan','total','over','negative','boolean','process','pid','scope','regressed','numeric'])
def test_unbounded_foreign_changed_or_restarted_download_is_rejected_without_append(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job({**SELECTION,'kind':'setup'});h.activity(JOB,transfer(1024**2));before=len(j.events)
        v=transfer(2*1024**2)
        if damage=='plan':v['plan_sha256']='d'*64
        elif damage=='total':v['total_bytes']+=1
        elif damage=='over':v['received_bytes']=4*1024**2
        elif damage=='negative':v['received_bytes']=-1
        elif damage=='boolean':v['received_bytes']=True
        elif damage=='process':v['process_instance']='e'*32
        elif damage=='pid':v['pid']+=1
        elif damage=='scope':v['scope']='verified-inputs'
        elif damage=='regressed':v['received_bytes']=0
        else:v=activity()
        with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert len(j.events)==before


def test_changed_download_contract_blocks_adoption(tmp_path):
    c=Clock();path=tmp_path/'j'
    with Journal(path).lease() as j:
        h=health(j,c);h.start_job({**SELECTION,'kind':'setup'})
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):health(j,c,{JOB:{**BINDING,'bytes':BINDING['bytes']+1}})


def setup_activity(completed=1,copied=0):
    from pod_runtime_setup import SETUP_PHASES
    return {'schema':'ovl.runtime-setup-activity.v1','process_instance':'e'*32,'pid':456,
            'completed':list(SETUP_PHASES[:completed]),'copied_bytes':copied,
            'scope':'operator-supervision-only-not-input-verification'}


@pytest.mark.parametrize('production_dispatch',[False,True])
def test_offline_progress_recovery_does_not_grant_export_or_extend_deadline(tmp_path,production_dispatch):
    c=Clock();path=tmp_path/'j'
    def selected_health(j):
        if not production_dispatch:return health(j,c)
        from production_run_health import ProductionRunHealth
        return ProductionRunHealth(j,intent(),'owned-pod',None,{}, {JOB:dict(BINDING)},wall=lambda:c.now,
            clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})
    with Journal(path).lease() as j:
        h=selected_health(j);h.start_job({**SELECTION,'kind':'setup'});h.activity(JOB,transfer(BINDING['bytes']))
        c.advance(40);assert h.activity(JOB,setup_activity());end=h.plan['external_terminate_epoch']
        c.advance(40);assert h.activity(JOB,setup_activity(copied=1024**2));credited=h.progress
        for k in range(1,5):
            c.advance(1);assert not h.activity(JOB,setup_activity(copied=1024**2+k))
        assert h.progress==credited and h.exported==NOW and not h.complete
    with Journal(path).lease() as j:
        h=selected_health(j);assert h.progress==credited and h.plan['external_terminate_epoch']==end
        c.advance(40);assert h.activity(JOB,setup_activity(2,BINDING['bytes']))
        same=h.progress;c.advance(301);assert not h.activity(JOB,setup_activity(2,BINDING['bytes']))
        assert h.progress==same and h.exported==NOW and not h.complete
        with pytest.raises(EvidenceError,match='download activity after'):h.activity(JOB,transfer(BINDING['bytes']))


@pytest.mark.parametrize('damage',['pid','process','reorder','regress','copy-regress','over','bool','premature-selection','postselection-copy','contract','scope'])
def test_invalid_setup_activity_does_not_change_retained_health(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job({**SELECTION,'kind':'setup'});h.activity(JOB,setup_activity(copied=1024**2))
        v=setup_activity(copied=2*1024**2)
        if damage=='pid':v['pid']+=1
        elif damage=='process':v['process_instance']='f'*32
        elif damage=='reorder':v['completed']=['selection']
        elif damage=='regress':v['completed']=[]
        elif damage=='copy-regress':v['copied_bytes']=0
        elif damage=='over':v['copied_bytes']=BINDING['bytes']+1
        elif damage=='bool':v['copied_bytes']=True
        elif damage=='premature-selection':v['completed']+=['selection']
        elif damage=='postselection-copy':
            h.activity(JOB,setup_activity(2,BINDING['bytes']));v=setup_activity(3,BINDING['bytes']-1)
        elif damage=='contract':h.download_bindings[JOB]={**BINDING,'bytes':BINDING['bytes']+1}
        elif damage=='scope':v['scope']='input-verification'
        before=len(j.events);progress=h.progress
        with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert len(j.events)==before and h.progress==progress and h.exported==NOW and not h.complete


def test_blocked_audit_read_produces_no_heartbeat_and_guard_stops(tmp_path,monkeypatch):
    import threading
    from pod_runtime_setup import SetupProgress
    from ovl_pipeline.canonical import read_json
    from ovl_pipeline.supervision import observe
    from run_rental_controller import normalized
    w=intent();w['baseline']['balance_usd']='100'
    c=Clock();output=tmp_path/'activity.json';monkeypatch.setenv('OVL_ACTIVITY_FILE',str(output))
    report=SetupProgress(clock=lambda:c.now);report('source');report(copied_bytes=BINDING['bytes']);report('selection')
    entered=threading.Event();release=threading.Event()
    def blocked_read():
        entered.set();assert release.wait(5)
    worker=threading.Thread(target=blocked_read);worker.start()
    with Journal(tmp_path/'j').lease() as j:
        h=SustainedHealth(j,w,'owned-pod',{}, {JOB:dict(BINDING)},wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms});h.start_job({**SELECTION,'kind':'setup'});h.activity(JOB,read_json(output));prior=h.progress
        try:
            assert entered.wait(2);c.advance(301)
            assert not h.activity(JOB,read_json(output)) and h.progress==prior
            file=tmp_path/'health.json';h.write(file);pod={'id':h.pod,'gpuCount':1,'costPerHr':'0.3','adjustedCostPerHr':'0.3'}
            account={'observed_epoch':c.now,'balance_usd':'99','account_hourly_usd':'0.3','pods':[pod]}
            from ovl_pipeline.canonical import digest
            watchdog={'observed_epoch':c.now,'plan_sha256':digest(w['plan']),'external_terminate_epoch':w['plan']['external_terminate_epoch'],'state':'ARMED'}
            decision=observe(w['plan'],normalized(w,account,pod,watchdog,read_json(file),c.now))
            assert 'stalled-or-future-progress' in decision['reasons'] and decision['action']!='CONTINUE'
            assert not h.complete and h.exported==NOW
        finally:release.set();worker.join(2)
    assert not worker.is_alive()
