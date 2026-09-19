"""Durable controller-host observations; no real provider or CUDA claim."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import workload_health as m
from test_external_watchdog import intent,NOW
from ovl_pipeline.canonical import EvidenceError,digest,inventory,write_json,read_json
from ovl_pipeline.supervision import Journal,ControllerBusy


JOB='a'*64
SELECTION={'schema':'ovl.selected-workload-job.v1','job_sha256':JOB,'pod_id':'owned-pod','kind':'pilot'}


class Clock:
    def __init__(self):self.now=NOW;self.ms=0;self.boot='fixture-boot'
    def advance(self,n):self.now+=n;self.ms+=n*1000
    def health(self,j):return m.Health(j,intent(),'owned-pod',wall=lambda:self.now,clock=lambda:{'boot_id':self.boot,'boottime_ms':self.ms})


def activity(sequence=1,step=1):
    return {'schema':'ovl.runtime-activity.v1','process_instance':'b'*32,'pid':123,'sequence':sequence,
            'kind':'completed-numerical-update','scope':'operator-supervision-only-not-training-verification',
            'control':{'phase':'wikipedia','global_step':step,'phase_step':step,'cursor':step*32,'transcript':digest(step),
                       'schedule':'constant-lr-v1','accumulation':'none','scaler':'none','pilot_cycle':0}}


def export(tmp_path,h):
    root=tmp_path/'export';root.mkdir()
    status={'schema':'ovl.workload-job-exit.v1','job_sha256':JOB,'state':'EXITED','exit_code':0}
    write_json(root/'exit.json',status);(root/'state.safetensors').write_bytes(b'explicit byte-preservation fixture; not safe state proof')
    files=inventory(root,['exit.json','state.safetensors']);assert h.exported_files(JOB,root,files)
    return root,files,status


def test_restart_and_heartbeat_cannot_renew_stale_work_or_repeated_bytes(tmp_path):
    c=Clock();path=tmp_path/'journal';health=tmp_path/'health.json'
    with Journal(path).lease() as j:
        h=c.health(j);h.start_job(SELECTION);c.advance(10)
        assert not h.bytes('c'*64,{'bytes_sent':0,'bytes_received':100},total=200)
        assert h.write(health)['progress_epoch']==NOW
    c.advance(300)
    with Journal(path).lease() as j:
        h=c.health(j);assert h.start_job(SELECTION) is False
        for n in (0,20,100,101):assert h.bytes('c'*64,{'bytes_sent':0,'bytes_received':n},total=200) is False
        v=h.write(health);assert v['observed_epoch']==NOW+310 and v['progress_epoch']==NOW and v['exported_checkpoint_epoch']==NOW
        assert h.bytes('c'*64,{'bytes_sent':0,'bytes_received':200},total=200)
        assert h.write(health)['progress_epoch']==NOW+310


def test_tiny_trickle_and_changed_total_cannot_keep_an_idle_rental_fresh(tmp_path):
    c=Clock()
    with Journal(tmp_path/'journal').lease() as j:
        h=c.health(j);h.start_job(SELECTION)
        for index in range(15):
            c.advance(30)
            assert not h.bytes('c'*64,{'bytes_sent':0,'bytes_received':(index+1)*200},total=2*1024**2)
        assert h.progress==NOW
        with pytest.raises(EvidenceError):h.bytes('c'*64,{'bytes_sent':0,'bytes_received':3001},total=3001)
        assert h.bytes('c'*64,{'bytes_sent':0,'bytes_received':1024**2},total=2*1024**2)
        assert h.progress==NOW+450
        with pytest.raises(EvidenceError):h.bytes('c'*64,{'bytes_sent':0,'bytes_received':3*1024**2},total=2*1024**2)


def test_changed_sequence_requires_new_control_but_warmup_reset_is_real_paid_work(tmp_path):
    c=Clock()
    with Journal(tmp_path/'journal').lease() as j:
        h=c.health(j);h.start_job(SELECTION);c.advance(1)
        assert h.activity(JOB,activity(1,4))
        c.advance(30);assert not h.activity(JOB,activity(1,4))
        assert not h.activity(JOB,activity(2,4));assert h.progress==NOW+1
        assert h.activity(JOB,activity(3,1));assert h.progress==NOW+31
        with pytest.raises(EvidenceError):h.activity(JOB,activity(2,4))
        changed=activity(3,2)
        with pytest.raises(EvidenceError):h.activity(JOB,changed)
        changed=activity(4,2);changed['pid']=999
        with pytest.raises(EvidenceError):h.activity(JOB,changed)


def test_reused_export_and_unverified_exit_do_not_renew_and_completion_rechecks_bytes(tmp_path):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=c.health(j);h.start_job(SELECTION)
        status={'schema':'ovl.workload-job-exit.v1','job_sha256':JOB,'state':'EXITED','exit_code':0}
        with pytest.raises(EvidenceError,match='no verified'):h.job_exit(JOB,status)
        c.advance(5);root,files,status=export(tmp_path,h);assert h.exported==NOW+5
        c.advance(60);assert not h.exported_files(JOB,root,files);assert h.exported==NOW+5
        with pytest.raises(EvidenceError,match='not all exited'):h.finish(root,files)
        h.job_exit(JOB,status);h.finish(root,files)
        v=h.write(tmp_path/'health.json');assert v['complete'] is True and v['exported_checkpoint_epoch']==v['progress_epoch']==NOW+65
    with Journal(path).lease() as j:
        h=c.health(j);assert h.write(tmp_path/'adopted.json')['complete'] is True
    (root/'state.safetensors').write_bytes(b'changed after completion')
    with Journal(path).lease() as j:
        h=c.health(j)
        with pytest.raises(EvidenceError):h.write(tmp_path/'bad-adoption.json')
    assert not(tmp_path/'bad-adoption.json').exists()


@pytest.mark.parametrize('damage',['missing','altered','extra','symlink','wrong-root'])
def test_export_must_match_actual_complete_selected_files(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'journal').lease() as j:
        h=c.health(j);h.start_job(SELECTION);root=tmp_path/'export';root.mkdir();(root/'file').write_bytes(b'correct')
        files=inventory(root,['file'])
        if damage=='missing':(root/'file').unlink()
        elif damage=='altered':(root/'file').write_bytes(b'changed')
        elif damage=='extra':(root/'extra').write_bytes(b'omitted')
        elif damage=='symlink':(root/'file').unlink();(root/'file').symlink_to(tmp_path/'outside')
        else:files[0]['sha256']='f'*64
        with pytest.raises(EvidenceError):h.exported_files(JOB,root,files)
        assert h.exported==NOW and not h.exports


@pytest.mark.parametrize('damage',['reboot','rollback','deadline'])
def test_health_cannot_renew_rental_after_clock_or_boot_change(tmp_path,damage):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=c.health(j);h.start_job(SELECTION);c.advance(10);h.activity(JOB,activity())
    if damage=='reboot':c.boot='other-boot'
    elif damage=='rollback':c.now-=20
    else:c.advance(1000)
    with Journal(path).lease() as j:
        h=c.health(j)
        with pytest.raises(EvidenceError):h.write(tmp_path/'health.json')
    assert not(tmp_path/'health.json').exists()


def test_active_or_changed_job_and_duplicate_controller_are_rejected(tmp_path):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=c.health(j);h.start_job(SELECTION)
        with pytest.raises(EvidenceError):h.start_job({**SELECTION,'job_sha256':'d'*64})
        with pytest.raises(EvidenceError):h.start_job({**SELECTION,'kind':'full-replay'})
        with pytest.raises(ControllerBusy):
            with Journal(path).lease():pass


def test_final_export_requires_matching_terminal_records_and_preserved_prior_exports(tmp_path):
    c=Clock()
    with Journal(tmp_path/'journal').lease() as j:
        h=c.health(j);h.start_job(SELECTION);root,files,status=export(tmp_path,h);h.job_exit(JOB,status)
        final=tmp_path/'final';final.mkdir();write_json(final/'exit.json',{**status,'job_sha256':'f'*64})
        with pytest.raises(EvidenceError,match='terminal'):h.finish(final,inventory(final,['exit.json']))
        write_json(final/'exit.json',status);(root/'state.safetensors').unlink()
        with pytest.raises(EvidenceError):h.finish(final,inventory(final,['exit.json']))
        assert not h.complete


def test_actual_health_writer_causes_real_controller_to_teardown_only_after_exports(tmp_path):
    from test_rental_controller import RentalFake
    f=RentalFake(tmp_path);f.healthy=False;original=f.account
    with Journal(tmp_path/'health-journal').lease() as j:
        h=m.Health(j,f.i,'owned-pod',wall=lambda:f.now,clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
        h.start_job(SELECTION)
        def account():
            if f.alive:
                if f.elapsed>=60 and not h.complete:
                    root,files,status=export(tmp_path,h);h.job_exit(JOB,status);h.finish(root,files)
                h.write(f.health)
            return original()
        f.account=account;f.run()
        assert h.complete and not f.alive and f.writes==1
        term=next(row for row in f.calls if row[0]=='terminate')
        assert term[2]==NOW+60
        result=read_json(f.directory/'result.json')
        assert result['confirmed_absent_epoch']>=NOW+75 and result['provider_billing_reconciliation']=='PENDING'


def test_real_controller_stops_when_heartbeat_refreshes_without_actual_work(tmp_path):
    from test_rental_controller import RentalFake
    f=RentalFake(tmp_path);f.healthy=False;original=f.account
    with Journal(tmp_path/'health-journal').lease() as j:
        h=m.Health(j,f.i,'owned-pod',wall=lambda:f.now,clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
        h.start_job(SELECTION)
        def account():
            if f.alive:h.write(f.health)
            return original()
        f.account=account;f.run()
        assert h.progress==NOW and h.exported==NOW and not h.complete and not f.alive
        stop=read_json(f.directory/'stop-request.json')
        assert stop['observed_epoch']<=NOW+310
        assert any(c[0]=='terminate' for c in f.calls)


@pytest.mark.parametrize('kind',['production-record','full-replay'])
def test_generic_artifact_health_cannot_authorize_unwired_production_completion(tmp_path,kind):
    c=Clock()
    with Journal(tmp_path/'journal').lease() as j:
        h=c.health(j)
        with pytest.raises(EvidenceError):h.start_job({**SELECTION,'kind':kind})
        assert not h.jobs and not h.complete
