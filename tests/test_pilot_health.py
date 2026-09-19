import copy
import pytest
from pilot_health import PilotHealth
from test_workload_health import Clock,JOB,SELECTION,activity
from test_external_watchdog import intent,NOW
from ovl_pipeline.canonical import EvidenceError
from ovl_pipeline.supervision import Journal


BINDING={'schema':'ovl.pilot-validation-binding.v1','stream_sha256':'c'*64,'documents':100}


def health(j,c,bindings=None):
    return PilotHealth(j,intent(),'owned-pod',bindings or {JOB:dict(BINDING)},
                       wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})


def validation(sequence=1,count=0,complete=False):
    return {'schema':'ovl.runtime-stream-validation.v1','process_instance':'b'*32,'pid':123,'sequence':sequence,
            'stream_sha256':'c'*64,'documents':100,'completed_documents':count,'complete':complete,
            'scope':'operator-supervision-only-not-training-verification'}


def test_prefix_restart_no_heartbeat_or_export_credit_then_numerical_work(tmp_path):
    c=Clock();path=tmp_path/'j'
    with Journal(path).lease() as j:
        h=health(j,c);h.start_job(SELECTION);c.advance(1)
        assert h.activity(JOB,validation());assert h.progress==NOW+1
        c.advance(30);assert h.activity(JOB,validation(2,20));assert h.exported==NOW
    with Journal(path).lease() as j:
        h=health(j,c);assert not h.start_job(SELECTION)
        c.advance(30);assert not h.activity(JOB,validation(2,20))
        assert not h.activity(JOB,validation(3,20));assert h.progress==NOW+31
        assert h.activity(JOB,validation(4,100));c.advance(30)
        assert h.activity(JOB,validation(5,100,True));assert h.exported==NOW
        assert h.activity(JOB,activity(6));assert not h.complete
        with pytest.raises(EvidenceError):h.activity(JOB,validation(7,100,True))
    with Journal(path).lease() as j:
        h=health(j,c);assert h.activities[JOB]['sequence']==6 and h.exported==NOW


@pytest.mark.parametrize('damage',['root','total','over','negative','bool-count','premature-complete','pid','uuid','sequence','changed-repeat','scope','extra'])
def test_foreign_changed_unbounded_or_regressing_validation_fails_closed(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job(SELECTION);h.activity(JOB,validation(3,40));before=len(j.events)
        v=validation(4,50)
        if damage=='root':v['stream_sha256']='d'*64
        elif damage=='total':v['documents']=101
        elif damage=='over':v['completed_documents']=101
        elif damage=='negative':v['completed_documents']=-1
        elif damage=='bool-count':v['completed_documents']=True
        elif damage=='premature-complete':v['complete']=True
        elif damage=='pid':v['pid']=124
        elif damage=='uuid':v['process_instance']='e'*32
        elif damage=='sequence':v['sequence']=2
        elif damage=='changed-repeat':v['sequence']=3
        elif damage=='scope':v['scope']='verified'
        else:v['heartbeat']=True
        with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert len(j.events)==before and h.exported==NOW


def test_binding_changes_rejected_live_and_after_restart(tmp_path):
    c=Clock();b={JOB:dict(BINDING)};path=tmp_path/'j'
    with Journal(path).lease() as j:
        h=health(j,c,b);h.start_job(SELECTION);b[JOB]['documents']=101
        with pytest.raises(EvidenceError):h.activity(JOB,validation())
        with pytest.raises(EvidenceError):h.activity(JOB,activity())
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):health(j,c,b)


@pytest.mark.parametrize('damage',['pid','sequence','instance'])
def test_numerical_transition_preserves_process_and_sequence(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job(SELECTION);h.activity(JOB,validation(3,40))
        v=activity(4)
        if damage=='pid':v['pid']+=1
        elif damage=='sequence':v['sequence']=3
        else:v['process_instance']='e'*32
        with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert not h.activities


def test_missing_final_snapshot_is_not_required_for_operational_liveness(tmp_path):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job(SELECTION);h.activity(JOB,validation(3,40))
        assert h.activity(JOB,activity(5))
        assert h.exported==NOW and not h.complete


def test_generic_health_does_not_admit_new_protocol(tmp_path):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=c.health(j);h.start_job(SELECTION)
        with pytest.raises(EvidenceError):h.activity(JOB,validation())


def test_completed_prefix_cannot_be_recycled_for_credit(tmp_path):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job(SELECTION);h.activity(JOB,validation(1,100,True));c.advance(300)
        assert not h.activity(JOB,validation(1,100,True))
        for v in (validation(2,100,True),validation(2,100,False),validation(2,99)):
            with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert h.progress==h.exported==NOW


def test_nonpilot_binding_does_not_poison_journal_after_append(tmp_path):
    c=Clock();path=tmp_path/'j'
    with Journal(path).lease() as j:
        h=health(j,c);h.start_job({**SELECTION,'kind':'setup'})
        assert h.activity(JOB,activity())
    with Journal(path).lease() as j:
        h=health(j,c);assert h.activities[JOB]==activity()


def test_dedicated_sustained_protocol_rejects_tiny_wrapper_without_append(tmp_path):
    from test_pilot_phase_activity import phase
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job(SELECTION);before=len(j.events)
        with pytest.raises(EvidenceError):h.activity(JOB,phase(1))
        assert len(j.events)==before


def test_missing_historical_binding_is_rejected_instead_of_adopted_from_peer(tmp_path):
    c=Clock();path=tmp_path/'j'
    with Journal(path).lease() as j:
        h=health(j,c);h.start_job(SELECTION)
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):health(j,c,{'d'*64:dict(BINDING)})
