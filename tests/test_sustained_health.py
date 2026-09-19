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
