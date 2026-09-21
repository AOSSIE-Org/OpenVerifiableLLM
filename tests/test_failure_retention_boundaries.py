"""Progress identity and final deadline failures stay strict across backup routes."""
import time
import pytest

import pod_versioned_export as versioned
import failed_stage_retention as backup
from ovl_pipeline.canonical import EvidenceError,read_json
from ovl_pipeline.supervision import Journal
from workload_health import Health
from test_workload_stage import intent,staged
from test_pod_job_client import fixture
from test_failed_stage_retention import failed,arguments
from sustained_pilot_abort import stop_and_retain


def test_checkpoint_and_whole_output_retransmission_share_durable_highwater(tmp_path):
    t,remote,calls,job,root,worker,worker_root=fixture(tmp_path)
    path=remote/'output/checkpoint';path.mkdir(parents=True);(path/'state').write_bytes(b'x'*(3*1024**2))
    w=intent();clock=[w['plan']['input']['now_epoch']];original=t.get;reports=[]
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,w,t.profile['pod_id']);h.now=lambda:clock[0]
        def progress(operation,counts,total):
            reports.append(operation);h.bytes(operation,counts,total=total)
        def partial(name,destination,expected,deadline,**kw):
            kw['progress']({'bytes_sent':0,'bytes_received':2*1024**2})
            raise EvidenceError('synthetic incomplete prefix')
        t.get=partial;clock[0]+=10
        with pytest.raises(EvidenceError):versioned.export(t,'output/checkpoint',tmp_path/'store',tmp_path/'first',int(time.time())+30,progress=progress)
        old=h.progress
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,w,t.profile['pod_id']);h.now=lambda:clock[0];clock[0]+=20
        with pytest.raises(EvidenceError):versioned.export(t,'output',tmp_path/'store',tmp_path/'backup',int(time.time())+30,progress=progress,allow_bulk=False)
        assert reports[0]==reports[1] and h.progress==old


def test_receipt_written_after_deadline_cannot_be_adopted_as_returned_backup(tmp_path,monkeypatch):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    with Journal(tmp_path/'journal').lease() as journal:
        h=Health(journal,intent(),t.profile['pod_id']);out,store=failed(tmp_path,h,t,remote,job,root,worker,worker_root)
        args,kw=arguments(tmp_path,t,h,job,root,worker,worker_root,out,store)
        clock=[time.time()];t.wall=t.monotonic=lambda:clock[0];original=versioned.write_json
        def delayed(path,value):
            original(path,value)
            if path.name=='export.json':clock[0]+=2000
        monkeypatch.setattr(versioned,'write_json',delayed)
        with pytest.raises(EvidenceError,match='original deadline'):stop_and_retain(*args,**kw)
        assert (out/'failure-retention/export-000/export.json').exists()
        assert not(out/'failure-retention/export-000/returned.json').exists()
        with pytest.raises(EvidenceError,match='preserved failure-retention refusal'):stop_and_retain(*args,**kw)
        assert not h.jobs[root]['finished'] and not(out/'failure-retention/retention.json').exists()
