"""State-aware cost completion on real CPU files; no live production acceptance."""
from pathlib import Path
import sys,time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from production_health import ProductionHealth
import production_retention
import pod_checkpoint_handoff
import publication_activity
from test_pipeline import prepared
from test_production_retention import terminal_fixture
from test_workload_stage import intent
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json
from ovl_pipeline.supervision import Journal


def bound(prepared,tmp_path):
    control,record,remote,job,root,worker,r=terminal_fixture(prepared,tmp_path,with_registration=True)
    bindings={root:{'job_file':job,'worker_sha256':worker,'control':control,'transports':[record]}}
    return control,record,job,root,worker,r,bindings


def test_logs_only_cannot_finish_production_but_all_retained_outputs_can(prepared,tmp_path):
    control,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=intent()
    proof=production_retention.retain(control,[record],job,root,worker,tmp_path/'objects',tmp_path/'retained',int(time.time())+60)
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'production-record'})
        logs=tmp_path/'logs';logs.mkdir();write_json(logs/'exit.json',proof['terminal'])
        h.exported_files(root,logs,inventory(logs,['exit.json']))
        with pytest.raises((EvidenceError,FileNotFoundError)):h.job_exit(root,proof['terminal'])
        h.terminal_retained(root,tmp_path/'retained/retention.json');h.job_exit(root,proof['terminal'])
        h.finish(logs,inventory(logs,['exit.json']));assert h.write(tmp_path/'health.json')['complete']
    # A restart rechecks actual retained states, not just a saved PASS field.
    receipt=read_json(Path(proof['roots'][1]['receipt_path']));state=next(f['path'] for f in receipt['files'] if f['path'].endswith('state.safetensors'))
    p=Path(receipt['files_directory'])/state;p.chmod(0o600);p.write_bytes(b'changed after completion')
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings)
        with pytest.raises(EvidenceError):h.write(tmp_path/'health.json')


def test_verified_snapshot_publication_credit_is_finite_and_restart_deduplicated(prepared,tmp_path):
    control,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=intent();deadline=int(time.time())+60
    snapshot=tmp_path/'snapshot';export=pod_checkpoint_handoff.snapshot(record,r,digest(r),snapshot,deadline)
    publication_activity.emit(tmp_path/'activity',digest(r),export['boundary_sha256'],'request-public-commit-verified',{'revision':'a'*40},deadline)
    value=read_json(tmp_path/'activity/request-public-commit-verified.json')
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings)
        h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':root,'pod_id':h.pod,'kind':'production-record'})
        assert h.publication(root,snapshot,deadline,value)
        assert not h.publication(root,snapshot,deadline,value)
        with pytest.raises(EvidenceError):h.publication(root,snapshot,deadline+1,value)
        assert len(h.publications)==1
    with Journal(tmp_path/'health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings)
        assert not h.publication(root,snapshot,deadline,value)
        other={**value,'boundary_sha256':'f'*64}
        with pytest.raises(EvidenceError):h.publication(root,snapshot,deadline,other)
        bad={**value,'identity_sha256':'e'*64}
        with pytest.raises(EvidenceError):h.publication(root,snapshot,deadline,bad)
        assert len(h.publications)==1
