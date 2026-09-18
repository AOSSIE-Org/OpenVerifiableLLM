"""Production startup telemetry is finite liveness, never a verification proof."""
import pytest
from test_pipeline import prepared
from test_production_health import bound
from test_workload_stage import intent
from production_health import ProductionHealth
from ovl_pipeline.canonical import EvidenceError
from ovl_pipeline.supervision import Journal


def observation(registration,**changes):
    stream=registration['coverage']['wikipedia']
    return {'schema':'ovl.runtime-production-scan.v1','process_instance':'b'*32,'pid':42,'sequence':1,'pass_index':1,
            'operation':'stream-validation','stream_sha256':stream['stream_sha256'],'documents':stream['documents'],
            'completed_documents':0,'complete':False,'scope':'operator-supervision-only-not-training-verification',**changes}


def start(h,job):
    h.start_job({'schema':'ovl.selected-workload-job.v1','job_sha256':job,'pod_id':h.pod,'kind':'production-record'})


def test_bounded_scan_prefixes_survive_restart_without_renewing_export_or_completion(prepared,tmp_path):
    control,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=intent()
    with Journal(tmp_path/'scan-health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings);start(h,root)
        exported=h.exported;progress=h.progress
        assert not h.activity(root,observation(r)) and h.progress==progress
        one=observation(r,sequence=2,completed_documents=1)
        assert h.activity(root,one) and not h.activity(root,one)
        assert not h.activity(root,{**one,'sequence':3})
        current=observation(r,sequence=4,pass_index=2,operation='coverage-census')
        assert not h.activity(root,current)
        assert h.exported==exported and not h.complete
    with Journal(tmp_path/'scan-health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings)
        assert not h.activity(root,current)
        complete={**current,'sequence':5,'completed_documents':current['documents'],'complete':True}
        assert h.activity(root,complete) and not h.activity(root,complete)
        c={
            'phase':'wikipedia','phase_step':1,'global_step':1,'cursor':1,'transcript':'a'*64,
            'schedule':'constant-lr-v1','accumulation':'none','scaler':'none'}
        numeric={'schema':'ovl.runtime-activity.v1','process_instance':'b'*32,'pid':42,'sequence':6,
                 'kind':'completed-numerical-update','control':c,'scope':'operator-supervision-only-not-training-verification'}
        assert h.activity(root,numeric)
        with pytest.raises(EvidenceError):h.activity(root,{**complete,'pass_index':3,'sequence':7})
        assert h.exported==exported and not h.complete


def test_changed_inputs_process_pass_or_counts_fail_before_credit(prepared,tmp_path):
    control,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=intent()
    with Journal(tmp_path/'bad-scan-health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings);start(h,root)
        first=observation(r,sequence=3,pass_index=2,completed_documents=1)
        assert h.activity(root,first)
        mutations=[{'stream_sha256':'f'*64},{'documents':first['documents']+1},{'documents':True},
            {'pid':43},{'pid':True},{'process_instance':'c'*32},{'sequence':2},{'sequence':True},
            {'pass_index':1},{'pass_index':17},{'pass_index':True},{'operation':'unselected-work'},
            {'operation':'coverage-census'},{'completed_documents':0},{'completed_documents':-1},
            {'completed_documents':first['documents']+1},{'completed_documents':True},{'complete':'yes'},
            {'complete':True,'completed_documents':0}]
        count=len(j.events)
        for change in mutations:
            with pytest.raises(EvidenceError):h.activity(root,{**first,'sequence':4,**change})
            assert len(j.events)==count
        numeric={'schema':'ovl.runtime-activity.v1','process_instance':'c'*32,'pid':42,'sequence':5}
        with pytest.raises(EvidenceError):h.activity(root,numeric)
        assert len(j.events)==count
        # Even a caller-side in-memory mutation cannot silently redefine the
        # registered stream used by later telemetry or retention checks.
        r['coverage']['wikipedia']['documents']+=1
        with pytest.raises(EvidenceError,match='registration changed'):
            h.activity(root,{**first,'sequence':4,'documents':first['documents']+1})
        assert len(j.events)==count


def test_restarted_production_job_rechecks_historical_retention_binding(prepared,tmp_path):
    control,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=intent()
    with Journal(tmp_path/'historical-scan-health').lease() as j:
        h=ProductionHealth(j,w,control.profile['pod_id'],r,bindings);start(h,root)
        h.activity(root,observation(r,completed_documents=1))
    altered={root:{**bindings[root],'worker_sha256':'f'*64}}
    with Journal(tmp_path/'historical-scan-health').lease() as j:
        with pytest.raises(EvidenceError,match='historical production retention'):
            ProductionHealth(j,w,control.profile['pod_id'],r,altered)
