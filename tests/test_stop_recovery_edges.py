"""Synthetic stop recovery faults: no new workload or renewed time allowance."""
from pathlib import Path
import time
import pytest
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from pod_job_client import stop_delivery,worker_stop_request,retain_stop_reason
from test_workload_stage import staged


def test_ack_lost_after_immutable_install_reuses_bytes_then_validates_receipt(tmp_path):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    marker=tmp_path/'marker';receipt=tmp_path/'receipt.json';write_json(marker,worker_stop_request(root))
    name='jobs/'+root+'/request-stop';original=t.stream
    def lost(*a,**kw):
        original(*a,**kw)
        raise EvidenceError('synthetic acknowledgement loss after installation')
    t.stream=lost
    with pytest.raises(EvidenceError,match='acknowledgement loss'):
        stop_delivery(t,root,name,marker,receipt,int(time.time())+20)
    assert not receipt.exists()
    dest=remote/name;before=(dest.read_bytes(),dest.stat().st_ino,dest.stat().st_mtime_ns)
    t.stream=original
    value=stop_delivery(t,root,name,marker,receipt,int(time.time())+20)
    assert before==(dest.read_bytes(),dest.stat().st_ino,dest.stat().st_mtime_ns)
    count=len(calls)
    assert stop_delivery(t,root,name,marker,receipt,int(time.time())+20)==value and len(calls)==count


@pytest.mark.parametrize('damage',['empty','json','job','profile','destination','hash','bytes','endpoint','exit','sent','received','symlink'])
def test_wrong_retained_stop_receipt_never_skips_or_redelivers(tmp_path,damage):
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    marker=tmp_path/'marker';receipt=tmp_path/'receipt.json';write_json(marker,worker_stop_request(root))
    name='jobs/'+root+'/request-stop'
    saved=stop_delivery(t,root,name,marker,receipt,int(time.time())+20)
    if damage=='empty':saved={}
    elif damage=='job':saved['identity']['job_sha256']='f'*64
    elif damage=='profile':saved['identity']['profile_sha256']='f'*64
    elif damage=='destination':saved['identity']['destination']='request-stop'
    elif damage=='hash':saved['identity']['sha256']='f'*64
    elif damage=='bytes':saved['identity']['bytes']+=1
    elif damage=='endpoint':saved['transfer']['endpoint_observation_sha256']='f'*64
    elif damage=='exit':saved['transfer']['process_exit_code']=1
    elif damage=='sent':saved['transfer']['bytes_sent']=0
    elif damage=='received':saved['transfer']['bytes_received']=0
    if damage=='json':receipt.write_bytes(b'{broken')
    elif damage=='symlink':
        target=tmp_path/'copy';write_json(target,saved);receipt.unlink();receipt.symlink_to(target)
    else:write_json(receipt,saved)
    count=len(calls)
    with pytest.raises(EvidenceError):stop_delivery(t,root,name,marker,receipt,int(time.time())+20)
    assert len(calls)==count


def test_lost_ack_cause_transition_keeps_first_reason_and_no_clock_reset(tmp_path):
    job='a'*64;first={'schema':'ovl.dispatcher-stop-request.v1','job_sha256':job,'reason':'fixed graceful-stop deadline'}
    later={'schema':'ovl.rental-stop-request.v1','intent_sha256':'d'*64,'pod_id':'synthetic',
           'observed_epoch':1000,'reasons':['synthetic later cause']}
    retain_stop_reason(tmp_path,first,job,graceful_reason=first['reason'])
    original=(tmp_path/'stop-reason.json').read_bytes()
    # No delivery receipt exists: installation acknowledgement may have been lost.
    retain_stop_reason(tmp_path,later,job,graceful_reason=first['reason'])
    assert (tmp_path/'stop-reason.json').read_bytes()==original
    assert read_json(tmp_path/'controller-stop-reason.json')['request']==later
    for invalid in (first,None,{**later,'observed_epoch':1001}):
        with pytest.raises(EvidenceError):retain_stop_reason(tmp_path,invalid,job,graceful_reason=first['reason'])


def test_first_controller_stop_cannot_disappear_before_graceful_clock(tmp_path):
    from run_production_stage import stop_window
    job={'deadline_epoch':4000,'stop_grace_seconds':30}
    plan={'request_checkpoint_epoch':2000,'input':{'now_epoch':900,'checkpoint_grace_seconds':1000}}
    request={'schema':'ovl.rental-stop-request.v1','observed_epoch':1000}
    _,bound=stop_window(tmp_path,request,job,plan,1100,200)
    with pytest.raises(EvidenceError,match='disappeared'):stop_window(tmp_path,None,job,plan,1101,200)
    assert read_json(tmp_path/'stop-intent.json')['hard_stop_epoch']==bound


def test_transient_failure_with_unresolved_cleanup_never_retries(tmp_path,monkeypatch):
    import pod_observation_retry as m
    from pod_transfer import TransientTransportError
    from test_pod_observation_retry import Health
    error=TransientTransportError('synthetic transient');error.transport_cleanup_diagnostic={'exception_class':'OSError'}
    calls=[]
    def fail(*args):calls.append(args);raise error
    monkeypatch.setattr(m,'job_supervision',fail)
    with pytest.raises(TransientTransportError) as caught:
        m.read('supervision',None,('a','b'),Health(),tmp_path/'health',tmp_path,sleep=lambda _:pytest.fail('cleanup failure retried'))
    assert caught.value is error and len(calls)==1


def test_diagnostics_import_failure_preserves_primary(tmp_path,monkeypatch):
    import builtins
    import pod_observation_retry as m
    from test_pod_observation_retry import Health
    original=builtins.__import__;error=EvidenceError('synthetic primary')
    def blocked(name,*a,**kw):
        if name=='private_transport_diagnostics':raise ImportError('synthetic unavailable diagnostics')
        return original(name,*a,**kw)
    def fail(*a):raise error
    monkeypatch.setattr(m,'job_supervision',fail);monkeypatch.setattr(builtins,'__import__',blocked)
    with pytest.raises(EvidenceError) as caught:m.read('supervision',None,('a','b'),Health(),tmp_path/'h',tmp_path)
    assert caught.value is error


def test_supervision_parse_failure_retains_reply_prefix(tmp_path):
    import base64
    from pod_job_client import job_supervision
    t,remote,calls,job,root,worker,worker_root=staged(tmp_path)
    t.stream=lambda argv,sink,*a,**kw:sink.write(b'{broken')
    with pytest.raises(EvidenceError) as caught:job_supervision(t,root,worker_root,int(time.time())+20)
    assert base64.b64decode(caught.value.metadata_response_diagnostic['prefix_b64'])==b'{broken'


def test_abort_failure_cannot_replace_primary_and_stack_closes(tmp_path):
    from production_run_coordinator import Run
    from types import SimpleNamespace
    run=Run.__new__(Run);run.output=tmp_path;run.health=SimpleNamespace(complete=False);closed=[]
    def fail(*a):raise EvidenceError('synthetic abort failure')
    run.abort=fail;run.stack=SimpleNamespace(__exit__=lambda *a:closed.append(a))
    error=EvidenceError('synthetic original')
    with pytest.raises(EvidenceError) as caught:
        try:raise error
        finally:run.__exit__(EvidenceError,error,error.__traceback__)
    assert caught.value is error and len(closed)==1
