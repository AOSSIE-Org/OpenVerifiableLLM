"""Selected upload timing is distinct from export speed and never renews a rental."""
import time
import pytest
import run_sustained_pilot as m
from test_sustained_pilot_dispatch import fixture
from ovl_pipeline.canonical import EvidenceError,file_hash,read_json,write_json,digest


def selected(tmp_path, size=34143739):
    values=fixture(tmp_path)
    run,plan,rental,t,remote,calls,controller,heartbeat=values
    path=tmp_path/'bootstrap.bin'
    with path.open('wb') as f:f.truncate(size)
    plan['uploads']=[{'path':path.name,'remote_path':'inputs/'+path.name,'bytes':size,'sha256':file_hash(path)}]
    bind_uploads(tmp_path,plan,t)
    return values


def bind_uploads(tmp_path,plan,t):
    for stage in plan['stages']:
        path=tmp_path/stage['template_path'];job=read_json(path)
        job['required_files']=[f for f in job['required_files'] if not f['path'].startswith(t.profile['remote_root']+'/inputs/')]
        job['required_files'] += [{'path':t.profile['remote_root']+'/'+u['remote_path'],'bytes':u['bytes'],'sha256':u['sha256']} for u in plan['uploads']]
        write_json(path,job);stage['template_sha256']=digest(job)


def test_measured_slow_upload_receives_its_selected_window_without_export_change(tmp_path,monkeypatch):
    run,plan,rental,t,remote,calls,controller,heartbeat=selected(tmp_path)
    plan['stages'][0]['export_reserve_seconds']=100  # fits the synthetic 300-second work window
    original=plan['timing']['transfer_floor_bytes_per_second']
    assert m.upload_seconds(plan,plan['uploads'][0])==64
    plan['timing']['upload_floor_bytes_per_second']=256*1024
    assert m.upload_seconds(plan,plan['uploads'][0])==162
    observed=[]
    def put(name,source,deadline,**kw):
        observed.append(deadline-int(time.time()))
        raise RuntimeError('synthetic stop before transfer')
    monkeypatch.setattr(t,'put',put)
    with pytest.raises(RuntimeError,match='synthetic stop'):run()
    assert observed and 160<=observed[0]<=162
    assert plan['timing']['transfer_floor_bytes_per_second']==original
    assert not calls


@pytest.mark.parametrize('floor',[True,65535,2**34+1,'262144',0])
def test_invalid_upload_floor_fails_before_transport(tmp_path,floor):
    run,plan,_,_,_,calls,_,_=selected(tmp_path,1)
    plan['timing']['upload_floor_bytes_per_second']=floor
    with pytest.raises(EvidenceError):run()
    assert not calls


def test_excess_upload_window_and_unknown_timing_are_rejected(tmp_path):
    run,plan,_,_,_,calls,_,_=selected(tmp_path,60*1024**2)
    plan['timing']['upload_floor_bytes_per_second']=65536
    with pytest.raises(EvidenceError,match='bounded setup upload window'):run()
    plan['timing']['upload_floor_bytes_per_second']=262144
    plan['timing']['unselected_upload_extension']=1
    with pytest.raises(EvidenceError,match='timing fields'):run()
    assert not calls


def test_phase_budget_counts_every_upload_acknowledgement_allowance(tmp_path):
    run,plan,_,t,_,calls,_,_=selected(tmp_path,1)
    one=plan['uploads'][0]
    plan['uploads']=[{**one,'remote_path':'inputs/copy-'+str(i)} for i in range(100)]
    bind_uploads(tmp_path,plan,t)
    with pytest.raises(EvidenceError,match='phase budgets exceed'):run()
    assert not calls


def test_late_post_transfer_hash_cannot_receive_an_upload_receipt(tmp_path,monkeypatch):
    from sustained_health import SustainedHealth
    run,plan,_,t,_,calls,_,_=selected(tmp_path,1)
    base=int(time.time());clock=[base]
    monkeypatch.setattr(SustainedHealth,'now',lambda self:clock[0])
    def put(name,source,deadline,**kw):
        clock[0]=deadline
        return {'sha256':file_hash(source),'bytes_sent':source.stat().st_size}
    monkeypatch.setattr(t,'put',put)
    with pytest.raises(EvidenceError,match='acknowledgement exceeded'):run()
    assert not list((tmp_path/'sustained/uploads').glob('*.json')) and not calls


def test_v2_admission_counts_per_upload_ack_windows(tmp_path):
    from test_sustained_run_phase import configured,phase
    from ovl_pipeline.supervision import Journal
    plan,rental,t,remote,calls,health,run=configured(tmp_path)
    path=tmp_path/'small.bin';path.write_bytes(b'x')
    plan['uploads']=[{'path':path.name,'remote_path':'inputs/small.bin','bytes':1,'sha256':file_hash(path)}]
    bind_uploads(tmp_path,plan,t)
    with Journal(tmp_path/'journal').lease() as journal:
        h=health(journal);p=phase(plan,0,[])
        work=sum(s['work_seconds']+s['export_reserve_seconds'] for s in p['stages'])
        h.now=lambda:rental['watchdog_intent']['plan']['request_checkpoint_epoch']-work-2
        with pytest.raises(EvidenceError,match='remaining original work window'):run(p,h,'one')
    assert not calls


@pytest.mark.parametrize('allowance,passes',[(64,False),(162,True)])
def test_actual_installed_upload_requires_delayed_ack_within_allowance(tmp_path,monkeypatch,allowance,passes):
    # Real pipe IO and atomic installation; only elapsed clocks are accelerated.
    import pod_transfer
    from test_pod_transfer import setup
    script=pod_transfer.REMOTE_PUT
    script=script.replace("print(", "__import__('time').sleep(0.9);print(")
    assert script!=pod_transfer.REMOTE_PUT
    monkeypatch.setattr(pod_transfer,'REMOTE_PUT',script)
    t,remote,calls,processes=setup(tmp_path)
    source=tmp_path/'payload';source.write_bytes(b'synthetic complete input')
    start=time.monotonic();clock=lambda:1000+(time.monotonic()-start)*100
    t.wall=clock;t.monotonic=clock
    if passes:
        result=t.put('inputs/payload',source,1000+allowance)
        assert result['sha256']==file_hash(source)
    else:
        with pytest.raises(pod_transfer.TransientTransportError):t.put('inputs/payload',source,1000+allowance)
    assert (remote/'inputs/payload').read_bytes()==source.read_bytes()
    assert all(p.poll() is not None for p in processes)
