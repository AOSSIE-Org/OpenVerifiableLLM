"""Synthetic filesystem observations and provider lifecycle; no paid resources."""
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import local_storage as storage
import pod_bulk_export as bulk
import run_rental_controller as controller
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from test_rental_controller import RentalFake
from test_pod_bulk_export import many
from pod_job_client import tree
import time

G=1024**3

def policy(tmp_path):
    return {'schema':'ovl.local-storage-budget.v1','directory':str(tmp_path),'device':tmp_path.stat().st_dev,
            'peak_incremental_bytes':100*G,'host_reserve_bytes':8*G,'shutdown_export_bytes':20*G,
            'observation_slack_bytes':4*G,'basis_sha256':'a'*64}


def selected(tmp_path,v):
    path=tmp_path/'budget.json';write_json(path,v);return storage.Budget(path,digest(v))


def test_capacity_is_workload_based_not_filesystem_percentage(tmp_path,monkeypatch):
    v=policy(tmp_path);monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(total=10000*G,free=108*G))
    b=selected(tmp_path,v);assert b.admit()==108*G and not b.shutdown_needed()
    storage.require_space(tmp_path,100*G)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(total=10000*G,free=108*G-1))
    with pytest.raises(EvidenceError,match='does not fit'):b.admit()
    with pytest.raises(EvidenceError,match='headroom'):storage.require_space(tmp_path,100*G)


@pytest.mark.parametrize('change',['pin','reserve','negative','bool','device','relative','symlink','missing-field'])
def test_changed_or_incomplete_budget_never_admits(tmp_path,change):
    v=policy(tmp_path);path=tmp_path/'budget.json'
    if change=='reserve':v['host_reserve_bytes']=G
    elif change=='negative':v['shutdown_export_bytes']=-1
    elif change=='bool':v['peak_incremental_bytes']=True
    elif change=='device':v['device']+=1
    elif change=='relative':v['directory']='relative'
    elif change=='missing-field':del v['observation_slack_bytes']
    elif change=='symlink':
        link=tmp_path/'alias';link.symlink_to(tmp_path,target_is_directory=True);v['directory']=str(link)
    write_json(path,v)
    with pytest.raises(EvidenceError):storage.Budget(path,'0'*64 if change=='pin' else digest(v))


def run(f,b):
    controller.run(f.directory,f.value,digest(f.value),f.heartbeat,f.health,
        get_account=f.account,provider_request=f.provider,wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,
        boot=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)},fence_root=f.directory.parent/'fences',storage_budget=b)


def test_low_space_prevents_creation(tmp_path,monkeypatch):
    f=RentalFake(tmp_path);b=selected(tmp_path,policy(tmp_path))
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=40*G))
    with pytest.raises(EvidenceError,match='does not fit'):run(f,b)
    assert f.writes==0 and not f.alive and not f.calls


def test_space_drop_requests_original_grace_before_provider_read_and_never_renews(tmp_path,monkeypatch):
    f=RentalFake(tmp_path);b=selected(tmp_path,policy(tmp_path));observed=[];initial=f.now
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=(120 if not f.alive else 31)*G))
    account=f.account
    def checked():
        if f.alive:
            stop=read_json(f.directory/'stop-request.json');observed.append(stop)
            assert stop['reasons']==['local-storage-headroom'] and stop['observed_epoch']==initial
        return account()
    f.account=checked;run(f,b)
    assert f.writes==1 and not f.alive and observed and all(x==observed[0] for x in observed)
    assert next(x for x in f.calls if x[0]=='terminate')[2]==initial+f.i['plan']['input']['checkpoint_grace_seconds']
    assert read_json(f.directory/'result.json')['complete'] is True
    run(f,b);assert f.writes==1


def test_bulk_capacity_accounts_for_stream_and_split_and_preserves_partial_on_drop(tmp_path,monkeypatch):
    t,remote,calls,data=many(tmp_path);files=tree(t,'output',int(time.time())+30);calls.clear()
    total=sum(x['bytes'] for x in files)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(total=10000*G,free=8*G+2*total-1))
    with pytest.raises(EvidenceError,match='headroom'):bulk.receive(t,'output',files,tmp_path/'short',int(time.time())+30)
    assert not calls
    free=[8*G+2*total]
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(total=10000*G,free=free[0]))
    def interrupted(argv,destination,maximum,deadline,**kw):
        destination.write(b'a'*32);free[0]=8*G-1
        kw['progress']({'bytes_received':32,'bytes_sent':0})
        raise AssertionError('low-space callback must stop transport')
    t.stream=interrupted
    with pytest.raises(EvidenceError,match='headroom'):bulk.receive(t,'output',files,tmp_path/'drop',int(time.time())+30)
    assert (tmp_path/'drop/stream.partial').read_bytes()==b'a'*32
    assert not(tmp_path/'drop/transfer.json').exists()


def test_interrupted_storage_stop_is_restored_with_original_identity_before_provider_read(tmp_path,monkeypatch):
    f=RentalFake(tmp_path);b=selected(tmp_path,policy(tmp_path));original=controller.write_json;started=f.now
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=(120 if not f.alive else 31)*G))
    def crash(path,value):
        if Path(path).name=='stop-request.json':raise KeyboardInterrupt('injected crash after journal')
        original(path,value)
    monkeypatch.setattr(controller,'write_json',crash)
    with pytest.raises(KeyboardInterrupt):run(f,b)
    assert f.alive and not(f.directory/'stop-request.json').exists()
    monkeypatch.setattr(controller,'write_json',original);account=f.account
    def checked():
        stop=read_json(f.directory/'stop-request.json')
        assert stop['observed_epoch']==started and stop['intent_sha256']==digest(f.value)
        return account()
    f.account=checked;run(f,b)
    assert f.writes==1 and not f.alive
    assert next(x for x in f.calls if x[0]=='terminate')[2]==started+f.i['plan']['input']['checkpoint_grace_seconds']


@pytest.mark.parametrize('mutation',['pod','intent','timestamp','duplicate'])
def test_stop_recovery_rejects_conflicting_identity_and_time(tmp_path,mutation):
    event={'kind':'decision','body':{'action':'CHECKPOINT_AND_STOP','reasons':['local-storage-headroom'],'pod_id':'fixture','observed_epoch':10}}
    events=[event];expected='a'*64
    controller.restore_storage_stop(tmp_path,events,expected,'fixture')
    v=read_json(tmp_path/'stop-request.json')
    if mutation=='pod':v['pod_id']='other'
    elif mutation=='intent':v['intent_sha256']='b'*64
    elif mutation=='timestamp':v['observed_epoch']=11
    else:events.append(event)
    write_json(tmp_path/'stop-request.json',v)
    with pytest.raises(EvidenceError):controller.restore_storage_stop(tmp_path,events,expected,'fixture')
    assert read_json(tmp_path/'stop-request.json')==v


def test_budget_covers_actual_destinations_and_rejects_other_root(tmp_path):
    b=selected(tmp_path,policy(tmp_path));b.covers(tmp_path/'future/object-store')
    with pytest.raises(EvidenceError,match='outside'):b.covers(tmp_path.parent/'elsewhere')
    alias=tmp_path/'escape';alias.symlink_to(tmp_path.parent,target_is_directory=True)
    with pytest.raises(EvidenceError,match='symlink'):b.covers(alias/'future')


def test_telemetry_enospc_does_not_skip_durable_low_space_stop(tmp_path,monkeypatch):
    f=RentalFake(tmp_path);b=selected(tmp_path,policy(tmp_path));original=controller.write_json;initial=f.now
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=(120 if not f.alive else 31)*G))
    def no_telemetry(path,value):
        if Path(path).name=='local-storage-observation.json':raise OSError(28,'synthetic telemetry failure')
        original(path,value)
    monkeypatch.setattr(controller,'write_json',no_telemetry);run(f,b)
    assert read_json(f.directory/'stop-request.json')['observed_epoch']==initial
    assert next(x for x in f.calls if x[0]=='terminate')[2]==initial+f.i['plan']['input']['checkpoint_grace_seconds']
    assert not f.alive


@pytest.mark.parametrize('change',['omitted','different'])
def test_live_controller_restart_cannot_drop_or_change_storage_policy(tmp_path,monkeypatch,change):
    f=RentalFake(tmp_path);b=selected(tmp_path,policy(tmp_path));f.create_crash=True
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=120*G))
    with pytest.raises(KeyboardInterrupt):run(f,b)
    assert f.alive and f.writes==1
    if change=='different':
        v=policy(tmp_path);v['peak_incremental_bytes']+=G;chosen=selected(tmp_path,v)
    else:chosen=None
    with pytest.raises(EvidenceError,match='storage selection differs'):run(f,chosen)
    assert f.writes==1


def test_budget_rejects_dotdot_escape_and_existing_file_through_symlink(tmp_path):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'outside';outside.mkdir()
    existing=outside/'file';existing.write_bytes(b'synthetic existing file')
    b=selected(root,policy(root))
    with pytest.raises(EvidenceError,match='outside'):b.covers(root/'..'/'outside'/'file')
    (root/'alias').symlink_to(outside,target_is_directory=True)
    with pytest.raises(EvidenceError,match='symlink'):b.covers(root/'alias'/'file')


def test_missing_sidecar_cannot_erase_journaled_storage_binding(tmp_path,monkeypatch):
    f=RentalFake(tmp_path);b=selected(tmp_path,policy(tmp_path));f.create_crash=True
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=120*G))
    with pytest.raises(KeyboardInterrupt):run(f,b)
    sidecar=f.directory/'local-storage-selection.json';sidecar.rename(sidecar.with_suffix('.preserved'))
    with pytest.raises(EvidenceError,match='storage selection differs'):run(f,None)
    assert f.writes==1
    run(f,b)
    assert read_json(sidecar)['budget_sha256']==digest(b.value) and not f.alive


def test_expired_recovered_stop_terminates_before_blocking_provider_read(tmp_path,monkeypatch):
    f=RentalFake(tmp_path);b=selected(tmp_path,policy(tmp_path));original=controller.write_json;started=f.now
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda _:SimpleNamespace(free=(120 if not f.alive else 31)*G))
    def crash(path,value):
        if Path(path).name=='stop-request.json':raise KeyboardInterrupt('synthetic crash')
        original(path,value)
    monkeypatch.setattr(controller,'write_json',crash)
    with pytest.raises(KeyboardInterrupt):run(f,b)
    monkeypatch.setattr(controller,'write_json',original)
    grace=f.i['plan']['input']['checkpoint_grace_seconds'];f.now+=grace;f.elapsed+=grace
    account=f.account
    def delayed():
        assert not f.alive,'termination must precede the provider read at original grace limit'
        f.now+=2;f.elapsed+=2
        return account()
    f.account=delayed;run(f,b)
    assert next(x for x in f.calls if x[0]=='terminate')[2]==started+grace
