"""Reuse actual retained tiny states; provider/signing fixtures are explicit doubles."""
from pathlib import Path
import os
import time
import pytest
import pod_checkpoint_handoff as m
from test_pipeline import prepared
from test_pod_checkpoint_handoff import paused
from test_production_boundary_poll import fixture
from ovl_pipeline.canonical import EvidenceError,read_json,write_json,digest
from pod_versioned_export import export


def prime(t,remote,store,tmp_path):
    name=read_json(remote/'awaiting-anchor.json')['checkpoint_path']
    return export(t,name,store,tmp_path/'first-retention',int(time.time())+60)


def forbid_payload(*a,**kw):
    raise AssertionError('a retained publication snapshot must not transfer payload again')


def test_rechecks_full_retained_bytes_and_remote_inventory_without_second_transfer(prepared,tmp_path):
    t,remote,r,root,chain,calls=paused(prepared,tmp_path);store=tmp_path/'store'
    first=prime(t,remote,store,tmp_path);t.get=forbid_payload
    out=tmp_path/'publication';receipt=m.snapshot(t,r,root,out,int(time.time())+60,retained_store=store)
    assert receipt['result']=='PASS' and receipt['transfers']==[]
    assert receipt['training_replay']=='NOT_RUN' and receipt['files']==first['files']
    reuse=read_json(out/'retained/export.json')
    assert reuse['bounds']['maximum_uncached_bytes']==0 and reuse['transfers']==[]
    for item in receipt['files']:
        assert not (out/receipt['checkpoint_path']/item['path']).samefile(store/'objects'/item['sha256'])
    from publication_export_gate import exact_tree
    exact_tree(out/receipt['checkpoint_path'],receipt['files'])
    assert m.state_check(r,root,out)[2]['checkpoint']==receipt['checkpoint']


@pytest.mark.parametrize('damage',['missing','corrupt','symlink','remote-extra','remote-altered','remote-missing','waiting-race','store-symlink'])
def test_bad_reuse_fails_closed_without_cold_transfer_or_export_credit(prepared,tmp_path,damage,monkeypatch):
    t,remote,r,root,chain,calls=paused(prepared,tmp_path);store=tmp_path/'store'
    first=prime(t,remote,store,tmp_path);item=next(i for i in first['files'] if i['path']=='state.safetensors');obj=store/'objects'/item['sha256']
    t.get=forbid_payload
    if damage=='missing':obj.unlink()
    elif damage=='corrupt':obj.chmod(0o600);obj.write_bytes(b'altered')
    elif damage=='symlink':obj.unlink();obj.symlink_to(remote/'boundary-00000/state.safetensors')
    elif damage=='remote-extra':(remote/'boundary-00000/unselected.json').write_bytes(b'{}')
    elif damage=='remote-altered':(remote/'boundary-00000/state.safetensors').write_bytes(b'altered')
    elif damage=='remote-missing':(remote/'boundary-00000/state.safetensors').unlink()
    elif damage=='store-symlink':
        alias=tmp_path/'alias';alias.symlink_to(store,target_is_directory=True);store=alias
    else:
        import pod_versioned_export
        original=pod_versioned_export.export
        def changing(*a,**kw):
            result=original(*a,**kw);v=read_json(remote/'awaiting-anchor.json');v['index']+=1;write_json(remote/'awaiting-anchor.json',v);return result
        monkeypatch.setattr(pod_versioned_export,'export',changing)
    out=tmp_path/'publication'
    with pytest.raises(EvidenceError):m.snapshot(t,r,root,out,int(time.time())+60,retained_store=store)
    assert not(out/'export.json').exists()


def test_publisher_reuses_store_and_restart_adopts_original_deadline(prepared,tmp_path,monkeypatch):
    store=tmp_path/'store';hook,remote,out,h,starts,provider=fixture(prepared,tmp_path,monkeypatch,retained_store=store)
    first=hook();prime(first.transport,remote,store,tmp_path);first.transport.get=forbid_payload
    from production_live_retention import record_selection
    selection=record_selection(first.registration,first.root,read_json(remote/'chain.json'))
    result=first.poll(retained_selection=selection);deadline=read_json(out/'boundaries/boundary-00000/intent.json')
    assert result['index']==0 and len(starts)==1
    assert hook().poll(retained_selection=selection)==result and len(starts)==1
    assert read_json(out/'boundaries/boundary-00000/intent.json')==deadline
    assert read_json(out/'selection.json')['retained_store']==str(store)
    assert read_json(out/'boundaries/boundary-00000/snapshot/export.json')['transfers']==[]
    import production_boundary_poll
    with pytest.raises(EvidenceError):
        production_boundary_poll.BoundaryPublisher(first.registration,first.job,h,first.health_file,first.transport,out,first.arguments,first.policy,retained_store=tmp_path/'changed')


@pytest.mark.parametrize('corrupt_download',[False,True])
def test_reused_snapshot_runs_publication_and_complete_download_checks(prepared,tmp_path,monkeypatch,corrupt_download):
    import production_boundary_poll as boundary
    import publish_progress_boundary as public
    import publication_export_gate as gate
    from production_live_retention import record_selection
    store=tmp_path/'store';hook,remote,out,h,starts,provider=fixture(prepared,tmp_path,monkeypatch,retained_store=store)
    current=hook();prime(current.transport,remote,store,tmp_path);current.transport.get=forbid_payload
    selection=record_selection(current.registration,current.root,read_json(remote/'chain.json'))
    # The reference fixture only supplies technical parents. Exercise a new,
    # empty in-memory provider through the actual publisher and downloader.
    provider.files.clear();provider.commits=0;provider.sha='2'*40
    def review(plan_path,staging,**kwargs):
        gate.exact_tree(staging,read_json(plan_path)['files'])
        return {'synthetic-semantic-review-double':True,'plan_sha256':digest(read_json(plan_path))}
    monkeypatch.setattr(gate,'require_review',review)
    original_request=public.request_commit
    def request(value,registration,directory,**kwargs):
        revision=original_request(value,registration,directory,**kwargs)
        Path(directory).mkdir(parents=True,exist_ok=True);write_json(Path(directory)/'public-commit.json',{'revision':revision})
        return revision
    monkeypatch.setattr(public,'request_commit',request)
    fetched=[];original_fetch=provider.fetch
    def fetch(**kw):
        file=original_fetch(**kw);fetched.append(kw['filename'])
        if corrupt_download and kw['filename'].endswith('state.safetensors'):file.write_bytes(b'altered-download')
        return file
    provider.fetch=fetch
    def start(spec,expected,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);write_json(directory/'selection.json',spec)
        boundary.publisher.worker(spec,expected,directory)
        return {'observation':{'ActiveState':'inactive'}}
    monkeypatch.setattr(boundary.publisher,'start_or_adopt',start)
    if corrupt_download:
        with pytest.raises(EvidenceError):current.poll(retained_selection=selection)
        assert not (out/'publications/boundary-00000/ack.json').exists()
        assert not (remote/'external-progress-policies.json').exists()
    else:
        result=current.poll(retained_selection=selection)
        assert result['index']==0 and provider.commits==2
        ack=read_json(out/'publications/boundary-00000/ack.json')
        assert ack['checkpoint_download']['result']=='PASS' and ack['anchor_download']['result']=='PASS'
        assert len(read_json(remote/'external-progress-policies.json'))==1
    assert any(name.endswith('state.safetensors') for name in fetched)
