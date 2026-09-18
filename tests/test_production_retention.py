"""Actual tiny signed CPU files; terminal worker and SSH substitutes are explicit."""
from pathlib import Path
import shutil,sys,time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import production_retention as m
from pod_job_client import launch
from test_pod_job_client import fixture
from test_pod_job_worker import exited
from test_pod_transfer import setup as ssh
from test_pipeline import prepared
from test_production_chain import actual_artifacts
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json


def terminal_fixture(prepared,tmp_path,*,with_registration=False):
    control_dir=tmp_path/'control';control_dir.mkdir()
    t,remote,calls,job,root,worker,worker_sha=fixture(control_dir)
    record_dir=tmp_path/'record';record_dir.mkdir();record,record_remote,record_calls,processes=ssh(record_dir)
    record.profile['remote_root']=t.profile['remote_root']+'-record'
    numerical=tmp_path/'numerical';numerical.mkdir()
    r,registration,envelopes,key,chain,streams=actual_artifacts(prepared,numerical)
    shutil.copytree(chain,record_remote,dirs_exist_ok=True)
    write_json(record_remote/'chain.json',{'schema':'ovl.production-chain.v1','complete':False,'boundaries':envelopes})
    body=envelopes[-1]['body']
    write_json(record_remote/'awaiting-anchor.json',{'schema':'ovl.awaiting-public-progress.v1','registration_sha256':registration,
               'index':body['index'],'boundary_sha256':digest(envelopes[-1]),'checkpoint_path':body['checkpoint_path'],'checkpoint':body['checkpoint']})
    # Preserve an unfinished recovery as bytes too; it cannot count as valid state.
    partial=record_remote/'recovery-partial';partial.mkdir();(partial/'state.safetensors').write_bytes(b'explicit incomplete sole recovery')
    value=read_json(job);value['kind']='production-record';value['export_roots']=[record.profile['remote_root']]
    write_json(job,value);root=digest(value)
    launch(t,job,root,worker,worker_sha,tmp_path/'launch',int(time.time())+30)
    assert exited(remote/'jobs'/root)['exit_code']==0
    result=t,record,record_remote,job,root,worker_sha
    return (*result,r) if with_registration else result


def test_terminal_retention_covers_complete_declared_tree_and_partial_states(prepared,tmp_path):
    t,record,remote,job,root,worker=terminal_fixture(prepared,tmp_path)
    result=m.retain(t,[record],job,root,worker,tmp_path/'objects',tmp_path/'retention',int(time.time())+60)
    receipts=m.verify(result,t,[record],job,root,worker)
    actual=receipts[1];names={f['path'] for f in actual['files']}
    expected={p.relative_to(remote).as_posix() for p in remote.rglob('*') if p.is_file()}
    assert names==expected and 'recovery-partial/state.safetensors' in names
    assert result['training_replay']=='NOT_RUN' and len(result['roots'])==2
    assert read_json(Path(receipts[0]['files_directory'])/'exit.json')['job_sha256']==root
    path=Path(actual['files_directory'])/'recovery-partial/state.safetensors';assert path.read_bytes()==b'explicit incomplete sole recovery'
    path.chmod(0o600);path.write_bytes(b'changed after export')
    with pytest.raises(EvidenceError):m.verify(result,t,[record],job,root,worker)


@pytest.mark.parametrize('damage',['omit-record','swap-root','foreign-profile'])
def test_logs_only_or_misbound_retention_refused(prepared,tmp_path,damage):
    t,record,remote,job,root,worker=terminal_fixture(prepared,tmp_path)
    result=m.retain(t,[record],job,root,worker,tmp_path/'objects',tmp_path/'retention',int(time.time())+60)
    if damage=='omit-record':result['roots']=result['roots'][:1]
    elif damage=='swap-root':result['roots'].reverse()
    else:record.profile['pod_id']='another-pod'
    with pytest.raises(EvidenceError):m.verify(result,t,[record],job,root,worker)
