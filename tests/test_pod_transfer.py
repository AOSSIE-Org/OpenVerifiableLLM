"""Real bounded subprocess/file transfers with an explicit local SSH endpoint double."""
from pathlib import Path
import io
import os
import shlex
import subprocess
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_transfer as m
from ovl_pipeline.canonical import EvidenceError,file_hash,read_json,sha256


def setup(tmp_path,*,fault=None):
    key=tmp_path/'key';key.write_bytes(b'explicit noncredential test key');key.chmod(0o600)
    known=tmp_path/'known-hosts';known.write_bytes(b'explicit non-hostkey test file');known.chmod(0o600)
    profile={'schema':'ovl.pod-ssh-profile.v1','pod_id':'fixture-only','host':'127.0.0.1','port':2222,'user':'root',
             'remote_root':'/workspace/ovllm/fixture','endpoint_observation_sha256':'a'*64,
             'known_hosts_sha256':file_hash(known),'host_key_trust':'operator-pinned-TOFU'}
    remote=tmp_path/'remote';remote.mkdir();calls=[];processes=[]
    def popen(command,**kw):
        calls.append(command);assert command[0]=='ssh' and command[1:3]==['-F','/dev/null']
        for expected in ('StrictHostKeyChecking=yes','IdentityAgent=none','ForwardAgent=no','BatchMode=yes','ControlMaster=no','ClearAllForwardings=yes'):
            assert expected in command
        assert kw['start_new_session'] is True
        argv=shlex.split(command[-1]);assert argv.pop(0)=='exec'
        if fault=='hang':argv=[sys.executable,'-c','import time;time.sleep(15)']
        elif fault=='stderr-flood':argv=[sys.executable,'-c','import os;os.write(2,b"x"*131072)']
        elif argv[0]=='/bin/cat':argv[-1]=str(remote/argv[-1].removeprefix(profile['remote_root']+'/'))
        else:
            from pod_job_client import REMOTE_TREE
            from pod_bulk_export import REMOTE_BULK
            from pod_observe import REMOTE_OBSERVE
            assert argv[:2]==['/usr/bin/python3','-c'] and argv[2] in (m.REMOTE_PUT,m.REMOTE_GET,REMOTE_TREE,REMOTE_BULK,REMOTE_OBSERVE)
            argv[0]=sys.executable;assert argv[3]==profile['remote_root'];argv[3]=str(remote)
        p=subprocess.Popen(argv,**kw);processes.append(p);return p
    return m.Transport(profile,key,known,popen=popen),remote,calls,processes


def test_full_upload_download_and_verified_immutable_adoption(tmp_path):
    t,remote,calls,processes=setup(tmp_path);source=tmp_path/'state';data=b'actual complete transfer\0'*70000;source.write_bytes(data)
    updates=[];deadline=int(time.time())+30
    uploaded=t.put('boundary-00000/state.safetensors',source,deadline,progress=updates.append)
    assert uploaded['bytes_sent']==len(data) and uploaded['sha256']==sha256(data)
    assert (remote/'boundary-00000/state.safetensors').read_bytes()==data
    assert stat_mode(remote/'boundary-00000/state.safetensors')==0o600
    assert t.put('boundary-00000/state.safetensors',source,deadline)['sha256']==sha256(data)
    expected={'path':'boundary-00000/state.safetensors','bytes':len(data),'sha256':sha256(data)}
    got=t.get(expected['path'],tmp_path/'download',expected,deadline,progress=updates.append)
    assert got['bytes_received']==len(data) and (tmp_path/'download').read_bytes()==data
    assert updates and all(e['bytes_sent']>0 or e['bytes_received']>0 for e in updates)
    assert len(calls)==3 and all(p.returncode==0 for p in processes)
    with pytest.raises(EvidenceError):t.get(expected['path'],tmp_path/'download',expected,deadline)


def stat_mode(path):
    import stat
    return stat.S_IMODE(path.stat().st_mode)


def test_only_policy_or_stop_controls_can_be_atomically_replaced(tmp_path):
    t,remote,calls,processes=setup(tmp_path);source=tmp_path/'control';source.write_bytes(b'[]');deadline=int(time.time())+30
    t.put('external-progress-policies.json',source,deadline);source.write_bytes(b'[1]')
    with pytest.raises(EvidenceError):t.put('external-progress-policies.json',source,deadline)
    assert (remote/'external-progress-policies.json').read_bytes()==b'[]'
    t.put('external-progress-policies.json',source,deadline,replace=True)
    assert (remote/'external-progress-policies.json').read_bytes()==b'[1]'
    before=len(calls)
    with pytest.raises(EvidenceError):t.put('boundary-00000/state.json',source,deadline,replace=True)
    assert len(calls)==before


@pytest.mark.parametrize('damage',['changed-bytes','oversized','short','missing'])
def test_untrusted_download_bytes_preserve_partial_but_never_success(tmp_path,damage):
    t,remote,calls,processes=setup(tmp_path);expected={'path':'state.json','bytes':3,'sha256':sha256(b'abc')}
    if damage!='missing':(remote/'state.json').write_bytes({'changed-bytes':b'bad','oversized':b'extra','short':b'a'}[damage])
    with pytest.raises(EvidenceError):t.get('state.json',tmp_path/'checked',expected,int(time.time())+30)
    assert not(tmp_path/'checked').exists() and (tmp_path/'checked.partial').exists()
    assert stat_mode(tmp_path/'checked.partial')==0o600
    assert all(p.poll() is not None for p in processes)
    partial=(tmp_path/'checked.partial').read_bytes()
    with pytest.raises(EvidenceError,match='fresh destination'):
        t.get('state.json',tmp_path/'checked',expected,int(time.time())+30)
    assert (tmp_path/'checked.partial').read_bytes()==partial


@pytest.mark.parametrize('fault',['hang','stderr-flood'])
def test_bounded_failure_closes_owned_process_and_does_not_renew_deadline(tmp_path,fault):
    t,remote,calls,processes=setup(tmp_path,fault=fault);before=time.monotonic()
    with pytest.raises(EvidenceError):t.stream(['/bin/cat','--','ignored'],io.BytesIO(),10,int(time.time())+1)
    assert time.monotonic()-before<4 and all(p.poll() is not None for p in processes)


@pytest.mark.parametrize('phase',['create','register','close'])
def test_selector_setup_failure_reaps_owned_child(tmp_path,monkeypatch,phase):
    t,remote,calls,processes=setup(tmp_path,fault='hang');closed=[]
    class BrokenSelector:
        def register(self,*args):raise OSError('synthetic registration failure')
        def close(self):
            closed.append(True)
            if phase=='close':raise OSError('synthetic selector close failure')
    def create():
        if phase=='create':raise OSError('synthetic selector construction failure')
        return BrokenSelector()
    monkeypatch.setattr(m.selectors,'DefaultSelector',create)
    with pytest.raises(OSError,match='synthetic'):
        t.stream(['/bin/cat','--','ignored'],io.BytesIO(),10,int(time.time())+30)
    assert len(processes)==1 and processes[0].poll() is not None
    assert processes[0].stdout.closed and processes[0].stderr.closed
    assert closed==([] if phase=='create' else [True])


@pytest.mark.parametrize('diagnostic,code',[(b'Permission denied\n',255),(b'unknown failure\n',255),(b'',1),(b'',255),(b'Connection reset by peer\n',1)])
def test_expiry_drains_waiting_diagnostics_and_checks_available_exit(tmp_path,monkeypatch,diagnostic,code):
    t,remote,calls,processes=setup(tmp_path);clock=[1000];original=m.selectors.DefaultSelector
    def child(command,**kw):
        p=subprocess.Popen([sys.executable,'-c','import os,sys;os.write(2,sys.argv[1].encode());sys.exit(int(sys.argv[2]))',diagnostic.decode(),str(code)],**kw)
        processes.append(p);return p
    class EdgeSelector:
        def __init__(self):self.inner=original()
        def register(self,*a):return self.inner.register(*a)
        def get_map(self):return self.inner.get_map()
        def select(self,*a):
            # Child has finished and stderr is queued, but the caller has not
            # consumed readiness events before its next deadline observation.
            processes[0].wait(timeout=3);clock[0]=1004;return []
        def close(self):self.inner.close()
    t.popen=child;t.wall=lambda:clock[0];t.monotonic=lambda:clock[0]
    monkeypatch.setattr(m.selectors,'DefaultSelector',EdgeSelector)
    with pytest.raises(EvidenceError) as error:t.stream(['/bin/true'],io.BytesIO(),65536,1003)
    assert not isinstance(error.value,m.TransientTransportError)
    assert all(p.poll() is not None and p.stdout.closed and p.stderr.closed for p in processes)


@pytest.mark.parametrize('path',['../seed.key','/root/key','a/../../key','-o ProxyCommand=bad','a;command','a//b','a/./b'])
def test_remote_path_cannot_escape_or_become_a_shell_option(tmp_path,path):
    t,remote,calls,processes=setup(tmp_path);source=tmp_path/'source';source.write_bytes(b'x')
    with pytest.raises(EvidenceError):t.put(path,source,int(time.time())+30)
    assert calls==[]


@pytest.mark.parametrize('damage',['key-mode','key-symlink','hostkey-bytes','profile-extra','hostname-shell','unscoped-root'])
def test_invalid_local_key_or_endpoint_selection_fails_before_network(tmp_path,damage):
    t,remote,calls,processes=setup(tmp_path)
    if damage=='key-mode':t.key.chmod(0o644)
    elif damage=='key-symlink':
        actual=tmp_path/'actual';t.key.rename(actual);t.key.symlink_to(actual)
    elif damage=='hostkey-bytes':t.known.write_bytes(b'changed')
    elif damage=='profile-extra':t.profile['ProxyCommand']='bad'
    elif damage=='hostname-shell':t.profile['host']='host; bad'
    else:t.profile['remote_root']='/workspace/other'
    with pytest.raises(EvidenceError):t.command(['/bin/cat','--','file'])
    assert calls==[]


def test_remote_symlink_cannot_redirect_acknowledgement_or_overwrite_checkpoint(tmp_path):
    t,remote,calls,processes=setup(tmp_path);outside=tmp_path/'preserved';outside.write_bytes(b'unchanged')
    (remote/'request-stop').symlink_to(outside);source=tmp_path/'source';source.write_bytes(b'stop')
    with pytest.raises(EvidenceError):t.put('request-stop',source,int(time.time())+30,replace=True)
    assert outside.read_bytes()==b'unchanged'
    (remote/'bad-parent').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(EvidenceError):t.put('bad-parent/preserved',source,int(time.time())+30)
    assert outside.read_bytes()==b'unchanged'



def test_even_matching_bytes_cannot_be_fetched_through_remote_symlink(tmp_path):
    t,remote,calls,processes=setup(tmp_path);outside=tmp_path/'private';outside.write_bytes(b'synthetic-private')
    (remote/'state.json').symlink_to(outside)
    expected={'path':'state.json','bytes':outside.stat().st_size,'sha256':file_hash(outside)}
    with pytest.raises(EvidenceError):t.get('state.json',tmp_path/'download',expected,int(time.time())+30)
    assert not(tmp_path/'download').exists() and (tmp_path/'download.partial').stat().st_size==0



def test_bounded_live_json_inventory_is_only_a_peer_observation(tmp_path):
    t,remote,calls,processes=setup(tmp_path);deadline=int(time.time())+30
    assert t.inspect('awaiting-anchor.json',4096,deadline) is None
    data=b'{"test":true}';(remote/'awaiting-anchor.json').write_bytes(data)
    expected=t.inspect('awaiting-anchor.json',4096,deadline)
    assert expected=={'path':'awaiting-anchor.json','bytes':len(data),'sha256':sha256(data)}
    t.get('awaiting-anchor.json',tmp_path/'observed.json',expected,deadline)
    assert read_json(tmp_path/'observed.json')=={'test':True}
    with pytest.raises(EvidenceError):t.inspect('awaiting-anchor.json',2,deadline)
