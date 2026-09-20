"""Actual detached worker and complete transfers behind an explicit local SSH double."""
from pathlib import Path
import io
import shlex
import subprocess
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_job_client as m
import pod_job_worker as worker
from test_pod_transfer import setup as ssh
from test_pod_job_worker import exited
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,inventory,read_json,write_json


def fixture(tmp_path):
    transport,remote,calls,processes=ssh(tmp_path);original=transport.popen
    def popen(command,**kw):
        argv=shlex.split(command[-1]);assert argv.pop(0)=='exec'
        if len(argv)>2 and argv[1]=='-c' and argv[2]==m.REMOTE_TREE:
            calls.append(command);argv[0]=sys.executable;argv[3]=str(remote)
        elif len(argv)>1 and argv[1].startswith(transport.profile['remote_root']+'/tools/pod-job-worker-'):
            calls.append(command);argv[0]=sys.executable
            argv=[a.replace(transport.profile['remote_root'],str(remote)) for a in argv]
        else:return original(command,**kw)
        p=subprocess.Popen(argv,**kw);processes.append(p);return p
    transport.popen=popen
    program=tmp_path/'program.py';program.write_text('print("owned synthetic workload")\n')
    executable=Path(sys.executable).resolve()
    job={'schema':'ovl.pod-job.v1','kind':'pilot','argv':[str(executable),str(program)],'cwd':str(tmp_path),
         'environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},'deadline_epoch':int(time.time())+30,
         'stop_grace_seconds':1,'minimum_free_bytes':1,
         'required_files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in [executable,program]],
         'export_roots':[str(remote)]}
    job_file=tmp_path/'job.json';write_json(job_file,job)
    return transport,remote,calls,job_file,digest(job),Path(worker.__file__),file_hash(Path(worker.__file__))


def test_actual_one_shot_launch_readonly_adoption_and_complete_export(tmp_path):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);deadline=int(time.time())+30
    a=m.launch(t,job,root,source,worker_root,tmp_path/'launch',deadline)
    assert a['job_sha256']==root and a['reissued_start'] is False
    assert exited(remote/'jobs'/root)['exit_code']==0
    before=len(calls);b=m.launch(t,job,root,source,worker_root,tmp_path/'launch',deadline)
    assert b['job_sha256']==root and not any("'start'" in c[-1] or ' start ' in c[-1] for c in calls[before:])
    updates=[];result=m.export_tree(t,'jobs/'+root,tmp_path/'export',deadline,progress=lambda op,counts,total:updates.append((op,counts)))
    assert result['result']=='PASS' and any(f['bytes']==0 for f in result['files'])
    assert read_json(tmp_path/'export/files/exit.json')['job_sha256']==root
    assert (tmp_path/'export/files/stdout.log').read_text().strip()=='owned synthetic workload'
    assert updates and all(len(op)==64 for op,c in updates)


def test_lost_start_reply_reconciles_without_launching_again(tmp_path):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);original=t.stream;failed=[False]
    def lost(argv,*args,**kwargs):
        value=original(argv,*args,**kwargs)
        if 'start' in argv and not failed[0]:failed[0]=True;raise EvidenceError('explicit lost SSH start reply')
        return value
    t.stream=lost;deadline=int(time.time())+30
    with pytest.raises(EvidenceError,match='lost SSH'):m.launch(t,job,root,source,worker_root,tmp_path/'launch',deadline)
    assert exited(remote/'jobs'/root)['exit_code']==0
    t.stream=original;before=len(calls);a=m.launch(t,job,root,source,worker_root,tmp_path/'launch',deadline)
    assert a['job_sha256']==root and not any(' start ' in c[-1] for c in calls[before:])


def test_fenced_but_unsent_launch_is_never_automatically_reissued(tmp_path):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path)
    out=tmp_path/'launch';out.mkdir()
    write_json(out/'launch-intent.json',{'schema':'ovl.offpod-job-launch-intent.v1','job_sha256':root,'worker_sha256':worker_root,'profile_sha256':digest(t.profile)})
    with pytest.raises(EvidenceError,match='never automatically restart'):
        m.launch(t,job,root,source,worker_root,out,int(time.time())+30)
    assert not(remote/'jobs'/root/'launch').exists() and not any(' start ' in c[-1] for c in calls)


@pytest.mark.parametrize('damage',['changing-tree','symlink','extra-after-transfer'])
def test_changed_or_unsafe_full_tree_never_gets_export_receipt(tmp_path,damage):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path)
    tree=remote/'output';tree.mkdir();(tree/'one').write_bytes(b'original')
    if damage=='symlink':(tree/'link').symlink_to(tmp_path/'program.py')
    else:
        original=t.get
        def get(*args,**kwargs):
            result=original(*args,**kwargs)
            if damage=='changing-tree':(tree/'one').write_bytes(b'mutated')
            else:(tree/'extra').write_bytes(b'new')
            return result
        t.get=get
    with pytest.raises(EvidenceError):m.export_tree(t,'output',tmp_path/'export',int(time.time())+30)
    assert not(tmp_path/'export/export.json').exists()


def test_zero_length_regular_files_transfer_but_extra_bytes_fail(tmp_path):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);empty=tmp_path/'empty';empty.write_bytes(b'')
    deadline=int(time.time())+30;t.put('empty.log',empty,deadline)
    item=t.inspect('empty.log',100,deadline);assert item['bytes']==0
    t.get('empty.log',tmp_path/'downloaded',item,deadline);assert (tmp_path/'downloaded').read_bytes()==b''
    (remote/'empty.log').write_bytes(b'x')
    with pytest.raises(EvidenceError):t.get('empty.log',tmp_path/'wrong',item,deadline)


def test_mutable_observation_uses_one_bounded_read_not_inspect_fetch(tmp_path,monkeypatch):
    from pod_checkpoint_handoff import observe
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path)
    write_json(remote/'status.json',{'state':'RUNNING','text':'mutable UTF-8 ☃'})
    def forbidden(*a,**k):raise AssertionError('mutable status cannot use inspect then fetch')
    monkeypatch.setattr(t,'inspect',forbidden);monkeypatch.setattr(t,'get',forbidden)
    deadline=int(time.time())+30
    assert observe(t,'status.json',tmp_path/'first.json',4096,deadline)=={'state':'RUNNING','text':'mutable UTF-8 ☃'}
    write_json(remote/'status.json',{'state':'EXITED'})
    assert observe(t,'status.json',tmp_path/'second.json',4096,deadline)=={'state':'EXITED'}
    assert observe(t,'absent.json',tmp_path/'absent.json',4096,deadline,optional=True) is None
    assert not(tmp_path/'absent.json').exists()
    with pytest.raises(EvidenceError):t.read_live('status.json',2,deadline)
    (remote/'link.json').symlink_to(remote/'status.json')
    with pytest.raises(EvidenceError):t.read_live('link.json',4096,deadline)


@pytest.mark.parametrize('batched',[False,True])
def test_launch_reconciliation_roundtrips_fit_same_deadline(tmp_path,monkeypatch,batched):
    """Actual worker/receipt checks with explicit nine-second transport latency.

    The previous three-read topology exceeds60seconds; batching finishes in45.
    This is a deterministic deadline regression, not a network throughput claim.
    """
    from pod_checkpoint_handoff import observe
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path)
    original=t.stream;offset=[0];base=int(time.time());deadlines=[]
    wall=t.wall;monotonic=t.monotonic
    t.wall=lambda:wall()+offset[0];t.monotonic=lambda:monotonic()+offset[0]
    def delayed(argv,destination,maximum,deadline,**kwargs):
        deadlines.append(deadline);offset[0]+=9
        return original(argv,destination,maximum,deadline,**kwargs)
    t.stream=delayed
    if not batched:
        def separate(transport,selection,maximum,deadline):
            return {name:observe(transport,name,path,maximum,deadline,optional=True) for name,path in selection.items()}
        monkeypatch.setattr(m,'observe_many',separate)
    out=tmp_path/'launch';deadline=base+60
    if batched:
        result=m.launch(t,job,root,source,worker_root,out,deadline)
        assert result['reissued_start'] is False and len(deadlines)==5 and offset[0]==45
    else:
        with pytest.raises(EvidenceError,match='deadline'):
            m.launch(t,job,root,source,worker_root,out,deadline)
        assert len(deadlines)==7 and offset[0]==63
        assert not list(out.glob('reconcile-*/adoption.json'))
    assert set(deadlines)=={deadline}
    assert exited(remote/'jobs'/root)['exit_code']==0
    assert len([c for c in calls if ' start ' in c[-1]])==1


@pytest.mark.parametrize('damage',['intent-job','intent-worker','receipt-job','receipt-worker','child-job','missing-intent','missing-receipts'])
def test_batched_launch_receipts_reject_identity_damage_before_supervision(tmp_path,damage):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);out=tmp_path/'launch';deadline=int(time.time())+30
    m.launch(t,job,root,source,worker_root,out,deadline);assert exited(remote/'jobs'/root)['exit_code']==0
    intent=read_json(out/'launch-intent.json');records=remote/'jobs'/root/'launch'
    if damage=='missing-intent':(records/'intent.json').unlink()
    elif damage=='missing-receipts':
        (records/'receipt.json').unlink();(records/'child.json').unlink()
    else:
        name,field=damage.split('-');path=records/(name+'.json');value=read_json(path)
        value['job_sha256' if field=='job' else 'worker_sha256']='0'*64;write_json(path,value)
    before=len(calls)
    with pytest.raises(EvidenceError):m.reconcile_launch(t,intent,out,deadline)
    assert len(calls)-before==1
    assert not any(' start ' in c[-1] or ' inspect ' in c[-1] for c in calls[before:])


@pytest.mark.parametrize('damage',['job','worker','runner','shape'])
def test_contradictory_start_reply_is_retained_and_never_overridden_by_good_remote_receipts(tmp_path,damage):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);original=t.stream
    def corrupt(argv,destination,*args,**kwargs):
        value=original(argv,destination,*args,**kwargs)
        if 'start' in argv:
            response=m.parse_json(destination.getvalue(),canonical_required=False)
            if damage in ('job','worker'):response[damage+'_sha256']='0'*64
            elif damage=='runner':response['runner']['start_ticks']+=1
            else:response['runner']['pid']=True
            destination.seek(0);destination.truncate();destination.write(worker.encoded(response))
        return value
    t.stream=corrupt;out=tmp_path/'launch';deadline=int(time.time())+30
    with pytest.raises(EvidenceError):m.launch(t,job,root,source,worker_root,out,deadline)
    assert exited(remote/'jobs'/root)['exit_code']==0
    retained=(out/'start-response.json').read_bytes();before=len(calls)
    with pytest.raises(EvidenceError):m.launch(t,job,root,source,worker_root,out,deadline)
    assert (out/'start-response.json').read_bytes()==retained
    assert not any(' start ' in c[-1] for c in calls[before:])
    assert not list(out.glob('reconcile-*/adoption.json'))


def test_uncertain_immutable_stop_delivery_preserves_existing_bytes_and_rejects_change(tmp_path):
    from pod_transfer import TransientTransportError
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);marker=tmp_path/'stop';marker.write_bytes(b'synthetic stop\n')
    original=t.stream;lost=[False];deadline=int(time.time())+30
    def lose_ack(*args,**kwargs):
        result=original(*args,**kwargs)
        if not lost[0]:lost[0]=True;raise TransientTransportError('synthetic lost stop acknowledgement')
        return result
    t.stream=lose_ack
    with pytest.raises(TransientTransportError):t.put('jobs/'+root+'/request-stop',marker,deadline)
    target=remote/'jobs'/root/'request-stop';before=target.stat()
    t.put('jobs/'+root+'/request-stop',marker,deadline)
    after=target.stat();assert (before.st_ino,before.st_mtime_ns)==(after.st_ino,after.st_mtime_ns)
    assert target.read_bytes()==marker.read_bytes()
    marker.write_bytes(b'changed stop\n')
    with pytest.raises(EvidenceError):t.put('jobs/'+root+'/request-stop',marker,deadline)
    assert target.read_bytes()==b'synthetic stop\n' and not any(' start ' in c[-1] for c in calls)


def test_stop_arriving_during_fence_persistence_prevents_start_and_preserves_fence(tmp_path,monkeypatch):
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);stopped=[False];original=m.save_once;out=tmp_path/'launch'
    def save(path,value):
        original(path,value)
        if path==out/'launch-intent.json':stopped[0]=True
    monkeypatch.setattr(m,'save_once',save)
    def eligible():
        if stopped[0]:raise EvidenceError('synthetic stop during persistence')
    with pytest.raises(EvidenceError,match='stop during persistence'):
        m.launch(t,job,root,source,worker_root,out,int(time.time())+30,before_start=eligible)
    assert (out/'launch-intent.json').exists() and not any(' start ' in c[-1] for c in calls)
    with pytest.raises(EvidenceError,match='never automatically restart'):
        m.launch(t,job,root,source,worker_root,out,int(time.time())+30,before_start=eligible)
    assert not any(' start ' in c[-1] for c in calls)


@pytest.mark.parametrize('damage',['wrong-job','malformed'])
def test_received_bad_reply_survives_subsequent_transport_failure(tmp_path,damage):
    from pod_transfer import TransientTransportError
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);original=t.stream;received=[]
    def lost(argv,destination,*args,**kwargs):
        value=original(argv,destination,*args,**kwargs)
        if 'start' in argv:
            if damage=='wrong-job':
                response=m.parse_json(destination.getvalue(),canonical_required=False);response['job_sha256']='0'*64
                raw=worker.encoded(response)
            else:raw=b'{malformed received acknowledgement'
            destination.seek(0);destination.truncate();destination.write(raw);received.append(raw)
            raise TransientTransportError('synthetic failure after received stdout')
        return value
    t.stream=lost;out=tmp_path/'launch';deadline=int(time.time())+30
    with pytest.raises(TransientTransportError):m.launch(t,job,root,source,worker_root,out,deadline)
    assert exited(remote/'jobs'/root)['exit_code']==0
    assert (out/'start-response.raw').read_bytes()==received[0] and not read_json(out/'start-transport.json')['completed']
    before=len(calls)
    with pytest.raises(EvidenceError):m.launch(t,job,root,source,worker_root,out,deadline)
    assert len(calls)==before and not list(out.glob('reconcile-*/adoption.json'))


@pytest.mark.parametrize('failed_transport',[False,True])
def test_empty_reply_distinguishes_uncertain_transport_from_completed_invalid_ack(tmp_path,failed_transport):
    from pod_transfer import TransientTransportError
    t,remote,calls,job,root,source,worker_root=fixture(tmp_path);original=t.stream
    def empty(argv,destination,*args,**kwargs):
        result=original(argv,destination,*args,**kwargs)
        if 'start' in argv:
            destination.seek(0);destination.truncate()
            if failed_transport:raise TransientTransportError('synthetic absent acknowledgement')
        return result
    t.stream=empty;out=tmp_path/'launch';deadline=int(time.time())+30
    with pytest.raises(EvidenceError):m.launch(t,job,root,source,worker_root,out,deadline)
    assert exited(remote/'jobs'/root)['exit_code']==0
    before=len(calls)
    if failed_transport:assert m.launch(t,job,root,source,worker_root,out,deadline)['reissued_start'] is False
    else:
        with pytest.raises(EvidenceError,match='lacks acknowledgement'):m.launch(t,job,root,source,worker_root,out,deadline)
    assert not any(' start ' in c[-1] for c in calls[before:])
