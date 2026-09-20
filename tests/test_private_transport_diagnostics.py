"""Private failure evidence must not relax transport gates or export raw data."""
import base64
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import pytest
from ovl_pipeline.canonical import EvidenceError
from test_pod_transfer import setup
from test_pod_observation_retry import Health
import pod_observation_retry as reads
import private_transport_diagnostics as diagnostics
from pod_transfer import TransientTransportError


def records(path):return [json.loads(p.read_text()) for p in path.glob('*.json')]


@pytest.mark.parametrize('diagnostic,code,transient',[
    ('Permission denied',255,False),('unknown synthetic remote refusal',1,False),
    ('Connection reset by peer',255,True)])
def test_failed_process_diagnostics_private_bounded_and_classification_unchanged(tmp_path,diagnostic,code,transient,capsys):
    t,remote,calls,processes=setup(tmp_path);attempts=[]
    def fail(command,**kw):
        attempts.append(command)
        p=subprocess.Popen([sys.executable,'-c','import sys;sys.stderr.write(sys.argv[1]);sys.exit(int(sys.argv[2]))',diagnostic,str(code)],**kw)
        processes.append(p);return p
    t.popen=fail;output=tmp_path/'private';health=Health();deadline=health.plan['external_terminate_epoch']
    with pytest.raises(EvidenceError) as caught:
        reads.read('metadata',t,{'status.json':tmp_path/'status.json'},health,tmp_path/'health',tmp_path,
                   sleep=lambda _:None,private_diagnostics=output)
    assert isinstance(caught.value,TransientTransportError)==transient
    assert len(attempts)==(2 if transient else 1) and health.plan['external_terminate_epoch']==deadline
    saved=records(output);assert len(saved)==len(attempts)
    assert all(base64.b64decode(r['transport']['stderr']['prefix_b64']).decode()==diagnostic for r in saved)
    assert diagnostic not in str(caught.value) and capsys.readouterr().out==''
    assert stat.S_IMODE(output.stat().st_mode)==0o700
    assert all(stat.S_IMODE(p.stat().st_mode)==0o600 for p in output.iterdir())
    assert all(p.poll() is not None for p in processes)


@pytest.mark.parametrize('phase',['before','after'])
def test_health_failure_has_distinct_phase_and_never_retries(tmp_path,monkeypatch,phase):
    error=TransientTransportError('synthetic health failure');calls=[]
    class BrokenHealth(Health):
        def write(self,path):
            self.writes+=1
            if self.writes==(1 if phase=='before' else 2):raise error
    monkeypatch.setattr(reads,'job_supervision',lambda *a:calls.append(a) or {'state':'synthetic'})
    with pytest.raises(TransientTransportError) as caught:
        reads.read('supervision',None,('a','b'),BrokenHealth(),tmp_path/'h',tmp_path,
                   private_diagnostics=tmp_path/'private',sleep=lambda _:pytest.fail('health error retried'))
    assert caught.value is error and len(calls)==(0 if phase=='before' else 1)
    assert records(tmp_path/'private')[0]['context']['phase']=='health-'+phase


def test_malformed_complete_metadata_retains_received_bytes_without_retry(tmp_path):
    t,remote,calls,processes=setup(tmp_path);attempts=[]
    def malformed(command,**kw):
        attempts.append(command);p=subprocess.Popen([sys.executable,'-c','print("{not-json")'],**kw);processes.append(p);return p
    t.popen=malformed
    with pytest.raises(EvidenceError):
        reads.read('metadata',t,{'status.json':tmp_path/'status.json'},Health(),tmp_path/'h',tmp_path,
                   private_diagnostics=tmp_path/'private',sleep=lambda _:pytest.fail('framing failure retried'))
    assert len(attempts)==1
    response=records(tmp_path/'private')[0]['metadata_response']
    assert base64.b64decode(response['prefix_b64'])==b'{not-json\n' and not response['truncated']


@pytest.mark.parametrize('fault',['symlink','public-mode','record-cap','byte-cap','unwritable'])
def test_retention_failure_preserves_primary_and_private_boundaries(tmp_path,monkeypatch,fault):
    out=tmp_path/'private';error=EvidenceError('synthetic original failure')
    if fault=='symlink':
        target=tmp_path/'target';target.mkdir();out.symlink_to(target,target_is_directory=True)
    elif fault=='public-mode':out.mkdir();out.chmod(0o755)
    elif fault=='record-cap':monkeypatch.setattr(diagnostics,'MAX_RECORDS',0)
    elif fault=='byte-cap':monkeypatch.setattr(diagnostics,'MAX_TOTAL_BYTES',1)
    else:
        def denied(*a,**k):raise PermissionError('synthetic unwritable destination')
        monkeypatch.setattr(diagnostics.os,'open',denied)
    diagnostics.retain(error,out,{'phase':'synthetic'})
    assert str(error)=='synthetic original failure' and error.private_diagnostic_status['result']=='UNAVAILABLE'
    assert not list(out.glob('*.json'))


def test_prefix_and_aggregate_caps_retain_no_unbounded_diagnostics(tmp_path,monkeypatch):
    raw=b'x'*(diagnostics.PREFIX_BYTES+1);value=diagnostics.bounded_bytes(raw)
    assert value['received_bytes']==len(raw) and value['truncated'] and len(base64.b64decode(value['prefix_b64']))==diagnostics.PREFIX_BYTES
    monkeypatch.setattr(diagnostics,'MAX_RECORDS',1);out=tmp_path/'private'
    one=EvidenceError('first');diagnostics.retain(one,out,{});assert one.private_diagnostic_status['result']=='RETAINED'
    before={p.name:p.read_bytes() for p in out.iterdir()}
    two=EvidenceError('second');diagnostics.retain(two,out,{})
    assert two.private_diagnostic_status['result']=='UNAVAILABLE' and before=={p.name:p.read_bytes() for p in out.iterdir()}


def test_cleanup_failure_does_not_hide_original_strict_transport_error(tmp_path,monkeypatch):
    import pod_transfer
    t,remote,calls,processes=setup(tmp_path)
    original=pod_transfer.selectors.DefaultSelector
    class BrokenClose:
        def __init__(self):self.inner=original()
        def __getattr__(self,name):return getattr(self.inner,name)
        def close(self):self.inner.close();raise OSError('synthetic cleanup failure')
    monkeypatch.setattr(pod_transfer.selectors,'DefaultSelector',BrokenClose)
    def fail(command,**kw):
        p=subprocess.Popen([sys.executable,'-c','import sys;sys.stderr.write("synthetic strict failure");sys.exit(1)'],**kw)
        processes.append(p);return p
    t.popen=fail
    with pytest.raises(EvidenceError,match='SSH transfer process failed'):
        reads.read('metadata',t,{'status.json':tmp_path/'status.json'},Health(),tmp_path/'h',tmp_path,
                   private_diagnostics=tmp_path/'private')
    saved=records(tmp_path/'private')[0]
    assert saved['exception_class']=='EvidenceError' and saved['cleanup']['exception_class']=='OSError'
    assert all(p.poll() is not None and p.stdout.closed and p.stderr.closed for p in processes)


def test_preprocess_creation_error_is_distinguished_and_never_retried(tmp_path):
    t,remote,calls,processes=setup(tmp_path);attempts=[];error=OSError('synthetic spawn refusal')
    def fail(*a,**kw):attempts.append(True);raise error
    t.popen=fail
    with pytest.raises(OSError) as caught:
        reads.read('metadata',t,{'status.json':tmp_path/'status.json'},Health(),tmp_path/'h',tmp_path,
                   private_diagnostics=tmp_path/'private')
    assert caught.value is error and len(attempts)==1
    saved=records(tmp_path/'private')[0]
    assert saved['context']['phase']=='metadata' and saved['transport'] is None
    assert saved['metadata_response']['received_bytes']==0
