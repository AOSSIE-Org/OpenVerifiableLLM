"""Future-only creation observation; fake transport, no provider mutations."""
import hashlib
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import provider_request_receipts as m
from ovl_pipeline.canonical import EvidenceError,read_json

SECRET='never-public-test-credential'


def transport(monkeypatch,raw,*,failure=None):
    calls=[]
    class Response:
        status=200
        url=m.provider.ENDPOINT
        headers={'Date':'Fri, 18 Sep 2026 19:00:00 GMT'}
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self,n):
            calls.append(('read',n))
            return raw[:n]
    def open(req,timeout):
        calls.append(('open',timeout))
        assert req.get_header('Authorization')=='Bearer '+SECRET
        if failure:raise failure
        return Response()
    factory=lambda *a,**k:SimpleNamespace(open=open)
    monkeypatch.setattr(m.provider,'credential',lambda:SECRET)
    monkeypatch.setattr(m.provider,'build_opener',factory)
    return calls,factory


@pytest.mark.parametrize('raw',[b'{"data":{"podFindAndDeployOnDemand":{"id":"test"}}}',
    b'{"data":{},"errors":[{"message":"never-public-test-credential","extensions":{"code":"BAD_USER_INPUT"}}]}',
    b'{"data":{},"data":{}}',b'{"data":{"n":NaN}}',b'x'*(m.LIMIT+100),
    b'['*2000+b']'*2000,b'{"errors":[{"message":"\\ud800"}]}'])
def test_unchanged_request_result_or_failure_and_one_bounded_read(tmp_path,monkeypatch,raw):
    calls,factory=transport(monkeypatch,raw)
    try:expected=m.provider.request('create',{'input':{'name':'test'}});category=None
    except m.provider.ProviderFailure as e:category=e.category
    calls.clear()
    recorder=m.Recorder(tmp_path/'private',tmp_path/'public')
    if category:
        with pytest.raises(m.provider.ProviderFailure) as error:recorder('create',{'input':{'name':'test'}})
        assert error.value.category==category
    else:
        actual=recorder('create',{'input':{'name':'test'}})
        assert actual[:2]==expected[:2]
    assert calls==[('open',20),('read',m.LIMIT)]
    assert m.provider.build_opener is factory
    retained=(tmp_path/'private/creation-response.bin').read_bytes()
    assert retained==raw[:m.LIMIT]
    receipt=read_json(tmp_path/'public/creation-response.json')
    assert receipt['retained_bytes_sha256']==hashlib.sha256(retained).hexdigest()
    assert SECRET not in json.dumps(receipt)
    assert receipt['request_result']==('RAISED' if category else 'RETURNED')
    assert stat.S_IMODE((tmp_path/'private').stat().st_mode)==0o700
    for f in (tmp_path/'private').iterdir():assert stat.S_IMODE(f.stat().st_mode)==0o600
    # Restart cannot issue a second creation even if a caller ignores its main fence.
    with pytest.raises(FileExistsError):m.Recorder(tmp_path/'private',tmp_path/'public')('create',{})
    assert len(calls)==2


@pytest.mark.parametrize('failure',[URLError(SECRET),HTTPError('https://secret.invalid',503,SECRET,{},None),TimeoutError(SECRET)])
def test_transport_failure_has_no_extra_read_or_secret(tmp_path,monkeypatch,failure):
    calls,factory=transport(monkeypatch,b'',failure=failure)
    with pytest.raises(m.provider.ProviderFailure):m.Recorder(tmp_path/'private',tmp_path/'public')('create',{})
    assert calls==[('open',20)] and m.provider.build_opener is factory
    r=read_json(tmp_path/'public/creation-response.json')
    assert r['request_result']=='RAISED' and r['retained_bytes']==0
    assert SECRET not in json.dumps(r)


def test_noncreation_uses_original_without_recording(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(m.provider,'request',lambda *a:calls.append(a) or 'unchanged')
    recorder=m.Recorder(tmp_path/'private',tmp_path/'public')
    assert recorder('terminate',{'input':{'podId':'test'}})=='unchanged'
    assert calls==[('terminate',{'input':{'podId':'test'}})]
    assert not list((tmp_path/'private').iterdir())


def test_private_permissions_aliases_and_symlinks_fail(tmp_path):
    p=tmp_path/'private';p.mkdir();p.chmod(0o755)
    with pytest.raises(EvidenceError):m.Recorder(p,tmp_path/'public')
    p.chmod(0o700)
    for public in (p,p/'child',tmp_path):
        with pytest.raises(EvidenceError):m.Recorder(p,public)
    link=tmp_path/'link';link.symlink_to(p,target_is_directory=True)
    with pytest.raises(EvidenceError):m.Recorder(link,tmp_path/'public')
    with pytest.raises(EvidenceError):m.Recorder(p,link/'public')


def test_receipt_write_failure_preserves_returned_id_and_cannot_retry(tmp_path,monkeypatch,capsys):
    calls,factory=transport(monkeypatch,b'{"data":{"podFindAndDeployOnDemand":{"id":"test"}}}')
    recorder=m.Recorder(tmp_path/'private',tmp_path/'public')
    monkeypatch.setattr(m,'write_json',lambda *a:(_ for _ in ()).throw(OSError('disk full')))
    assert recorder('create',{})[0]['podFindAndDeployOnDemand']['id']=='test'
    assert 'retention FAILED' in capsys.readouterr().err
    assert m.provider.build_opener is factory and len(calls)==2
    with pytest.raises(FileExistsError):recorder('create',{})
    assert len(calls)==2


def test_receipt_write_failure_preserves_original_provider_failure(tmp_path,monkeypatch,capsys):
    calls,factory=transport(monkeypatch,b'{"errors":[{"message":"refused"}]}')
    recorder=m.Recorder(tmp_path/'private',tmp_path/'public')
    monkeypatch.setattr(m,'write_json',lambda *a:(_ for _ in ()).throw(OSError(SECRET)))
    with pytest.raises(m.provider.ProviderFailure,match='invalid-response-Refused'):recorder('create',{})
    message=capsys.readouterr().err
    assert 'retention FAILED' in message and SECRET not in message
    assert m.provider.build_opener is factory and len(calls)==2
