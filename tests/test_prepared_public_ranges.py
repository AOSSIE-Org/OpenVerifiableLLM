import io
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import verify_prepared_public_ranges as m
from test_prepared_public_stream import fixture
from ovl_pipeline.canonical import EvidenceError,file_hash,inventory,read_json,write_json


def ranged(tmp_path,damage=None):
    plan,retained,p,revision,api,_=fixture(tmp_path)
    calls=[];lock=threading.Lock();attempts={};completed=[]
    class Response(io.BytesIO):
        def __init__(self,data,start,end,total,url):
            super().__init__(data);self.url=url;self.status=206
            self.headers={'Content-Range':f'bytes {start}-{end}/{total}','Content-Length':str(end-start+1)}
    class Opener:
        def open(self,request,timeout):
            assert timeout==60 and not request.has_header('Authorization')
            assert request.get_header('Accept-encoding')=='identity' and request.get_header('Cache-control')=='no-cache'
            name=request.full_url.split('/'+p['prefix']+'/')[1]
            start,end=map(int,request.get_header('Range').removeprefix('bytes=').split('-'))
            with lock:
                calls.append((name,start,end));n=attempts.get((name,start),0);attempts[name,start]=n+1
            source=(retained/name).read_bytes();data=source[start:end+1]
            if damage=='out-of-order' and start==0 and len(source)>65536:time.sleep(.05)
            if damage=='short' or damage=='first-short' and n==0:data=data[:-1]
            elif damage=='hash':data=b'X'*len(data)
            elif damage=='over':data+=b'X'
            elif damage=='network':raise TimeoutError('explicit test failure')
            response=Response(data,start,end,len(source),request.full_url)
            if damage=='offset':response.headers['Content-Range']=f'bytes {start+1}-{end+1}/{len(source)}'
            elif damage=='total':response.headers['Content-Range']=f'bytes {start}-{end}/{len(source)+1}'
            elif damage=='length':response.headers['Content-Length']=str(end-start)
            elif damage=='missing-range':del response.headers['Content-Range']
            elif damage=='encoding':response.headers['Content-Encoding']='gzip'
            elif damage=='http':response.url='http://example.invalid/file'
            elif damage=='ignored' and len(source)>65536:response.status=200
            elif damage=='whole-200' and start==0 and end+1==len(source):
                response.status=200;del response.headers['Content-Range']
            elif damage=='no-length':del response.headers['Content-Length']
            with lock:completed.append((name,start,end))
            return response
    return plan,retained,p,revision,api,Opener,calls,completed


@pytest.mark.parametrize('damage',[None,'out-of-order','first-short','whole-200','no-length'])
def test_full_ordered_file_hash_and_bounded_retries(tmp_path,monkeypatch,damage):
    plan,retained,p,revision,api,factory,calls,completed=ranged(tmp_path,damage)
    if damage=='first-short':monkeypatch.setattr(m.time,'sleep',lambda _:None)
    result=m.verify(plan,file_hash(plan),revision,retained,tmp_path/'v',chunk_bytes=65536,workers=4,
                    api=api,opener_factory=factory)
    assert result['result']=='PASS' and result['files']==p['files']
    assert result['bytes']==sum(f['bytes'] for f in p['files'])
    assert result['raw_reconstruction']==result['release_verification']=='NOT_RUN'
    assert inventory(retained,[f['path'] for f in p['files']])==p['files']
    for f in p['files']:
        expected=[(f['path'],start,min(start+65536,f['bytes'])-1) for start in range(0,f['bytes'],65536)]
        actual=[c for c in calls if c[0]==f['path']]
        assert sorted(set(actual))==expected
        assert all(actual.count(c)==(2 if damage=='first-short' else 1) for c in expected)
    if damage=='out-of-order':
        large=[x[1] for x in completed if x[0]=='wikipedia/tokens.u16'];assert large[0]!=0
    if damage=='first-short':
        failed=[read_json(f) for f in (tmp_path/'v').rglob('attempt-0.json')]
        assert failed and all(f['result']=='FAIL' and f['retry_selected'] for f in failed)
    assert not any(f.name.endswith('.u16') for f in (tmp_path/'v').rglob('*'))
    with pytest.raises(EvidenceError):m.verify(plan,file_hash(plan),revision,retained,tmp_path/'v',api=api)


@pytest.mark.parametrize('damage',['short','over','hash','network','offset','total','length','missing-range','encoding','ignored','http'])
def test_response_substitution_or_truncation_never_passes(tmp_path,monkeypatch,damage):
    plan,retained,p,revision,api,factory,calls,_=ranged(tmp_path,damage)
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    with pytest.raises(EvidenceError):
        m.verify(plan,file_hash(plan),revision,retained,tmp_path/'v',chunk_bytes=65536,workers=1,api=api,opener_factory=factory)
    assert not (tmp_path/'v/verification.json').exists()
    assert read_json(tmp_path/'v/failure.json')['result']=='FAIL'
    assert max(calls.count(c) for c in calls)==(2 if damage in ('short','network') else 1)
    assert inventory(retained,[f['path'] for f in p['files']])==p['files']


@pytest.mark.parametrize('damage',['private','extra','wrong-plan','lost-local','moving-revision','too-many-workers','unbounded-range'])
def test_identity_or_bounds_fail_before_network(tmp_path,damage):
    plan,retained,p,revision,api,factory,calls,_=ranged(tmp_path);expected=file_hash(plan);kw={}
    if damage=='private':api.private=True
    elif damage=='extra':api.extra=True
    elif damage=='wrong-plan':expected='f'*64
    elif damage=='lost-local':(retained/'wikipedia/tokens.u16').unlink()
    elif damage=='moving-revision':revision='main'
    elif damage=='too-many-workers':kw['workers']=5
    else:kw['chunk_bytes']=64*1024**2+1
    with pytest.raises(EvidenceError):m.verify(plan,expected,revision,retained,tmp_path/'v',api=api,opener_factory=factory,**kw)
    assert not calls


def test_empty_file_is_an_actual_complete_response(tmp_path):
    class Response(io.BytesIO):
        url='https://example.invalid/empty';status=200;headers={'Content-Length':'0'}
    class Opener:
        def open(self,request,timeout):
            assert not request.has_header('Range');return Response(b'')
    item={'path':'empty','bytes':0,'sha256':__import__('hashlib').sha256(b'').hexdigest()}
    result=m.file_responses(Response.url,item,tmp_path/'v',chunk_bytes=64,workers=1,opener_factory=Opener)
    assert result['actual_bytes']==0 and result['ranges']==1 and result['actual_sha256']==item['sha256']


def test_slow_response_has_fixed_lifetime_and_one_retry(tmp_path,monkeypatch):
    clock=[0];calls=[]
    class Response(io.BytesIO):
        url='https://example.invalid/file';status=206
        headers={'Content-Range':'bytes 0-1/3','Content-Length':'2'}
        def read1(self,n):
            clock[0]+=601
            return super().read1(1)
    class Opener:
        def open(self,request,timeout):calls.append(request);return Response(b'ab')
    monkeypatch.setattr(m,'time',SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda _:None))
    with pytest.raises(EvidenceError):m.fetch(Response.url,0,2,3,tmp_path/'v',opener_factory=Opener)
    assert len(calls)==2
    for attempt in range(2):
        value=read_json(tmp_path/f'v/attempt-{attempt}.json')
        assert value['error_type']=='TimeoutError' and value['actual_bytes']==1
        assert value['retry_selected'] is (attempt==0)


@pytest.mark.parametrize('failures',[0,3,5,6])
def test_selected_network_backoff_has_fixed_attempt_count_and_full_hash(tmp_path,monkeypatch,failures):
    from urllib.error import URLError
    import hashlib
    calls=[];sleeps=[]
    class Response(io.BytesIO):
        url='https://example.invalid/file';status=206
        headers={'Content-Range':'bytes 0-2/3','Content-Length':'3'}
    class Opener:
        def open(self,request,timeout):
            calls.append(request)
            if len(calls)<=failures:raise URLError(OSError(11,'explicit temporary network error'))
            return Response(b'abc')
    monkeypatch.setattr(m.time,'sleep',sleeps.append)
    item={'path':'file','bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest()}
    def run():return m.file_responses(Response.url,item,tmp_path/'v',chunk_bytes=3,workers=1,
                                     opener_factory=Opener,maximum_attempts=6)
    if failures==6:
        with pytest.raises(EvidenceError):run()
    else:assert run()['actual_sha256']==item['sha256']
    assert len(calls)==min(failures+1,6)
    assert sleeps==[5,15,30,60,120][:min(failures,5)]
    for i in range(min(failures,6)):
        v=read_json(tmp_path/f'v/range-000000000000/attempt-{i}.json')
        assert v['reason_type']=='BlockingIOError' and v['reason_errno']==11
        assert v['retry_selected'] is (i<5)


@pytest.mark.parametrize('attempts',[0,7,True,1.5])
def test_attempt_limits_fail_before_network(tmp_path,attempts):
    plan,retained,p,revision,api,factory,calls,_=ranged(tmp_path)
    with pytest.raises(EvidenceError):
        m.verify(plan,file_hash(plan),revision,retained,tmp_path/'v',api=api,opener_factory=factory,maximum_attempts=attempts)
    assert not calls


def test_expanded_retries_never_retry_protocol_failure(tmp_path,monkeypatch):
    plan,retained,p,revision,api,factory,calls,_=ranged(tmp_path,'offset');sleeps=[]
    monkeypatch.setattr(m.time,'sleep',sleeps.append)
    with pytest.raises(EvidenceError):
        m.verify(plan,file_hash(plan),revision,retained,tmp_path/'v',api=api,opener_factory=factory,
                 maximum_attempts=6,workers=1,chunk_bytes=65536)
    assert len(calls)==1 and not sleeps
