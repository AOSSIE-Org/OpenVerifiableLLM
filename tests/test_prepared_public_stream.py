import io
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
import verify_prepared_public_stream as m
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,inventory,read_json,write_json


def fixture(tmp_path):
    retained=tmp_path/'retained';retained.mkdir();write_json(retained/'preparation.json',{'fixture':True})
    (retained/'wikipedia').mkdir();(retained/'wikipedia/tokens.u16').write_bytes(b'a\x00'*700000)
    root=file_hash(retained/'preparation.json')
    p={'schema':'ovl.evidence-publication-plan.v1','repo':m.REPO,'kind':'prepared','prefix':'prepared/'+root,
       'subject_sha256':root,'files':inventory(retained,['preparation.json','wikipedia/tokens.u16'])}
    plan=tmp_path/'plan.json';write_json(plan,p);revision='a'*40
    objects={f['path']:(retained/f['path']).read_bytes() for f in p['files']}
    class API:
        private=False;extra=False
        def repo_info(self,*args,**kwargs):return SimpleNamespace(sha=revision,private=self.private)
        def list_repo_files(self,*args,**kwargs):return [p['prefix']+'/'+f['path'] for f in p['files']]+([p['prefix']+'/extra'] if self.extra else [])
    class Response(io.BytesIO):
        def __init__(self,data):super().__init__(data);self.status=200;self.headers={'Content-Length':str(len(data))}
    class Opener:
        calls=[];damage=None
        def open(self,request,timeout):
            assert timeout==60 and not request.has_header('Authorization') and not request.has_header('Range')
            assert request.get_header('Accept-encoding')=='identity' and request.get_header('Cache-control')=='no-cache'
            name=request.full_url.split('/'+p['prefix']+'/')[1];self.calls.append(name);data=objects[name]
            if self.damage=='hash':data=b'X'*len(data)
            elif self.damage=='short':data=data[:-1]
            elif self.damage=='over':data+=b'X'
            elif self.damage=='network':raise TimeoutError('explicit failure; do not put URL/credentials in report')
            response=Response(data)
            if self.damage=='range':response.status=206
            elif self.damage=='encoding':response.headers['Content-Encoding']='gzip'
            elif self.damage=='no-length':response.headers={}
            elif self.damage in ('short','over'):response.headers={}
            return response
    return plan,retained,p,revision,API(),Opener()


@pytest.mark.parametrize('length',['declared','no-length'])
def test_every_response_byte_hashed_without_an_extra_disk_copy(tmp_path,length):
    plan,retained,p,revision,api,opener=fixture(tmp_path)
    if length=='no-length':opener.damage=length
    result=m.verify(plan,file_hash(plan),revision,retained,tmp_path/'verification',api=api,opener=opener)
    assert result['result']=='PASS' and result['bytes']==sum(f['bytes'] for f in p['files'])
    assert opener.calls==[f['path'] for f in p['files']] and result['response_bytes_retained'] is False
    assert result['raw_reconstruction']==result['release_verification']=='NOT_RUN'
    assert sum(f.stat().st_size for f in (tmp_path/'verification').rglob('*') if f.is_file())<20000
    assert inventory(retained,[f['path'] for f in p['files']])==p['files']
    with pytest.raises(EvidenceError):m.verify(plan,file_hash(plan),revision,retained,tmp_path/'verification',api=api,opener=opener)


@pytest.mark.parametrize('damage',['hash','short','over','network','range','encoding'])
def test_failed_full_response_cannot_create_success(tmp_path,damage):
    plan,retained,p,revision,api,opener=fixture(tmp_path);opener.damage=damage
    with pytest.raises(EvidenceError):m.verify(plan,file_hash(plan),revision,retained,tmp_path/'verification',api=api,opener=opener)
    assert len(opener.calls)==1 and not (tmp_path/'verification/verification.json').exists()
    assert read_json(tmp_path/'verification/failure.json')['result']=='FAIL'
    assert inventory(retained,[f['path'] for f in p['files']])==p['files']


@pytest.mark.parametrize('damage',['private','extra','wrong-plan','lost-local','moving-revision'])
def test_wrong_public_identity_or_missing_retained_source_fails_before_download(tmp_path,damage):
    plan,retained,p,revision,api,opener=fixture(tmp_path);expected=file_hash(plan)
    if damage=='private':api.private=True
    elif damage=='extra':api.extra=True
    elif damage=='wrong-plan':expected='f'*64
    elif damage=='lost-local':(retained/'wikipedia/tokens.u16').unlink()
    else:revision='main'
    with pytest.raises(EvidenceError):m.verify(plan,expected,revision,retained,tmp_path/'verification',api=api,opener=opener)
    assert not opener.calls
