import io
import os
from pathlib import Path
import time

import pytest
import pod_fetch_prepared as m
from ovl_pipeline.canonical import digest,file_hash,read_json,sha256,write_json


def fixture(tmp_path,monkeypatch):
    bodies={name:('explicit synthetic '+name).encode() for name in m.FILES}
    v={'schema':'ovl.public-prepared-inputs.v1','repo':m.REPO,'revision':'a'*40,
       'preparation_sha256':sha256(bodies['preparation.json']),
       'files':[{'path':name,'bytes':len(data),'sha256':sha256(data)} for name,data in bodies.items()]}
    plan=tmp_path/'plan.json';write_json(plan,v);report=tmp_path/'evidence/report.json';output=tmp_path/'inputs'
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(report.parent/'activity.json'))
    class Response(io.BytesIO):
        status=200
        def __init__(self,body):super().__init__(body);self.headers={'Content-Length':str(len(body))}
    class Opener:
        def __init__(self):self.calls=[];self.damage=None
        def open(self,request,timeout):
            assert not request.has_header('Authorization') and not request.has_header('Range')
            assert 0<timeout<=30
            name=request.full_url.split('/prepared/'+v['preparation_sha256']+'/')[1]
            self.calls.append(name);body=bodies[name]
            if self.damage=='bytes':body=b'X'*len(body)
            elif self.damage=='short':body=body[:-1]
            elif self.damage=='long':body+=b'X'
            elif self.damage=='network':raise TimeoutError('injected failure')
            r=Response(body)
            if self.damage=='range':r.status=206
            elif self.damage=='encoding':r.headers['Content-Encoding']='gzip'
            elif self.damage in ('short','long'):r.headers={}
            return r
    return plan,v,bodies,output,report,Opener()


def test_complete_public_inputs_and_finite_transfer_observation(tmp_path,monkeypatch):
    plan,v,bodies,out,report,opener=fixture(tmp_path,monkeypatch)
    result=m.fetch(plan,file_hash(plan),out,report,int(time.time())+60,opener=opener)
    assert result['result']=='PASS' and opener.calls==m.FILES
    assert all((out/name).read_bytes()==data for name,data in bodies.items())
    a=read_json(report.parent/'activity.json')
    assert a['received_bytes']==a['total_bytes']==sum(map(len,bodies.values())) and a['plan_sha256']==file_hash(plan)
    assert result['training_replay']=='NOT_RUN' and not list(out.rglob('*.partial'))
    with pytest.raises(ValueError):m.fetch(plan,file_hash(plan),out,report,int(time.time())+60,opener=opener)


@pytest.mark.parametrize('damage',['bytes','short','long','network','range','encoding'])
def test_failure_preserves_partial_bytes_without_retry_or_success(tmp_path,monkeypatch,damage):
    plan,v,bodies,out,report,opener=fixture(tmp_path,monkeypatch);opener.damage=damage
    with pytest.raises((ValueError,TimeoutError)):m.fetch(plan,file_hash(plan),out,report,int(time.time())+60,opener=opener)
    assert len(opener.calls)==1 and not report.exists() and read_json(report.parent/'failure.json')['result']=='FAIL'
    if damage in ('bytes','short','long'):assert list(out.rglob('*.partial'))


@pytest.mark.parametrize('damage',['missing','reordered','moving','foreign','path','root','deadline','activity'])
def test_changed_or_incomplete_selection_refused_before_network(tmp_path,monkeypatch,damage):
    plan,v,bodies,out,report,opener=fixture(tmp_path,monkeypatch);deadline=int(time.time())+60
    if damage=='missing':v['files'].pop()
    elif damage=='reordered':v['files'].reverse()
    elif damage=='moving':v['revision']='main'
    elif damage=='foreign':v['repo']='elsewhere/repo'
    elif damage=='path':v['files'][0]['path']='../escape'
    elif damage=='root':v['preparation_sha256']='f'*64
    elif damage=='deadline':deadline=int(time.time())-1
    else:monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'wrong/activity.json'))
    write_json(plan,v)
    with pytest.raises(ValueError):m.fetch(plan,file_hash(plan),out,report,deadline,opener=opener)
    assert not opener.calls
