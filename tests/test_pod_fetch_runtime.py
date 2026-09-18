"""Real bytes/files with an explicit bounded HTTPS response double."""
from pathlib import Path
import hashlib,io,json,sys,time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_fetch_runtime as m


class Response(io.BytesIO):
    def __init__(self,data,url,headers=None):
        super().__init__(data);self.url=url;self.status=200;self.headers=headers or {'Content-Length':str(len(data))}


class Opener:
    def __init__(self,data,*,mutate=None):self.data=data;self.calls=[];self.mutate=mutate
    def open(self,request,timeout):
        assert 0<timeout<=10 and not request.has_header('Authorization')
        self.calls.append(request.full_url);r=Response(self.data,request.full_url)
        if self.mutate:self.mutate(r)
        return r


def setup(tmp_path,data=b'complete selected archive bytes'):
    f={'path':'fixture-1.0-py3-none-any.whl','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'url':'https://files.pythonhosted.org/packages/fixture.whl'}
    p=tmp_path/'plan.json';p.write_text(json.dumps({'schema':'ovl.public-wheel-download.v1','files':[f]}))
    return p,f,Opener(data)


def test_complete_actual_bytes_and_retained_refusal_on_restart(tmp_path):
    p,f,client=setup(tmp_path);out=tmp_path/'wheels';report=tmp_path/'report.json'
    result=m.fetch(p,m.sha(p),out,report,int(time.time())+30,opener=client)
    assert (out/f['path']).read_bytes()==client.data
    assert result['files'][0]['result']=='COMPLETE_HASH_MATCH' and result['files'][0]['sha256']==f['sha256']
    assert not list(out.glob('*.partial')) and report.exists()
    with pytest.raises(ValueError,match='preserve'):m.fetch(p,m.sha(p),out,report,int(time.time())+30,opener=client)
    assert len(client.calls)==1


@pytest.mark.parametrize('fault',['content','length','encoding','redirect','oversize'])
def test_bad_response_never_promotes_or_reports_success(tmp_path,fault):
    p,f,client=setup(tmp_path)
    if fault=='content':client.data=b'x'*len(client.data)
    if fault=='oversize':client.data+=b'x'
    if fault=='length':client.mutate=lambda r:r.headers.update({'Content-Length':'1'})
    if fault=='encoding':client.mutate=lambda r:r.headers.update({'Content-Encoding':'gzip'})
    if fault=='redirect':client.mutate=lambda r:setattr(r,'url','http://127.0.0.1/private')
    with pytest.raises(ValueError):m.fetch(p,m.sha(p),tmp_path/'wheels',tmp_path/'report',int(time.time())+30,opener=client)
    assert not(tmp_path/'wheels'/f['path']).exists() and not(tmp_path/'report').exists()
    assert list((tmp_path/'wheels').glob('*.partial'))


@pytest.mark.parametrize('fault',['traversal','http','unselected-host','credentials','query','duplicate','boolean-size','wrong-pin','expired'])
def test_bad_selection_does_not_contact_network(tmp_path,fault):
    p,f,client=setup(tmp_path);v=json.loads(p.read_bytes())
    if fault=='traversal':v['files'][0]['path']='../fixture.whl'
    elif fault=='http':v['files'][0]['url']='http://files.pythonhosted.org/fixture.whl'
    elif fault=='unselected-host':v['files'][0]['url']='https://127.0.0.1/fixture.whl'
    elif fault=='credentials':v['files'][0]['url']='https://user:password@files.pythonhosted.org/fixture.whl'
    elif fault=='query':v['files'][0]['url']+='?token=synthetic'
    elif fault=='duplicate':v['files'].append(dict(v['files'][0]))
    elif fault=='boolean-size':v['files'][0]['bytes']=True
    p.write_text(json.dumps(v));pin='a'*64 if fault=='wrong-pin' else m.sha(p)
    with pytest.raises((ValueError,TimeoutError)):m.fetch(p,pin,tmp_path/'wheels',tmp_path/'report',int(time.time())+(-1 if fault=='expired' else 30),opener=client)
    assert not client.calls


def test_redirect_handler_refuses_outside_https_policy():
    handler=m.Redirects()
    with pytest.raises(ValueError):handler.redirect_request(None,None,302,'',{},'https://example.org/wheel.whl')


def test_transient_retry_retains_partial_and_original_deadline(tmp_path):
    p,f,client=setup(tmp_path);original=client.open;calls=[]
    def open(request,timeout):
        calls.append(timeout)
        r=original(request,timeout)
        if len(calls)==1:
            r.read=lambda *args:(_ for _ in ()).throw(TimeoutError('injected read timeout'))
        return r
    client.open=open
    result=m.fetch(p,m.sha(p),tmp_path/'wheels',tmp_path/'report.json',int(time.time())+30,opener=client)
    assert len(calls)==2 and len(result['files'][0]['attempts'])==2
    assert not list((tmp_path/'wheels').glob('*.partial'))
    assert list((tmp_path/'report.json.attempts').glob('*.partial'))
    assert (tmp_path/'wheels'/f['path']).read_bytes()==client.data


def test_transient_retry_is_bounded_and_integrity_never_retried(tmp_path):
    p,f,client=setup(tmp_path);calls=[]
    def open(*a,**kw):calls.append(1);raise TimeoutError('injected')
    client.open=open
    with pytest.raises(TimeoutError):m.fetch(p,m.sha(p),tmp_path/'wheels',tmp_path/'report.json',int(time.time())+30,opener=client)
    assert len(calls)==2 and not(tmp_path/'report.json').exists()
    assert len(list((tmp_path/'report.json.attempts').glob('*.json')))==2
