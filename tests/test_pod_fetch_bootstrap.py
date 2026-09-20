"""Synthetic full transfers and adversarial public bootstrap identities."""
import hashlib
import io
import json
import sys
from pathlib import Path
from urllib.request import Request
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
import pod_fetch_bootstrap as m


def fixture(tmp_path):
    payloads = [b'complete synthetic interpreter', b'complete synthetic source']
    value = {'schema': 'ovl.public-bootstrap.v1', 'repo': m.REPO, 'files': [
        {'path': name, 'repo_path': 'public/'+name, 'revision': 'a'*40,
         'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        for name, data in zip(m.NAMES, payloads)]}
    p = tmp_path/'plan.json'; p.write_text(json.dumps(value))
    return p, value, payloads


class Response(io.BytesIO):
    def __init__(self, data, url='https://cas-bridge.xethub.hf.co/selected?signature=synthetic'):
        super().__init__(data); self.url=url; self.status=200; self.headers={}


class Opener:
    def __init__(self, payloads, fault=None): self.payloads=iter(payloads); self.calls=[]; self.fault=fault
    def open(self, req, timeout):
        self.calls.append(req)
        assert 0<timeout<=15 and not any(k.lower() in ('authorization','cookie','range') for k in req.headers)
        r=Response(next(self.payloads))
        if self.fault: self.fault(r)
        return r


def invoke(tmp_path, p, opener, **kw):
    return m.fetch(p,m.digest(p),tmp_path/'out',tmp_path/'report.json',180,opener=opener,wall=lambda:100,monotonic=lambda:100,**kw)


def test_full_pinned_download_and_receipt(tmp_path):
    p,v,data=fixture(tmp_path);o=Opener(data);r=invoke(tmp_path,p,o)
    assert r['files']==v['files'] and r['result']=='PASS'
    assert [x.read_bytes() for x in sorted((tmp_path/'out').iterdir())]==data
    assert len(o.calls)==2 and all('/resolve/'+('a'*40)+'/' in x.full_url for x in o.calls)
    assert 'signature' not in (tmp_path/'report.json').read_text()


@pytest.mark.parametrize('fault', ['repo','revision','traversal','query','name','missing','extra','bytes','hash','duplicate'])
def test_plan_failures_before_network(tmp_path,fault):
    p,v,data=fixture(tmp_path)
    if fault=='repo':v['repo']='unreviewed/repo'
    elif fault=='revision':v['files'][0]['revision']='main'
    elif fault=='traversal':v['files'][0]['repo_path']='public/../secret'
    elif fault=='query':v['files'][0]['repo_path']='path?download=true'
    elif fault=='name':v['files'][0]['path']='../python.tar.gz'
    elif fault=='missing':v['files'].pop()
    elif fault=='extra':v['files'][0]['command']='run'
    elif fault=='bytes':v['files'][0]['bytes']=True
    elif fault=='hash':v['files'][0]['sha256']='z'*64
    p.write_text(json.dumps(v))
    if fault=='duplicate':p.write_text(p.read_text()[:-1]+',"repo":"'+m.REPO+'"}')
    o=Opener(data)
    with pytest.raises(ValueError):invoke(tmp_path,p,o)
    assert not o.calls and not(tmp_path/'out').exists()


@pytest.mark.parametrize('fault',['changed','short','long','partial','encoding','length','origin','network'])
def test_failed_bytes_preserved_without_success(tmp_path,fault):
    p,v,data=fixture(tmp_path)
    if fault=='changed':data[0]=b'x'*len(data[0])
    elif fault=='short':data[0]=data[0][:-1]
    elif fault=='long':data[0]+=b'x'
    def mutate(r):
        if fault=='partial':r.status=206
        elif fault=='encoding':r.headers['Content-Encoding']='gzip'
        elif fault=='length':r.headers['Content-Length']='-1'
        elif fault=='origin':r.url='https://unselected.invalid/archive'
        elif fault=='network':raise ConnectionError('synthetic transport failure')
    o=Opener(data,mutate)
    with pytest.raises((ValueError,ConnectionError)):invoke(tmp_path,p,o)
    assert len(o.calls)==1 and not(tmp_path/'report.json').exists()
    assert not(tmp_path/'out/python.tar.gz').exists()
    assert json.loads((tmp_path/'report.json.failure.json').read_bytes())['result']=='FAIL'
    if fault in ('changed','short'):assert (tmp_path/'out/python.tar.gz.partial').read_bytes()==data[0]


@pytest.mark.parametrize('url',['http://huggingface.co/a','https://huggingface.co.evil.invalid/a',
    'https://user:pass@huggingface.co/a','https://127.0.0.1/a','https://huggingface.co:444/a','file:///tmp/a'])
def test_redirect_rejected_before_following(url):
    with pytest.raises(ValueError):m.Redirects().redirect_request(Request('https://huggingface.co/a'),None,302,'',{},url)


@pytest.mark.parametrize('fault',['existing','output-symlink','parent-symlink','report-symlink','failure-exists','plan-symlink'])
def test_preserve_paths_and_no_symlink_escape(tmp_path,fault):
    p,v,data=fixture(tmp_path);outside=tmp_path/'outside';outside.mkdir();out=tmp_path/'out';report=tmp_path/'report.json'
    if fault=='existing':out.mkdir();(out/'sole').write_bytes(b'keep')
    elif fault=='output-symlink':out.symlink_to(outside,target_is_directory=True)
    elif fault=='parent-symlink':
        link=tmp_path/'link';link.symlink_to(outside,target_is_directory=True);out=link/'out'
    elif fault=='report-symlink':report.symlink_to(outside/'missing')
    elif fault=='failure-exists':report.with_name(report.name+'.failure.json').write_text('keep')
    elif fault=='plan-symlink':
        link=tmp_path/'linked-plan';link.symlink_to(p);p=link
    o=Opener(data)
    with pytest.raises(ValueError):m.fetch(p,m.digest(p),out,report,180,opener=o,wall=lambda:100,monotonic=lambda:100)
    assert not o.calls and not list(outside.iterdir())
    if fault=='existing':assert(out/'sole').read_bytes()==b'keep'


@pytest.mark.parametrize('backward',[False,True])
def test_late_bytes_never_promoted_even_with_backward_wall_clock(tmp_path,backward):
    p,v,data=fixture(tmp_path);wall=[100];mono=[100]
    def mutate(r):
        original=r.read
        def read(n):
            value=original(n);mono[0]=181;wall[0]=0 if backward else 181;return value
        r.read=read
    with pytest.raises(TimeoutError):m.fetch(p,m.digest(p),tmp_path/'out',tmp_path/'report.json',180,opener=Opener(data,mutate),wall=lambda:wall[0],monotonic=lambda:mono[0])
    assert not(tmp_path/'out/python.tar.gz').exists() and not(tmp_path/'report.json').exists()


def test_second_archive_interruption_preserves_first_and_partial(tmp_path):
    p,v,data=fixture(tmp_path);o=Opener(data);original=o.open
    def interrupted(req,timeout):
        r=original(req,timeout)
        if len(o.calls)==2:
            calls=[0];read=r.read
            def broken(n):
                calls[0]+=1
                if calls[0]>1:raise ConnectionError('synthetic mid-transfer disconnect')
                return read(5)
            r.read=broken
        return r
    o.open=interrupted
    with pytest.raises(ConnectionError):invoke(tmp_path,p,o)
    assert(tmp_path/'out/python.tar.gz').read_bytes()==data[0]
    assert(tmp_path/'out/source.tar.gz.partial').read_bytes()==data[1][:5]
    failure=json.loads((tmp_path/'report.json.failure.json').read_bytes())
    assert failure['completed_files']==1 and failure['partial_bytes']==5
    with pytest.raises(ValueError,match='fresh'):invoke(tmp_path,p,Opener(data))
    assert not(tmp_path/'report.json').exists()


def test_final_receipt_fsync_expiry_preserves_pending_without_success(tmp_path,monkeypatch):
    p,v,data=fixture(tmp_path);mono=[100];original=m.os.fsync
    def slow(fd):
        original(fd)
        if (tmp_path/'report.json.pending').exists():mono[0]=181
    monkeypatch.setattr(m.os,'fsync',slow)
    with pytest.raises(TimeoutError):m.fetch(p,m.digest(p),tmp_path/'out',tmp_path/'report.json',180,opener=Opener(data),wall=lambda:100,monotonic=lambda:mono[0])
    assert not(tmp_path/'report.json').exists() and (tmp_path/'report.json.pending').exists()


def test_actual_opener_validates_each_redirect_and_ignores_proxy_environment(tmp_path,monkeypatch):
    from email.message import Message
    from urllib.request import BaseHandler,build_opener as real_build,ProxyHandler
    from urllib.response import addinfourl
    from urllib.error import HTTPError
    p,v,data=fixture(tmp_path);seen=[]
    class HTTPS(BaseHandler):
        handler_order=100
        def https_open(self,request):
            seen.append(request.full_url)
            h=Message();h['Location']='https://cas-bridge.xethub.hf.co/allowed' if len(seen)==1 else 'http://127.0.0.1/private'
            r=addinfourl(io.BytesIO(b''),h,request.full_url,302);r.msg='Found';return r
    def build(*handlers):
        proxy=next(h for h in handlers if isinstance(h,ProxyHandler));assert proxy.proxies=={}
        return real_build(*handlers,HTTPS())
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:1')
    monkeypatch.setattr(m,'build_opener',build)
    with pytest.raises(ValueError,match='origin'):invoke(tmp_path,p,None)
    assert len(seen)==2 and seen[1]=='https://cas-bridge.xethub.hf.co/allowed'


def test_published_receipt_directory_flush_expiry_revokes_success(tmp_path,monkeypatch):
    p,v,data=fixture(tmp_path);mono=[100];fsync=m.os.fsync
    def slow(fd):
        fsync(fd)
        if (tmp_path/'report.json').exists():mono[0]=181
    monkeypatch.setattr(m.os,'fsync',slow)
    with pytest.raises(TimeoutError):m.fetch(p,m.digest(p),tmp_path/'out',tmp_path/'report.json',180,opener=Opener(data),wall=lambda:100,monotonic=lambda:mono[0])
    assert not(tmp_path/'report.json').exists() and (tmp_path/'report.json.pending').exists()


def test_current_public_runtime_cdn_and_lookalike_rejection():
    url='https://us.aws.cdn.hf.co/xet-bridge-us/archive?signature=synthetic'
    request=m.Redirects().redirect_request(Request('https://huggingface.co/a'),None,302,'',{},url)
    assert request.full_url==url
    with pytest.raises(ValueError,match='origin'):
        m.Redirects().redirect_request(Request('https://huggingface.co/a'),None,302,'',{},url.replace('cdn.hf.co','cdn.hf.co.unselected.invalid'))
