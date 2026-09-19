"""Pinned reconstruction-report transport; synthetic bodies, no private records."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import assemble_computation_evidence as assembly
from ovl_pipeline.canonical import EvidenceError,canonical,sha256

URL='https://raw.githubusercontent.com/AOSSIE-Org/OpenVerifiableLLM/'+'a'*40+'/project/evidence/example/report.json'


def response(monkeypatch,body,*,status=200,url=URL,encoding='identity'):
    seen=[]
    def open_request(request,timeout):
        seen.append((request.full_url,timeout))
        stream=BytesIO(body)
        stream.status=status;stream.url=url;stream.headers={'Content-Encoding':encoding}
        return stream
    monkeypatch.setattr(assembly,'build_opener',lambda handler:SimpleNamespace(open=open_request))
    return seen


def test_pinned_github_report_download_and_hash(monkeypatch):
    body=canonical({'result':'synthetic'})
    seen=response(monkeypatch,body)
    assert assembly.public_object({'url':URL,'sha256':sha256(body)})=={'result':'synthetic'}
    assert seen==[(URL,60)]


@pytest.mark.parametrize('override',[{'status':206},{'encoding':'gzip'},{'url':URL+'?changed=1'}])
def test_partial_transformed_or_relocated_response_rejected(monkeypatch,override):
    body=canonical({'result':'synthetic'});response(monkeypatch,body,**override)
    with pytest.raises(EvidenceError,match='unexpected pinned'):
        assembly.public_object({'url':URL,'sha256':sha256(body)})


def test_changed_bytes_rejected(monkeypatch):
    body=canonical({'result':'synthetic'});response(monkeypatch,body+b'\n')
    with pytest.raises(EvidenceError,match='actual public execution evidence differs'):
        assembly.public_object({'url':URL,'sha256':sha256(body)})


def test_hash_matching_noncanonical_json_still_rejected(monkeypatch):
    body=b'{"result": "synthetic"}\n';response(monkeypatch,body)
    with pytest.raises(EvidenceError,match='noncanonical JSON'):
        assembly.public_object({'url':URL,'sha256':sha256(body)})


def test_oversized_response_rejected(monkeypatch):
    body=b'x'*(16*1024*1024+1);response(monkeypatch,body)
    with pytest.raises(EvidenceError,match='exceeds byte bound'):
        assembly.public_object({'url':URL,'sha256':sha256(body)})


def test_redirect_rejected():
    with pytest.raises(EvidenceError,match='must not redirect'):
        assembly.RejectEvidenceRedirects().redirect_request(None,None,302,'',{},'https://example.org/')


@pytest.mark.parametrize('url',[URL.replace('a'*40,'main'),URL.replace('AOSSIE-Org','unapproved'),URL.replace('/example/','/../')])
def test_wrong_or_mutable_location_rejected_before_transport(monkeypatch,url):
    seen=response(monkeypatch,b'{}')
    with pytest.raises(EvidenceError):assembly.public_object({'url':url,'sha256':sha256(b'{}')})
    assert seen==[]


def test_huggingface_retains_existing_bounded_transport(monkeypatch):
    from ovl_pipeline import source_commitment
    url='https://huggingface.co/datasets/AOSSIE/openverifiable-example-evidence/resolve/'+'b'*40+'/report.json'
    body=canonical({'result':'synthetic'})
    seen=[]
    monkeypatch.setattr(source_commitment,'fetch_metadata',lambda actual:seen.append(actual) or body)
    assert assembly.public_object({'url':url,'sha256':sha256(body)})=={'result':'synthetic'}
    assert seen==[url]
