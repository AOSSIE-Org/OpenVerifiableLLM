from datetime import datetime, timezone
import io
import json
from urllib.error import HTTPError

import pytest

from ovl_pipeline.canonical import EvidenceError, digest, write_json
from ovl_pipeline.lifecycle import Pending, RetryableRead
from ovl_pipeline.lifecycle_runpod import API, Runpod, CreatePod, NoRedirect


class Response:
    status = 200
    headers = {}
    def __init__(self, url, body, status=200):
        self.url, self.body, self.status = url, body, status
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def read(self, n):
        return self.body[:n]


class Opener:
    def __init__(self, values):
        self.values = iter(values)
        self.requests = []
    def open(self, request, timeout):
        self.requests.append(request)
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return Response(request.full_url, json.dumps(value).encode())


def test_v2_pagination_follows_empty_pages_and_rejects_cycles():
    pages = [{"pods": [], "pagination": {"hasNextPage": True, "nextCursor": "next"}},
             {"pods": [{"id": "synthetic"}], "pagination": {"hasNextPage": False, "nextCursor": None}}]
    opener = Opener(pages)
    p = Runpod("synthetic-value", opener=opener)
    assert p.pods() == [{"id": "synthetic"}]
    assert "cursor=next" in opener.requests[1].full_url
    cyclic = Runpod("synthetic-value", opener=Opener([pages[0], pages[0]]))
    with pytest.raises(EvidenceError, match="cyclic"):
        cyclic.pods()


@pytest.mark.parametrize("status,error", [(401, EvidenceError), (403, EvidenceError), (429, RetryableRead), (503, RetryableRead), (422, EvidenceError)])
def test_provider_errors_are_classified_without_reflected_body(status, error):
    reflected = "synthetic-private-response"
    e = HTTPError(API+"/v2/pods", status, reflected, {}, io.BytesIO(reflected.encode()))
    provider = Runpod("synthetic-value", opener=Opener([e]))
    with pytest.raises(error) as caught:
        provider.request("GET", "/v2/pods")
    assert reflected not in str(caught.value)


def test_creation_timeout_is_uncertain_not_retryable_read():
    provider = Runpod("synthetic-value", opener=Opener([TimeoutError()]))
    with pytest.raises(Pending, match="uncertain"):
        provider.request("POST", "/graphql", {})


def test_redirect_never_forwards_credentials():
    with pytest.raises(EvidenceError, match="redirect"):
        NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.invalid")


def test_creation_requires_live_pinned_guard_and_empty_inventory(tmp_path):
    now = 100
    intent = {"schema": "ovl.lifecycle-guard.v1",
              "identity": {"name": "synthetic-resource", "gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1, "cloud": "SECURE"},
              "created_not_before": 100, "terminate_at": 200, "grace_seconds": 120,
              "hourly_usd": "1", "rental_ceiling_usd": "0.10", "prior_upper_usd": "60", "stop_usd": "120", "cap_usd": "130"}
    payload = dict(name="synthetic-resource",cloudType="SECURE",gpuCount=1,gpuTypeId="NVIDIA GeForce RTX 5090",
                   imageName="synthetic@sha256:"+"0"*64,containerDiskInGb=20,volumeInGb=0,minVcpuCount=8,minMemoryInGb=16,
                   dockerArgs="",startSsh=True,startJupyter=False,ports="22/tcp",
                   terminateAfter=datetime.fromtimestamp(200,timezone.utc).isoformat(),networkVolumeId="synthetic-volume",
                   dataCenterId="synthetic-dc",volumeMountPath="/workspace",allowedCudaVersions=["13.0"])
    class Provider:
        rows = []
        writes = []
        def pods(self):
            return self.rows
        def request(self, method, path, body):
            self.writes.append((method,path,body))
            return {"data":{"podFindAndDeployOnDemand":{"id":"synthetic"}}}
    provider = Provider()
    write_json(tmp_path/"guard.json",{"intent_sha256":digest(intent),"status":"ARMED","last_observed":100})
    request = {"payload":payload,"guard_sha256":digest(intent)}
    dead = CreatePod(provider,intent,digest(intent),tmp_path,lambda:False,clock=lambda:now)
    with pytest.raises(EvidenceError,match="guard"):
        dead.submit("a"*64,request)
    adapter = CreatePod(provider,intent,digest(intent),tmp_path,lambda:True,clock=lambda:now)
    provider.rows=[{"id":"other"}]
    with pytest.raises(EvidenceError,match="empty"):
        adapter.submit("a"*64,request)
    assert not provider.writes
    provider.rows=[]
    adapter.submit("a"*64,request)
    assert len(provider.writes)==1
    assert provider.writes[0][2]["variables"]["input"]["terminateAfter"]==payload["terminateAfter"]


def slow_transport(credential, method, path, body, timeout, connection):
    import time
    time.sleep(10)


@pytest.mark.parametrize('method,error',[('GET',RetryableRead),('POST',Pending)])
def test_total_transport_deadline_cancels_and_reaps_slow_worker(monkeypatch,method,error):
    import multiprocessing,time
    import ovl_pipeline.lifecycle_runpod as transport
    monkeypatch.setattr(transport,'_fetch',slow_transport)
    before={p.pid for p in multiprocessing.active_children()}
    provider=Runpod('synthetic-value',timeout=0.2)
    started=time.monotonic()
    with pytest.raises(error):
        provider.request(method,'/synthetic')
    assert time.monotonic()-started < 2
    assert {p.pid for p in multiprocessing.active_children()} == before


def test_pagination_shares_one_total_deadline(monkeypatch):
    import ovl_pipeline.lifecycle_runpod as transport
    ticks=[100.0]
    monkeypatch.setattr(transport.time,'monotonic',lambda:ticks[0])
    class SlowPages(Opener):
        def open(self,request,timeout):
            ticks[0]+=0.75
            return super().open(request,timeout)
    first={'pods':[],'pagination':{'hasNextPage':True,'nextCursor':'next'}}
    second={'pods':[],'pagination':{'hasNextPage':False,'nextCursor':None}}
    opener=SlowPages([first,second])
    provider=Runpod('synthetic-value',opener=opener,timeout=1)
    with pytest.raises(RetryableRead,match='total deadline'):
        provider.pods()
    assert len(opener.requests)==2


@pytest.mark.parametrize('method,error', [('GET', RetryableRead), ('DELETE', RetryableRead), ('POST', Pending)])
def test_late_inline_response_preserves_mutation_uncertainty(monkeypatch, method, error):
    import ovl_pipeline.lifecycle_runpod as transport
    ticks = [100.0]
    monkeypatch.setattr(transport.time, 'monotonic', lambda: ticks[0])
    class Late(Opener):
        def open(self, request, timeout):
            ticks[0] += 2
            return super().open(request, timeout)
    opener = Late([{}])
    with pytest.raises(error):
        Runpod('synthetic-value', opener=opener, timeout=1).request(method, '/synthetic')
    assert len(opener.requests) == 1


@pytest.mark.parametrize('missing', ['gpu', 'image', 'disk', 'mounts', 'cloud', 'dataCenterId'])
def test_malformed_observation_retains_resource_identity_for_cleanup(tmp_path, monkeypatch, missing):
    import ovl_pipeline.lifecycle_runpod as transport
    pod = {'id': 'synthetic-id', 'name': 'synthetic-name', 'createdAt': '2026-01-01T00:00:00Z',
           'gpu': None, 'image': 'synthetic', 'disk': 20, 'mounts': {}, 'cloud': 'SECURE', 'dataCenterId': 'synthetic'}
    if missing != 'gpu':
        pod['gpu'] = {'id': 'NVIDIA GeForce RTX 5090', 'count': 1}
        del pod[missing]
    accepted = []
    monkeypatch.setattr(transport, 'creation_update', lambda *args, **kw: accepted.append(kw['accepted']))
    adapter = object.__new__(CreatePod)
    adapter.provider = type('Provider', (), {'pods': lambda self: [pod]})()
    adapter.directory, adapter.expected = tmp_path, 'b'*64
    adapter.selected = lambda request: {'name': 'synthetic-name'}
    with pytest.raises(EvidenceError, match='required pod fields'):
        adapter.observe('a'*64, {})
    assert accepted == [('a'*64, {}, 'synthetic-id')]
