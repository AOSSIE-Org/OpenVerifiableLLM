"""Narrow RunPod transport adapter; no retries, fallback or admission shortcuts.

REST v2 supplies inventory/termination. Creation retains the supported GraphQL
terminateAfter request because REST v2 CreatePodRequest does not expose it.
Credentials are supplied by the caller's normal credential mechanism and never
appear in results, operation requests, exception messages or provider receipts.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
import json
import multiprocessing
import re
import socket
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .canonical import EvidenceError, digest, read_json, require_digest
from .lifecycle import Observation, Pending, RetryableRead
from .lifecycle_guard import validate_intent
from .lifecycle_creation import update as creation_update


API = "https://api.runpod.io"
CREATE = """mutation OvlLifecycleCreate($input: PodFindAndDeployOnDemandInput!) {
 podFindAndDeployOnDemand(input:$input) { id name createdAt gpuCount imageName }
}"""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise EvidenceError("provider redirect rejected")


def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise EvidenceError("duplicate provider JSON key")
        out[key] = value
    return out


def parse(raw):
    try:
        def nonfinite(_):
            raise EvidenceError("nonfinite provider JSON")
        return json.loads(raw, object_pairs_hook=pairs, parse_float=Decimal, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise EvidenceError("invalid provider JSON") from None


class Runpod:
    def __init__(self, credential, *, opener=None, timeout=18):
        if type(credential) is not str or not 1 <= len(credential) <= 512 or any(ord(c) < 33 or ord(c) > 126 for c in credential):
            raise EvidenceError("invalid configured provider credential")
        if not 0 < timeout <= 20:
            raise EvidenceError("invalid provider timeout")
        self._credential = credential
        self._isolated = opener is None
        self._opener = opener or build_opener(NoRedirect())
        self.timeout = timeout

    def request(self, method, path, body=None, *, deadline=None):
        end = min(time.monotonic()+self.timeout, deadline) if deadline is not None else time.monotonic()+self.timeout
        if end <= time.monotonic():
            raise RetryableRead('provider total deadline expired')
        if not self._isolated:
            result = self._request_inline(method, path, body)
            if time.monotonic() >= end:
                raise RetryableRead('provider total deadline expired')
            return result
        context = multiprocessing.get_context('spawn')
        receive, send = context.Pipe(duplex=False)
        process = context.Process(target=_fetch, args=(self._credential, method, path, body, self.timeout, send))
        process.start()
        send.close()
        try:
            if not receive.poll(max(0, end-time.monotonic())):
                if method in ('GET','DELETE'):
                    raise RetryableRead('provider total deadline expired')
                raise Pending('creation response uncertain; reconcile original identity')
            try:
                kind, value = receive.recv()
            except EOFError:
                raise EvidenceError('provider transport worker failed') from None
            if kind == 'result':
                return value
            raise {'identity': EvidenceError, 'retry': RetryableRead, 'pending': Pending}[kind](value)
        finally:
            receive.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            if process.is_alive():
                raise EvidenceError('provider transport worker cleanup failed')
            process.close()

    def _request_inline(self, method, path, body=None):
        if not path.startswith("/") or path.startswith("//"):
            raise EvidenceError("invalid provider path")
        url = API + path
        raw = None if body is None else json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
        req = Request(url, data=raw, headers={"Authorization": "Bearer " + self._credential,
                      "Content-Type": "application/json", "Accept-Encoding": "identity",
                      "User-Agent": "OpenVerifiableLLM-lifecycle/1"}, method=method)
        try:
            with self._opener.open(req, timeout=self.timeout) as response:
                if response.url != url or response.headers.get("Content-Encoding", "identity") != "identity":
                    raise EvidenceError("unexpected provider response identity")
                data = response.read(2**20+1)
                if len(data) > 2**20:
                    raise EvidenceError("provider response exceeds bound")
                if response.status == 204 and method == "DELETE":
                    if data:
                        raise EvidenceError("unexpected DELETE response body")
                    return True
                if response.status not in (200, 201):
                    raise EvidenceError("unexpected successful provider status")
                return parse(data)
        except HTTPError as exc:
            status = exc.code
            if method == "DELETE" and status == 404:
                return False  # A 404 alone does not exclude delayed provisioning.
            if status in (401, 403):
                raise EvidenceError("provider authentication or authorization rejected") from None
            if status == 429 or 500 <= status <= 599:
                if method == "GET" or method == "DELETE":
                    raise RetryableRead("provider temporarily unavailable") from None
                raise Pending("creation response uncertain; reconcile original identity") from None
            raise EvidenceError("provider rejected request: HTTP " + str(status)) from None
        except URLError as exc:
            # TLS authenticity failures are never transport-retry permission.
            if isinstance(exc.reason, ssl.SSLError):
                raise EvidenceError("provider TLS validation failed") from None
            if not isinstance(exc.reason, (TimeoutError, ConnectionError, socket.gaierror)):
                raise EvidenceError("unclassified provider transport failure") from None
            if method in ("GET", "DELETE"):
                raise RetryableRead("provider read transport unavailable") from None
            raise Pending("creation response uncertain; reconcile original identity") from None
        except (TimeoutError, ConnectionError):
            if method in ("GET", "DELETE"):
                raise RetryableRead("provider read transport unavailable") from None
            raise Pending("creation response uncertain; reconcile original identity") from None

    def pods(self, *, deadline=None):
        end = min(time.monotonic()+self.timeout, deadline) if deadline is not None else time.monotonic()+self.timeout
        cursor = None
        seen_cursors, seen_ids, result = set(), set(), []
        for _ in range(100):
            params = {"includeClusterPods": "true", "limit": "1000"}
            if cursor is not None:
                params["cursor"] = cursor
            value = self.request("GET", "/v2/pods?" + urlencode(params), deadline=end)
            if type(value) is not dict or type(value.get("pods")) is not list or type(value.get("pagination")) is not dict:
                raise EvidenceError("invalid provider inventory")
            for pod in value["pods"]:
                if type(pod) is not dict or type(pod.get("id")) is not str or pod["id"] in seen_ids:
                    raise EvidenceError("duplicate or invalid provider resource")
                seen_ids.add(pod["id"])
                result.append(pod)
            page = value["pagination"]
            if page.get("hasNextPage") is False:
                if page.get("nextCursor") is not None:
                    raise EvidenceError("inconsistent provider pagination")
                return result
            cursor = page.get("nextCursor")
            if page.get("hasNextPage") is not True or type(cursor) is not str or not cursor or cursor in seen_cursors:
                raise EvidenceError("incomplete or cyclic provider pagination")
            seen_cursors.add(cursor)
        raise EvidenceError("provider inventory exceeds page bound")

    def list(self, *, deadline=None):
        normalized = []
        for p in self.pods(deadline=deadline):
            try:
                created = datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00"))
                if created.tzinfo is None:
                    raise ValueError()
                normalized.append({"id": p["id"], "name": p["name"],
                                   "gpu": (p.get("gpu") or {}).get("id"),
                                   "gpu_count": (p.get("gpu") or {}).get("count", 0),
                                   "cloud": p["cloud"], "created_at": int(created.timestamp())})
            except (KeyError, ValueError, TypeError, AttributeError):
                raise EvidenceError("invalid provider resource identity") from None
        return normalized

    def terminate(self, ident, *, deadline=None):
        if type(ident) is not str or not re.fullmatch("[A-Za-z0-9_-]{1,96}", ident):
            raise EvidenceError("invalid attributed resource id")
        return self.request("DELETE", "/v2/pods/" + ident, deadline=deadline)


def _fetch(credential, method, path, body, timeout, connection):
    """Short-lived cancellable HTTP worker; no retries or external mutation loop."""
    try:
        provider = Runpod(credential, opener=build_opener(NoRedirect()), timeout=timeout)
        connection.send(('result', provider._request_inline(method, path, body)))
    except RetryableRead as exc:
        connection.send(('retry', str(exc)))
    except Pending as exc:
        connection.send(('pending', str(exc)))
    except EvidenceError as exc:
        connection.send(('identity', str(exc)))
    except Exception:
        connection.send(('identity', 'unclassified provider transport worker failure'))
    finally:
        connection.close()


class CreatePod:
    """Journal effect whose preconditions bind an already supervised guard.

    guard_alive is a caller-owned check of the independent supervisor/process,
    not a boolean supplied by an untrusted run manifest. No public CLI exposes
    this mutation without the coordinator's admission and publication checks.
    """
    def __init__(self, provider, intent, expected, guard_directory, guard_alive, *, clock=time.time):
        validate_intent(intent, expected)
        self.provider, self.intent, self.expected = provider, intent, expected
        self.directory, self.guard_alive, self.clock = guard_directory, guard_alive, clock

    def selected(self, request):
        if set(request) != {"payload", "guard_sha256"} or request["guard_sha256"] != self.expected:
            raise EvidenceError("creation is not bound to frozen guard")
        p = request["payload"]
        allowed = set("name cloudType gpuCount gpuTypeId imageName containerDiskInGb volumeInGb minVcpuCount minMemoryInGb dockerArgs startSsh startJupyter ports terminateAfter networkVolumeId dataCenterId volumeMountPath allowedCudaVersions".split())
        if set(p) != allowed:
            raise EvidenceError("unexpected creation fields")
        if (p["name"] != self.intent["identity"]["name"] or p["cloudType"] != "SECURE"
                or p["gpuTypeId"] != "NVIDIA GeForce RTX 5090" or p["gpuCount"] != 1
                or p["startSsh"] is not True or p["startJupyter"] is not False
                or p["dockerArgs"] != "" or p["ports"] != "22/tcp"
                or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", p["imageName"])):
            raise EvidenceError("creation differs from authorized resource configuration")
        deadline = datetime.fromisoformat(p["terminateAfter"].replace("Z", "+00:00"))
        if deadline.tzinfo is None or int(deadline.timestamp()) != self.intent["terminate_at"]:
            raise EvidenceError("provider termination request differs from frozen deadline")
        if not p["networkVolumeId"] or not p["dataCenterId"] or p["volumeMountPath"] != "/workspace":
            raise EvidenceError("persistent volume selection required")
        for name in ("containerDiskInGb", "minVcpuCount", "minMemoryInGb"):
            if type(p[name]) is not int or not 1 <= p[name] <= 4096:
                raise EvidenceError("invalid resource size")
        if type(p["volumeInGb"]) is not int or p["volumeInGb"] != 0:
            raise EvidenceError("network storage must not add unaccounted pod volume")
        if (type(p["allowedCudaVersions"]) is not list or not p["allowedCudaVersions"]
                or any(type(v) is not str or not re.fullmatch(r"13\.[0-9]{1,2}", v) for v in p["allowedCudaVersions"])):
            raise EvidenceError("explicit compatible CUDA versions required")
        return p

    def observe(self, operation, request):
        p = self.selected(request)
        pods = self.provider.pods()
        matches = [x for x in pods if x.get("name") == p["name"]]
        if len(matches) > 1:
            raise EvidenceError("ambiguous original resource identity")
        if not matches:
            return Observation("absent")  # No proof an uncertain create was rejected.
        x = matches[0]
        creation_update(self.directory, self.expected, accepted=(operation, request, x['id']))
        result = {"operation": operation, "id": x["id"], "name": x["name"],
                  "created_at": x["createdAt"], "gpu": x.get("gpu", {}).get("id"),
                  "gpu_count": x.get("gpu", {}).get("count"), "cloud": x["cloud"],
                  "image": x["image"], "disk": x["disk"], "data_center": x["dataCenterId"],
                  "mounts": x["mounts"]}
        return Observation("complete", result)

    def validate(self, operation, request, result):
        p = self.selected(request)
        if (result.get("operation") != operation or result.get("name") != p["name"]
                or result.get("gpu") != p["gpuTypeId"] or result.get("gpu_count") != 1
                or result.get("cloud") != "SECURE" or result.get("image") != p["imageName"]
                or result.get("disk") != p["containerDiskInGb"] or result.get("data_center") != p["dataCenterId"]):
            raise EvidenceError("created resource differs from pinned request")
        expected_mount = {"network": [{"volumeId": p["networkVolumeId"], "path": p["volumeMountPath"]}]}
        if result.get("mounts") != expected_mount:
            raise EvidenceError("created resource storage differs from request")
        created = datetime.fromisoformat(result["created_at"].replace("Z", "+00:00"))
        if created.tzinfo is None or not self.intent["created_not_before"] <= created.timestamp() < self.intent["terminate_at"]:
            raise EvidenceError("created resource timestamp differs from original operation")

    def submit(self, operation, request):
        require_digest(operation)
        p = self.selected(request)
        if self.provider.pods():
            raise EvidenceError("single-resource creation requires reconciled empty pod inventory")
        heartbeat = read_json(self.directory / "guard.json")
        if (heartbeat.get("intent_sha256") != self.expected or heartbeat.get("status") != "ARMED"
                or type(heartbeat.get("last_observed")) is not int
                or not 0 <= self.clock()-heartbeat["last_observed"] <= 30
                or self.clock() >= self.intent["terminate_at"] or not self.guard_alive()):
            raise EvidenceError("independent guard is not freshly armed")
        if self.clock() >= self.intent["terminate_at"]:
            raise EvidenceError("creation deadline passed during guard observation")
        creation_update(self.directory, self.expected, claim=(operation, request))
        if self.clock() >= self.intent['terminate_at']:
            creation_update(self.directory, self.expected, unsent=(operation, request))
            raise EvidenceError('creation deadline passed before POST')
        response = self.provider.request("POST", "/graphql", {"query": CREATE, "variables": {"input": p}})
        if type(response) is not dict or response.get("errors") or not response.get("data", {}).get("podFindAndDeployOnDemand"):
            raise Pending("creation result unestablished; reconcile original operation")
        resource = response['data']['podFindAndDeployOnDemand'].get('id')
        creation_update(self.directory, self.expected, accepted=(operation, request, resource))
        # Do not trust an acknowledgement alone; the journal calls observe next.
