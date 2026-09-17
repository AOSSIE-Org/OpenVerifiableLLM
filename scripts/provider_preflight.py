#!/usr/bin/env python3
"""Read-only RunPod credential/schema/account check. Never creates a resource.

Only fixed queries are sent, to the official TLS endpoint with an Authorization
header. Reports contain selected nonsecret observations, never raw HTTP failures,
credentials, pod environments, payment details or provider keys.
"""
import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import stat
import time
import tomllib
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import HTTPError

ENDPOINT = "https://api.runpod.io/graphql"
QUERIES = {
    "schema": '''query OvlPreflightSchema {
      pod: __type(name: "Pod") { fields { name } }
      creation: __type(name: "PodFindAndDeployOnDemandInput") {
        inputFields { name type { kind name ofType { kind name } } }
      }
    }''',
    "account": '''query OvlPreflightAccount {
      myself { clientBalance currentSpendPerHr isAutoPayEnabled
        pods { id desiredStatus gpuCount costPerHr adjustedCostPerHr }
        networkVolumes { id }
      }
    }''',
}


class Refused(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Refused("provider redirect refused; credential not forwarded")


def credential():
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        path = Path.home() / ".runpod/config.toml"
        if not path.exists():raise Refused("RunPod API credential is not configured locally")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            info = os.fstat(f.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 65536:
                raise Refused("invalid local RunPod configuration file")
            data = tomllib.loads(f.read(65537).decode())
        keys = [v for k,v in data.items() if k.lower() == "apikey"]
        if len(keys) != 1:raise Refused("RunPod API credential is not configured locally")
        key = keys[0]
    if type(key) is not str or not 1 <= len(key) <= 512 or any(ord(c) < 33 or ord(c) > 126 for c in key):
        raise Refused("invalid local RunPod credential format")
    return key


def unique_pairs(pairs):
    d = {}
    for k,v in pairs:
        if k in d:raise Refused("duplicate provider JSON key")
        d[k] = v
    return d


def query(operation, key, *, opener=None):
    if operation not in QUERIES:raise Refused("only fixed read-only operations supported")
    req = Request(ENDPOINT, data=json.dumps({"query":QUERIES[operation]}).encode(),
                  headers={"Authorization":"Bearer " + key,"Content-Type":"application/json",
                           "User-Agent":"OpenVerifiableLLM-provider-preflight/1",
                           "Accept-Encoding":"identity"}, method="POST")
    try:
        with (opener or build_opener(NoRedirect())).open(req, timeout=30) as r:
            if r.url != ENDPOINT or r.status != 200 or r.headers.get("Content-Encoding","identity") != "identity":
                raise Refused("unexpected provider HTTP response")
            raw = r.read(1024*1024+1)
        if len(raw) > 1024*1024:raise Refused("provider response exceeds bound")
        obj = json.loads(raw, object_pairs_hook=unique_pairs, parse_float=Decimal,
                         parse_constant=lambda _: (_ for _ in ()).throw(Refused("nonfinite provider JSON")))
        if type(obj) is not dict or obj.get("errors") or type(obj.get("data")) is not dict:
            raise Refused("provider query failed or omitted data; response text withheld")
    except Refused:
        raise
    except HTTPError as e:
        raise Refused("provider read failed: HTTP status " + str(e.code)) from None
    except Exception as e:
        # HTTP errors and server messages can contain reflected credentials/URLs.
        raise Refused("provider read failed: " + type(e).__name__) from None
    return obj["data"], hashlib.sha256(raw).hexdigest()


def amount(value):
    if type(value) not in (int, Decimal):
        raise Refused("missing or invalid provider money value")
    d=Decimal(value)
    if not d.is_finite() or d < 0 or d > 10**12 or abs(d.as_tuple().exponent)>18:
        raise Refused("missing or invalid provider money value")
    return format(d, "f")


def check(key):
    account, account_hash = query("account",key)
    # Production RunPod currently disables introspection. Account authentication
    # can still be established; absent schema evidence never implies a guard.
    schema=None;schema_hash=None;schema_status="UNAVAILABLE";schema_reason=None
    try:
        schema,schema_hash=query("schema",key);schema_status="OBSERVED"
    except Refused as e:schema_reason=str(e)
    try:
        pod_fields = sorted(x["name"] for x in schema["pod"]["fields"]) if schema is not None else None
        creation_fields = {x["name"]:x["type"] for x in schema["creation"]["inputFields"]} if schema is not None else {}
        user = account["myself"]
        if type(user["isAutoPayEnabled"]) is not bool:raise Refused("missing account auto-pay observation")
        pods = []
        for p in user["pods"]:
            if (type(p["id"]) is not str or type(p["desiredStatus"]) is not str
                    or type(p["gpuCount"]) is not int or p["gpuCount"] < 0):
                raise Refused("invalid provider pod observation")
            pods.append({"id":p["id"],"status":p["desiredStatus"],"gpu_count":p["gpuCount"],
                         "cost_per_hour_usd":amount(p["costPerHr"]),
                         "adjusted_cost_per_hour_usd":amount(p["adjustedCostPerHr"])})
        volumes = [v["id"] for v in user["networkVolumes"]]
        if any(type(v) is not str for v in volumes):raise Refused("invalid provider storage observation")
        return {"schema":"ovl.provider-preflight.v1","result":"OBSERVED_NOT_ADMITTED",
                "observed_epoch":int(time.time()),"endpoint":ENDPOINT,
                "query_sha256":{k:hashlib.sha256(v.encode()).hexdigest() for k,v in QUERIES.items()},
                "response_sha256":{"schema":schema_hash,"account":account_hash},
                "schema_observation_status":schema_status,"schema_observation_reason":schema_reason,
                "account_balance_usd":amount(user["clientBalance"]),
                "account_spend_per_hour_usd":amount(user["currentSpendPerHr"]),
                "auto_pay_enabled_observation":user["isAutoPayEnabled"],
                "pods_unattributed":pods,"network_volume_ids_unattributed":volumes,
                "pod_read_fields":pod_fields,"creation_deadline_fields":{
                    k:creation_fields.get(k) for k in ("stopAfter","terminateAfter")},
                "deadline_readback_fields_present":[k for k in ("stopAfter","terminateAfter") if k in pod_fields] if pod_fields is not None else None,
                "provider_deadline_behavior":"NOT_RUN","resource_mutation":"NOT_RUN",
                "execution_admission":"NOT_RUN","project_spend_attribution":"NOT_RUN"}
    except (KeyError,TypeError,ValueError):
        raise Refused("provider schema/observation incomplete or unsupported") from None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.exit(1,"preflight output exists; preserve prior evidence\n")
    try:result=check(credential());code=0
    except Refused as e:result={"result":"UNAVAILABLE","reason":str(e),"resource_mutation":"NOT_RUN","execution_admission":"NOT_RUN"};code=1
    except Exception as e:result={"result":"FAIL","reason":type(e).__name__,"resource_mutation":"NOT_RUN","execution_admission":"NOT_RUN"};code=1
    with a.output.open("x") as f:json.dump(result,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps({"result":result["result"],"output":str(a.output),"execution_admission":"NOT_RUN"}))
    return code


if __name__=="__main__":raise SystemExit(main())
