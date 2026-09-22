"""Preserve one future creation response without changing provider semantics.

Raw response bytes stay in an owner-only directory. Public diagnostics contain
bounded shape information and hashes, never reflected provider messages or
credentials. Diagnostics are NOT authority to retry, release a creation fence,
shorten a watchdog deadline, or claim that a resource was not created.

The existing request function still owns parsing, timeouts and all decisions.
Only a future, single-threaded controller may install this scoped observer.
"""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import os
import stat
import sys
import time

import probe_provider_deadline as provider
from ovl_pipeline.canonical import EvidenceError,canonical,digest,write_json

LIMIT=1024*1024+1


def private_directory(path):
    p=Path(path).absolute()
    if any(x.is_symlink() for x in [p,*p.parents]):raise EvidenceError('private response directory symlink')
    p.mkdir(mode=0o700,parents=True,exist_ok=True)
    info=p.stat()
    if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:
        raise EvidenceError('private response directory must be owner-only')
    return p


def shape(raw):
    """No free-form server value is returned, even if it resembles a secret."""
    result={'json_shape':'unparsed','errors':None,'data_present':False,'mutation_value':'absent',
            'error_codes':[],'error_messages_sha256':None}
    try:
        value=json.loads(raw,object_pairs_hook=provider.unique_pairs,parse_float=Decimal,
            parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
    except (ValueError,UnicodeError,provider.Refused,RecursionError):return result
    if type(value) is not dict:result['json_shape']='non-object';return result
    result['json_shape']='object';result['data_present']='data' in value
    data=value.get('data')
    if type(data) is dict and 'podFindAndDeployOnDemand' in data:
        node=data['podFindAndDeployOnDemand']
        result['mutation_value']='null' if node is None else 'object' if type(node) is dict else 'other'
    errors=value.get('errors')
    if type(errors) is list:
        result['errors']=len(errors)
        messages=[]
        for e in errors:
            if type(e) is not dict:continue
            if type(e.get('message')) is str:messages.append(e['message'])
            ext=e.get('extensions');code=ext.get('code') if type(ext) is dict else None
            result['error_codes'].append(code if code in ('GRAPHQL_PARSE_FAILED','GRAPHQL_VALIDATION_FAILED','BAD_USER_INPUT','UNAUTHENTICATED','FORBIDDEN','INTERNAL_SERVER_ERROR') else 'OTHER')
        result['error_codes']=result['error_codes'][:32]
        result['error_messages_sha256']=hashlib.sha256(json.dumps(messages,ensure_ascii=True).encode()).hexdigest()
    return result


class Response:
    def __init__(self,inner,capture):self.inner=inner;self.capture=capture
    def __enter__(self):
        self.inner=self.inner.__enter__()
        status=getattr(self.inner,'status',None)
        self.capture['http_status']=status if type(status) is int else None
        encoding=self.inner.headers.get('Content-Encoding','identity')
        self.capture['content_encoding']=encoding if encoding in ('identity','','gzip','br','deflate') else 'OTHER'
        self.capture['same_endpoint']=self.inner.url==provider.ENDPOINT
        return self
    def __exit__(self,*args):return self.inner.__exit__(*args)
    def __getattr__(self,name):return getattr(self.inner,name)
    def read(self,*args,**kwargs):
        raw=self.inner.read(*args,**kwargs)
        room=LIMIT-len(self.capture['body']);self.capture['body'].extend(raw[:room])
        self.capture['observed_read_bytes']+=len(raw)
        return raw


class Opener:
    def __init__(self,inner,capture):self.inner=inner;self.capture=capture
    def open(self,*args,**kwargs):
        # Do not read HTTPError bodies: that could add another blocking read to
        # the original request bound. Existing HTTP status diagnostics suffice.
        return Response(self.inner.open(*args,**kwargs),self.capture)


class Recorder:
    def __init__(self,private,public):
        self.private=private_directory(private);self.public=Path(public).absolute()
        if any(x.is_symlink() for x in [self.public,*self.public.parents]):raise EvidenceError('public diagnostic directory symlink')
        if self.private==self.public or self.private in self.public.parents or self.public in self.private.parents:
            raise EvidenceError('private responses must be outside public evidence')
        self.public.mkdir(mode=0o700,parents=True,exist_ok=True)

    def __call__(self,operation,variables=None):
        if operation!='create':return provider.request(operation,variables)
        # An additional local fence never grants a second request. The original
        # controller's durable account lease and creation fence remain required.
        path=self.private/'creation-request.json'
        with path.open('xb') as f:
            os.fchmod(f.fileno(),0o600)
            f.write(canonical({'operation':'create','variables_sha256':digest(variables),'observed_epoch':int(time.time())}))
            f.flush();os.fsync(f.fileno())
        capture={'body':bytearray(),'observed_read_bytes':0,'http_status':None,'content_encoding':None,'same_endpoint':None}
        original=provider.build_opener
        provider.build_opener=lambda *a,**k:Opener(original(*a,**k),capture)
        error=None
        try:
            return provider.request(operation,variables)
        except BaseException as e:
            error=e;raise
        finally:
            provider.build_opener=original
            try:self.persist(capture,variables,error)
            except Exception:
                # A diagnostic failure must not hide an actual returned pod ID
                # or change the controller's original failure classification.
                # Absence of the receipt never passes a verification check.
                print('Provider creation diagnostic retention FAILED; original provider result preserved.',file=sys.stderr)

    def persist(self,capture,variables,error):
        raw=bytes(capture.pop('body'))
        with (self.private/'creation-response.bin').open('xb') as f:
            os.fchmod(f.fileno(),0o600);f.write(raw);f.flush();os.fsync(f.fileno())
        value={'schema':'ovl.provider-creation-response-diagnostic.v1','observed_epoch':int(time.time()),
            'variables_sha256':digest(variables),'request_result':'RETURNED' if error is None else 'RAISED',
            'failure':None if error is None else provider.diagnostic(error),**capture,
            'retained_bytes':len(raw),'retained_bytes_sha256':hashlib.sha256(raw).hexdigest(),
            'retained_read_complete':len(raw)==capture['observed_read_bytes'],'shape':shape(raw),
            'raw_response_location':'owner-only separate directory; never public evidence',
            'scope':'diagnostic observation only; no retry, absence, teardown or deadline authority'}
        write_json(self.public/'creation-response.json',value)


def main():
    import argparse
    from ovl_pipeline.canonical import read_json
    from run_rental_controller import run_guarded
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('intent','journal','watchdog-heartbeat','workload-health','private-responses','diagnostics'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--intent-sha256',required=True)
    p.add_argument('--local-storage-budget',type=Path);p.add_argument('--local-storage-budget-sha256');a=p.parse_args()
    from local_storage import Budget
    if (a.local_storage_budget is None)!=(a.local_storage_budget_sha256 is None):p.error('both local storage arguments required')
    budget=None if a.local_storage_budget is None else Budget(a.local_storage_budget,a.local_storage_budget_sha256)
    recorder=Recorder(a.private_responses,a.diagnostics)
    run_guarded(a.journal,read_json(a.intent),a.intent_sha256,a.watchdog_heartbeat,a.workload_health,provider_request=recorder,storage_budget=budget)


if __name__=='__main__':main()
