"""Bound a read-only provider call by elapsed time, including DNS/TLS/body reads.

The existing HTTP timeout bounds socket operations, not their aggregate. A local
owned child performs one read; timeout kills and reaps it. No mutation is accepted,
no retry occurs here, and no credential or reflected exception enters the pipe.
The caller retains its original retry policy and rental deadlines.
"""
import json
import os
import select
import signal
import time
from decimal import Decimal

from ovl_pipeline.canonical import EvidenceError

LIMIT=2*1024**2


def pack(value):
    if isinstance(value,Decimal):return {'kind':'decimal','value':str(value)}
    if type(value) in (list,tuple):return {'kind':'array','value':[pack(v) for v in value]}
    if type(value) is dict:return {'kind':'object','value':[[k,pack(v)] for k,v in value.items()]}
    if value is None or type(value) in (str,int,bool):return value
    raise ValueError('unsupported owned provider result')


def unpack(value):
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is not dict or set(value)!={'kind','value'}:raise ValueError('invalid owned result')
    kind,v=value['kind'],value['value']
    if kind=='decimal' and type(v) is str:
        d=Decimal(v)
        if not d.is_finite():raise ValueError('nonfinite owned decimal')
        return d
    if kind=='array' and type(v) is list:return [unpack(x) for x in v]
    if kind=='object' and type(v) is list:
        result={}
        for pair in v:
            if type(pair) is not list or len(pair)!=2 or type(pair[0]) is not str or pair[0] in result:
                raise ValueError('invalid owned object')
            result[pair[0]]=unpack(pair[1])
        return result
    raise ValueError('invalid owned result tag')


def call(operation,variables,execute,failure_type,diagnostic,*,seconds=18):
    if operation not in ('account','identities') or variables not in (None,{}):
        raise EvidenceError('bounded provider transport accepts only argument-free reads')
    if type(seconds) not in (int,float) or not 0<seconds<=18:
        raise EvidenceError('provider read wall budget must be within18seconds')
    started=time.monotonic();reader,writer=os.pipe()
    try:pid=os.fork()
    except BaseException:
        os.close(reader);os.close(writer);raise
    if pid==0:
        os.close(reader)
        try:
            try:
                result=execute(operation,variables)
                value={'ok':True,'value':pack(result)}
            except failure_type as error:
                value={'ok':False,'diagnostic':diagnostic(error)}
            except BaseException:
                value={'ok':False,'diagnostic':{'error_type':'ProviderFailure',
                    'category':'invalid-response-child-failure','http_status':None,'transient':False}}
            # Every container is tagged, so provider-supplied dictionaries cannot
            # impersonate a Decimal tag. Restore original strict money types.
            raw=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
            if len(raw)>LIMIT:os._exit(2)
            with os.fdopen(writer,'wb') as stream:stream.write(raw);stream.flush()
            os._exit(0)
        except BaseException:os._exit(2)
    os.close(writer);data=bytearray();status=None
    try:
        while True:
            left=seconds-(time.monotonic()-started)
            if left<=0:raise failure_type('transport',transient=True)
            ready,_,_=select.select([reader],[],[],left)
            if not ready:raise failure_type('transport',transient=True)
            chunk=os.read(reader,min(65536,LIMIT+1-len(data)))
            if not chunk:break
            data.extend(chunk)
            if len(data)>LIMIT:raise failure_type('invalid-response-child-size')
        # A valid response alone does not excuse an unfinished owned child.
        while status is None:
            done,status_value=os.waitpid(pid,os.WNOHANG)
            if done:status=status_value;break
            if time.monotonic()-started>=seconds:raise failure_type('transport',transient=True)
            time.sleep(.001)
        if status!=0:raise failure_type('invalid-response-child-exit')
        try:value=json.loads(data)
        except Exception:raise failure_type('invalid-response-child-json') from None
        if type(value) is not dict or type(value.get('ok')) is not bool:
            raise failure_type('invalid-response-child-shape')
        if value['ok']:
            try:result=unpack(value['value'])
            except Exception:raise failure_type('invalid-response-child-shape') from None
            if set(value)!={'ok','value'} or type(result) is not list or len(result)!=3:
                raise failure_type('invalid-response-child-shape')
            return tuple(result)
        if set(value)!={'ok','diagnostic'}:raise failure_type('invalid-response-child-shape')
        d=value['diagnostic']
        if type(d) is not dict or set(d)!={'error_type','category','http_status','transient'}:
            raise failure_type('invalid-response-child-diagnostic')
        import re
        if (d['error_type']!='ProviderFailure' or type(d['category']) is not str
            or not re.fullmatch(r'http|transport|invalid-response-[A-Za-z0-9-]+',d['category'])
            or type(d['transient']) is not bool
            or d['http_status'] is not None and (type(d['http_status']) is not int or not 100<=d['http_status']<=599)):
            raise failure_type('invalid-response-child-diagnostic')
        raise failure_type(d['category'],status=d['http_status'],transient=d['transient'])
    finally:
        os.close(reader)
        if status is None:
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
            os.waitpid(pid,0)
