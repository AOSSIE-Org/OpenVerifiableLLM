#!/usr/bin/env python3
"""Complete hash-selected public training inputs under one original deadline.

Standalone standard-library bootstrap; no credentials, ranges or retries. Explicit immutable local reuse still
requires full selected byte hashes on every use; it is not reconstruction credit.
All prepared Wikipedia and conversation stream files are mandatory. Corpus text
and transformation ledgers remain in the complete public preparation archive.
Transfer observations are liveness only; full hashes gate each final input file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid
from urllib.parse import urlsplit
from urllib.request import build_opener,HTTPRedirectHandler,ProxyHandler,Request

REPO='AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence'
FILES=sorted(['preparation.json','tokenizer/tokenizer.json','tokenizer/tokenizer-manifest.json']+
    [phase+'/'+name for phase in ('wikipedia','conversation','conversation-validation')
     for name in ('documents.jsonl','mask.u8','stream.json','tokens.u16')])


def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def write(path,value):
    if path.is_symlink():raise ValueError('output symlink')
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as f:f.write(encoded(value));f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Redirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        p=urlsplit(newurl)
        if p.scheme!='https' or p.username or p.password or p.port not in (None,443):raise ValueError('unsafe public redirect')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def selected(path,expected):
    if path.is_symlink() or not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('selected prepared plan required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('public prepared plan differs')
    def pairs(items):
        d={}
        for k,v in items:
            if k in d:raise ValueError('duplicate prepared plan key')
            d[k]=v
        return d
    v=json.loads(raw,object_pairs_hook=pairs)
    if set(v)!={'schema','repo','revision','preparation_sha256','files'} or v['schema']!='ovl.public-prepared-inputs.v1' or v['repo']!=REPO:
        raise ValueError('closed public prepared input plan required')
    if not re.fullmatch('[0-9a-f]{40}',v['revision']) or not re.fullmatch('[0-9a-f]{64}',v['preparation_sha256']):
        raise ValueError('immutable revision and preparation root required')
    if type(v['files']) is not list or len(v['files'])!=len(FILES):raise ValueError('all full stream inputs required')
    for f in v['files']:
        if (type(f) is not dict or set(f)!={'path','bytes','sha256'} or type(f['bytes']) is not int
            or not 0<f['bytes']<=32*1024**3 or not re.fullmatch('[0-9a-f]{64}',f['sha256'])):
            raise ValueError('bounded public file selection required')
    if [f['path'] for f in v['files']]!=FILES or sum(f['bytes'] for f in v['files'])>64*1024**3:
        raise ValueError('complete sorted public input inventory required')
    if next(f['sha256'] for f in v['files'] if f['path']=='preparation.json')!=v['preparation_sha256']:
        raise ValueError('prepared manifest identity differs')
    return v


def fetch(plan,expected,output,report,deadline,*,opener=None,wall=time.time,monotonic=time.monotonic,reuse=None):
    value=selected(Path(plan),expected);output=Path(output).absolute();report=Path(report).absolute()
    if type(deadline) is not int or not 0<deadline-wall()<=1500:raise ValueError('original bounded prepared download deadline required')
    if any(p.is_symlink() for root in (output,report) for p in [root,*root.parents]):raise ValueError('prepared output symlink')
    if output.exists() or report.exists() or report==output or output in report.parents:raise ValueError('fresh separate input and evidence paths required')
    reuse=Path(reuse).absolute() if reuse is not None else None
    if reuse is not None:
        if any(p.is_symlink() for p in [reuse,*reuse.parents]) or not reuse.is_dir():raise ValueError('regular explicit reuse directory required')
        if any(reuse==p or p in reuse.parents or reuse in p.parents for p in (output,report)):raise ValueError('separate immutable reuse source required')
    report.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    activity=report.parent/'activity.json'
    if os.environ.get('OVL_ACTIVITY_FILE')!=str(activity) or activity.exists():raise ValueError('fresh explicit public transfer activity required')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    end=monotonic()+deadline-wall();instance=uuid.uuid4().hex;total=sum(f['bytes'] for f in value['files']);received=0;last=None
    client=opener or build_opener(ProxyHandler({}),Redirects())
    def left():
        seconds=min(deadline-wall(),end-monotonic())
        if seconds<=0:raise TimeoutError('original public input download deadline expired')
        return seconds
    def observe(force=False):
        nonlocal last
        now=monotonic()
        if not force and last is not None and now-last<10:return
        write(activity,{'schema':('ovl.public-input-cache-read.v1' if reuse is not None else 'ovl.public-input-transfer.v1'),'process_instance':instance,'pid':os.getpid(),
                        'plan_sha256':expected,'total_bytes':total,'received_bytes':received,
                        'scope':'operator-supervision-only-not-input-verification'});last=now
    for index,f in enumerate(value['files']):
        destination=output/f['path'];destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        partial=destination.with_name(destination.name+'.partial');count=0;h=hashlib.sha256()
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{value["revision"]}/prepared/{value["preparation_sha256"]}/{f["path"]}'
        try:
            if reuse is not None:
                source=reuse/f['path']
                if any(p.is_symlink() for p in [source,*source.parents]):raise ValueError('cache symlink')
                # Closed, caller-owned input tree. Read-only regular files on the
                # same filesystem allow no-copy reuse; never silently download
                # over a missing, writable or corrupted selected cache entry.
                fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
                with os.fdopen(fd,'rb') as cached:
                    before=os.fstat(cached.fileno())
                    if not stat.S_ISREG(before.st_mode) or before.st_mode&0o222 or before.st_size!=f['bytes']:
                        raise ValueError('immutable regular cache input of selected size required')
                    while True:
                        left();chunk=cached.read(1024**2)
                        if not chunk:break
                        count+=len(chunk)
                        if count>f['bytes']:raise ValueError('oversized cached input')
                        h.update(chunk);received+=len(chunk);observe()
                    after=os.fstat(cached.fileno())
                    if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                        raise ValueError('cache changed during verification')
                    if count!=f['bytes'] or h.hexdigest()!=f['sha256']:raise ValueError('complete cached input hash differs')
                    left();os.link(source,destination,follow_symlinks=False)
                    linked=destination.lstat()
                    if (linked.st_dev,linked.st_ino,linked.st_size,linked.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) or linked.st_mode&0o222:
                        raise ValueError('cache identity changed at link')
                directory_fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(directory_fd)
                finally:os.close(directory_fd)
                write(report.parent/f'file-{index:03d}.json',{'schema':'ovl.public-prepared-file.v1','file':f,'result':'COMPLETE_HASH_MATCH','url':url,'acquisition':'LOCAL_FULL_HASH_CHECKED_HARDLINK'})
                continue
            request=Request(url,headers={'Accept-Encoding':'identity','Cache-Control':'no-cache','User-Agent':'OpenVerifiableLLM-prepared-inputs/1'})
            with client.open(request,timeout=min(30,left())) as response,partial.open('xb') as target:
                if response.status!=200 or response.headers.get('Content-Encoding','identity') not in ('','identity'):
                    raise ValueError('full identity response required')
                length=response.headers.get('Content-Length')
                if length is not None and (not length.isdecimal() or int(length)!=f['bytes']):raise ValueError('public input length differs')
                while True:
                    left();chunk=response.read(min(1024**2,f['bytes']-count+1))
                    if not chunk:break
                    count+=len(chunk)
                    if count>f['bytes']:raise ValueError('oversized public input')
                    target.write(chunk);h.update(chunk);received+=len(chunk);observe()
                target.flush();os.fsync(target.fileno())
            if count!=f['bytes'] or h.hexdigest()!=f['sha256']:raise ValueError('complete public input hash differs')
            left();os.link(partial,destination);partial.unlink()
            fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            write(report.parent/f'file-{index:03d}.json',{'schema':'ovl.public-prepared-file.v1','file':f,'result':'COMPLETE_HASH_MATCH','url':url})
        except Exception as error:
            write(report.parent/'failure.json',{'schema':'ovl.public-prepared-input-failure.v1','path':f['path'],
                'error_type':type(error).__name__,'partial_bytes':count,'completed_files':index,'result':'FAIL'})
            raise
    left();observe(force=True)
    result={'schema':'ovl.public-prepared-input-result.v1','result':'PASS','plan_sha256':expected,'files':value['files'],
            'bytes':received,'reused_bytes':received if reuse is not None else 0,'downloaded_bytes':received if reuse is None else 0,'original_deadline_epoch':deadline,'scope':'complete selected public training input byte identity only',
            'raw_reconstruction':'NOT_RUN','training_replay':'NOT_RUN'}
    write(report,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','output','report'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--deadline',type=int,required=True);p.add_argument('--reuse',type=Path);a=p.parse_args()
    try:fetch(a.plan,a.plan_sha256,a.output,a.report,a.deadline,reuse=a.reuse)
    except Exception as error:p.exit(1,'prepared input fetch refused: '+type(error).__name__+'; preserve partial input and evidence\n')


if __name__=='__main__':main()
