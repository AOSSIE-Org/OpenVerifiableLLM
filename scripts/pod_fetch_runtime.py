#!/usr/bin/env python3
"""Fetch only selected public wheel bytes before the separate offline audit.

No inference, package installation or credentials. Run as a bounded setup job;
the external deadline remains authoritative through network failures.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.error import HTTPError,URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler,ProxyHandler,Request,build_opener

HOSTS={'files.pythonhosted.org','download-r2.pytorch.org','download.pytorch.org'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def url(value):
    v=urlsplit(value)
    if (v.scheme!='https' or v.hostname not in HOSTS or v.port not in (None,443)
        or v.username or v.password or v.query or v.fragment):raise ValueError('unselected public wheel origin')
    return value


class Redirects(HTTPRedirectHandler):
    def __init__(self):self.chain=[]
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        url(newurl);self.chain.append({'status':code,'from':req.full_url,'to':newurl})
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def selected(path,expected):
    path=Path(path)
    if path.is_symlink() or not re.fullmatch('[0-9a-f]{64}',expected) or sha(path)!=expected:raise ValueError('download selection differs')
    def pairs(items):
        v={}
        for k,item in items:
            if k in v:raise ValueError('duplicate selection key')
            v[k]=item
        return v
    value=json.loads(path.read_bytes(),object_pairs_hook=pairs)
    if set(value)!={'schema','files'} or value['schema']!='ovl.public-wheel-download.v1':raise ValueError('wheel selection schema')
    files=value['files']
    if type(files) is not list or not 1<=len(files)<=256:raise ValueError('bounded wheel inventory required')
    names=[];total=0
    for f in files:
        if type(f) is not dict or set(f)!={'path','url','bytes','sha256'}:raise ValueError('download entry schema')
        if type(f['path']) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*\.whl',f['path']):raise ValueError('flat wheel name required')
        if type(f['bytes']) is not int or not 0<f['bytes']<=4*1024**3:raise ValueError('download size bound')
        if not re.fullmatch('[0-9a-f]{64}',f['sha256']):raise ValueError('download digest')
        url(f['url']);names.append(f['path']);total+=f['bytes']
    if names!=sorted(set(names)) or total>8*1024**3:raise ValueError('unique sorted bounded wheel inventory required')
    return files


def fetch(plan,expected,output,report,deadline,*,opener=None,wall=time.time,monotonic=time.monotonic):
    files=selected(plan,expected);output=Path(output).absolute();report=Path(report).absolute()
    if type(deadline) is not int or not 0<deadline-wall()<=600:raise ValueError('bounded original download deadline required')
    for path in (output,report):
        if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('output symlink')
    if report.exists() or output.exists():raise ValueError('preserve prior partial download; fresh output/report required')
    if report==output or output in report.parents:raise ValueError('report must be outside wheel directory')
    output.mkdir(parents=True,exist_ok=False);report.parent.mkdir(parents=True,exist_ok=True)
    end=monotonic()+deadline-wall();started=int(wall());cancelled=threading.Event()
    def left():
        if cancelled.is_set():raise RuntimeError('sibling download failed; preserve partials')
        remaining=min(deadline-wall(),end-monotonic())
        if remaining<=0:raise TimeoutError('fixed download deadline expired')
        return remaining
    attempts=report.parent/(report.name+'.attempts')
    if attempts.exists() or attempts.is_symlink():raise ValueError('fresh attempt receipt directory required')
    attempts.mkdir(mode=0o700)
    def write_receipt(path,value):
        with path.open('x') as f:json.dump(value,f,sort_keys=True,separators=(',',':'));f.write('\n');f.flush();os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def transient(error):
        if isinstance(error,HTTPError):return error.code in (408,429,500,502,503,504)
        if isinstance(error,URLError):return isinstance(error.reason,(TimeoutError,ConnectionError,OSError))
        return isinstance(error,(TimeoutError,ConnectionError))
    def one(item):
        start=time.monotonic_ns();receipts=[];failed=[]
        for attempt in range(2):
            left();partial=output/(item['path']+('.partial' if attempt==0 else '.retry-1.partial'))
            h=hashlib.sha256();count=0;handler=Redirects();http={}
            client=opener or build_opener(ProxyHandler({}),handler)
            request=Request(item['url'],headers={'User-Agent':'OpenVerifiableLLM-public-runtime/1','Accept-Encoding':'identity'})
            began=int(wall())
            try:
                with client.open(request,timeout=min(10,left())) as response,partial.open('xb') as target:
                    url(response.url)
                    http={'requested_url':item['url'],'final_url':response.url,'redirects':handler.chain,'status':response.status,
                          'headers':{k:response.headers.get(k) for k in ('Content-Length','Content-Encoding','ETag','Last-Modified') if response.headers.get(k) is not None}}
                    if response.status!=200 or response.headers.get('Content-Encoding','identity')!='identity':raise ValueError('unexpected public response')
                    length=response.headers.get('Content-Length')
                    if length is not None and int(length)!=item['bytes']:raise ValueError('public length differs')
                    while True:
                        left();data=response.read(min(1024**2,item['bytes']-count+1))
                        if not data:break
                        count+=len(data)
                        if count>item['bytes']:raise ValueError('public file too large')
                        target.write(data);h.update(data)
                    target.flush();os.fsync(target.fileno())
                if count!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('public wheel bytes differ')
                left();destination=output/item['path'];os.link(partial,destination);partial.unlink()
            except Exception as error:
                if partial.exists():
                    with partial.open('rb') as retained:os.fsync(retained.fileno())
                observed={'attempt':attempt,'result':'FAIL','error_type':type(error).__name__,'operator_observed_start_epoch':began,
                          'operator_observed_finish_epoch':int(wall()),'partial_path':str(partial),'written_bytes':partial.stat().st_size if partial.exists() else 0,
                          'written_sha256':h.hexdigest(),'http':http}
                write_receipt(attempts/(item['path']+'.attempt-'+str(attempt)+'.json'),observed);receipts.append(observed)
                if partial.exists():failed.append(partial)
                if attempt or not transient(error):raise
                left();continue
            # Retain failed attempt bytes outside the wheel-only audited directory.
            for old in failed:os.rename(old,attempts/old.name)
            observed={'attempt':attempt,'result':'PASS','operator_observed_start_epoch':began,'operator_observed_finish_epoch':int(wall()),'http':http,
                      'retained_partial_paths':[str(attempts/old.name) for old in failed]}
            write_receipt(attempts/(item['path']+'.attempt-'+str(attempt)+'.json'),observed);receipts.append(observed)
            return {**item,'elapsed_ns':time.monotonic_ns()-start,'result':'COMPLETE_HASH_MATCH','attempts':receipts,
                    'downloader':'OpenVerifiableLLM-public-runtime/1','final_url':response.url,'redirects':handler.chain,'http':http}
        raise RuntimeError('bounded download attempts exhausted')
    def guarded(item):
        try:return one(item)
        except BaseException:cancelled.set();raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(guarded,files))
    fd=os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    left()
    result={'schema':'ovl.public-wheel-download-result.v1','plan_sha256':expected,'started_epoch':started,'finished_epoch':int(wall()),
            'files':receipts,'operator_reported_timestamps':True,'original_deadline_epoch':deadline,'scope':'complete selected public wheel download hashes; installation/runtime/CUDA verification NOT_RUN'}
    write_receipt(report,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','output','report'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--deadline',required=True,type=int)
    a=p.parse_args();fetch(a.plan,a.plan_sha256,a.output,a.report,a.deadline)


if __name__=='__main__':main()
