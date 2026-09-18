#!/usr/bin/env python3
"""Read every public prepared byte through bounded ranges, then hash whole files.

Each accepted range has exact offsets, length and representation size. Hashing is
in file order even when transfers overlap. A short/transient response has at most
one fresh range attempt; rejected bytes never enter the whole-file hash. Original
local files remain separately retained. This does not reconstruct data or replay
training. No credentials, response cache, skipped ranges or sampled acceptance.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import re
import time
from urllib.error import HTTPError,URLError
from urllib.request import build_opener,ProxyHandler,Request

from huggingface_hub import HfApi
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,require_digest,verify_inventory,write_json
from ovl_pipeline.schema import integer
from publish_evidence_archive import REPO,existing,validate


class ShortResponse(EvidenceError):pass


def fetch(url,start,length,total,output,*,opener_factory=None):
    """At most two complete responses for one explicit, bounded byte interval."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    factory=opener_factory or (lambda:build_opener(ProxyHandler({})))
    for attempt in range(2):
        started=time.monotonic();body=bytearray();status=None;selected_headers=None
        headers={'Accept-Encoding':'identity','Cache-Control':'no-cache'}
        if total:headers['Range']=f'bytes={start}-{start+length-1}'
        try:
            with factory().open(Request(url,headers=headers),timeout=60) as response:
                status=response.status
                selected_headers={k:response.headers.get(k) for k in ('Content-Range','Content-Length','Content-Encoding')}
                if not response.url.startswith('https://'):raise EvidenceError('public response left HTTPS')
                if response.headers.get('Content-Encoding','identity') not in ('','identity'):
                    raise EvidenceError('unencoded representation required')
                if status==206:
                    if not total or response.headers.get('Content-Range')!=f'bytes {start}-{start+length-1}/{total}':
                        raise EvidenceError('public range offsets or total differ')
                elif status!=200 or start!=0 or length!=total or response.headers.get('Content-Range') is not None:
                    raise EvidenceError('server ignored a partial range or returned unsupported status')
                declared=response.headers.get('Content-Length')
                if declared is not None and (not declared.isdecimal() or int(declared)!=length):
                    raise EvidenceError('public range length differs')
                while True:
                    if time.monotonic()-started>=600:raise TimeoutError('fixed response lifetime reached')
                    # read1 returns available bytes instead of waiting to fill
                    # a large buffer while a slow peer renews socket timeouts.
                    chunk=response.read1(min(1024**2,length-len(body)+1))
                    if not chunk:break
                    body.extend(chunk)
                    if len(body)>length:raise EvidenceError('oversized public range')
                if len(body)!=length:raise ShortResponse('public range ended early')
            receipt={'schema':'ovl.prepared-public-range-response.v1','result':'PASS','start':start,
                'length':length,'total':total,'status':status,'headers':selected_headers,'actual_bytes':len(body),
                'actual_sha256':hashlib.sha256(body).hexdigest(),'attempt':attempt,
                'elapsed_ms':int((time.monotonic()-started)*1000),'response_bytes_retained':False}
            write_json(output/f'attempt-{attempt}.json',receipt)
            return body,receipt
        except Exception as error:
            transient=(isinstance(error,(ShortResponse,TimeoutError,ConnectionError))
                or isinstance(error,HTTPError) and (error.code==429 or 500<=error.code<=599)
                or isinstance(error,URLError) and not isinstance(error,HTTPError))
            write_json(output/f'attempt-{attempt}.json',{'schema':'ovl.prepared-public-range-failure.v1',
                'result':'FAIL','start':start,'length':length,'total':total,'status':status,
                'actual_bytes':len(body),'partial_sha256':hashlib.sha256(body).hexdigest(),
                'error_type':type(error).__name__,'http_status':error.code if isinstance(error,HTTPError) else None,
                'retry_selected':transient and attempt==0,'elapsed_ms':int((time.monotonic()-started)*1000)})
            if not transient or attempt:raise EvidenceError('public range failed: '+type(error).__name__) from None
            time.sleep(1)


def file_responses(url,item,output,*,chunk_bytes,workers,opener_factory=None,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    integer(chunk_bytes,1,64*1024**2,'bounded range bytes');integer(workers,1,4,'bounded response workers')
    size=item['bytes'];sha=hashlib.sha256();count=0;receipts=[]
    intervals=iter([(0,0)] if size==0 else ((start,min(chunk_bytes,size-start)) for start in range(0,size,chunk_bytes)))
    pending=deque()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit():
            try:start,length=next(intervals)
            except StopIteration:return False
            pending.append((start,length,pool.submit(fetch,url,start,length,size,output/f'range-{start:012d}',opener_factory=opener_factory)))
            return True
        for _ in range(workers):submit()
        try:
            while pending:
                start,length,future=pending.popleft();body,receipt=future.result()
                if start!=count or len(body)!=length:raise EvidenceError('complete ordered range partition differs')
                sha.update(body);count+=length;receipts.append(receipt)
                del body,future
                if progress:progress(count,len(receipts))
                submit()
        finally:
            for _,_,future in pending:future.cancel()
    if count!=size or sha.hexdigest()!=item['sha256']:
        write_json(output/'whole-file-failure.json',{'result':'FAIL','expected':item,'actual_bytes':count,
            'actual_sha256':sha.hexdigest(),'range_receipts_sha256':digest(receipts)})
        raise EvidenceError('complete downloaded file digest differs')
    return {'file':item,'actual_bytes':count,'actual_sha256':sha.hexdigest(),'ranges':len(receipts),
            'range_receipts_sha256':digest(receipts)}


def verify(plan_path,expected,revision,retained,output,*,chunk_bytes=64*1024**2,workers=4,api=None,opener_factory=None):
    plan_path=Path(plan_path);output=Path(output);retained=Path(retained)
    integer(chunk_bytes,1,64*1024**2,'bounded range bytes');integer(workers,1,4,'bounded response workers')
    require_digest(expected)
    if file_hash(plan_path)!=expected:raise EvidenceError('prepared plan differs from independent selection')
    plan=read_json(plan_path);validate(plan)
    if plan['kind']!='prepared':raise EvidenceError('range verification limited to prepared data')
    if type(revision) is not str or not re.fullmatch('[0-9a-f]{40}',revision):raise EvidenceError('fixed public revision required')
    if output.exists() or any(p.is_symlink() for p in [output,*output.absolute().parents]):
        raise EvidenceError('fresh regular download evidence directory required')
    if any(p.is_symlink() for p in [retained,*retained.absolute().parents]):raise EvidenceError('retained source symlink')
    verify_inventory(retained,plan['files'])
    api=api or HfApi(endpoint='https://huggingface.co',token=False)
    info=api.repo_info(REPO,repo_type='dataset',revision=revision)
    if info.private or info.sha!=revision:raise EvidenceError('selected public revision unavailable')
    if existing(api,plan,revision)!=sorted(plan['prefix']+'/'+e['path'] for e in plan['files']):
        raise EvidenceError('public prepared file set differs')
    output.mkdir(parents=True,exist_ok=False)
    intent={'schema':'ovl.prepared-range-download-intent.v1','plan_sha256':expected,'repo':REPO,'revision':revision,
        'prefix':plan['prefix'],'authentication':'anonymous-no-credentials','local_transport_cache':False,
        'response_bytes_retained':False,'original_files_rehashed':True,'retained_directory':str(retained.resolve()),
        'chunk_bytes':chunk_bytes,'workers':workers,'maximum_attempts_per_range':2,'response_lifetime_seconds':600,
        'started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'intent.json',intent);receipts=[]
    for index,item in enumerate(plan['files']):
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{plan["prefix"]}/{item["path"]}'
        def progress(count,ranges):
            write_json(output/'progress.json',{'schema':'ovl.prepared-range-download-progress.v1',
                'file_index':index,'path':item['path'],'complete_files':len(receipts),'ordered_bytes_read':count,
                'complete_ranges':ranges,'scope':'complete ordered response intervals; current whole-file hash pending'})
        started=time.monotonic()
        try:
            result=file_responses(url,item,output/f'file-{index:03d}-ranges',chunk_bytes=chunk_bytes,
                                  workers=workers,opener_factory=opener_factory,progress=progress)
        except Exception as error:
            write_json(output/'failure.json',{'schema':'ovl.prepared-range-download-failure.v1','result':'FAIL',
                'file_index':index,'path':item['path'],'error_type':type(error).__name__,'complete_files':len(receipts),
                'scope':'all attempts and original bytes preserved; no complete inventory acceptance'})
            raise
        receipt={'schema':'ovl.prepared-public-ranged-file.v1','result':'PASS','revision':revision,'url':url,**result,
                 'elapsed_ms':int((time.monotonic()-started)*1000),'response_bytes_retained':False}
        write_json(output/f'file-{index:03d}.json',receipt);receipts.append(receipt)
    result={'schema':'ovl.prepared-range-download-verification.v1','result':'PASS','intent_sha256':digest(intent),
        'plan_sha256':expected,'repo':REPO,'revision':revision,'files':plan['files'],
        'bytes':sum(e['bytes'] for e in plan['files']),'file_receipts_sha256':digest(receipts),
        'finished_utc':datetime.now(timezone.utc).isoformat(),'response_bytes_retained':False,
        'scope':'every selected public byte freshly downloaded in complete ordered ranges and whole-file SHA256 checked; original local files separately retained',
        'semantic_verification':'NOT_RUN','raw_reconstruction':'NOT_RUN','release_verification':'NOT_RUN'}
    write_json(output/'verification.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','retained','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('plan-sha256','revision'):p.add_argument('--'+name,required=True)
    p.add_argument('--chunk-bytes',type=int,default=64*1024**2);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    try:print(digest(verify(a.plan,a.plan_sha256,a.revision,a.retained,a.output,chunk_bytes=a.chunk_bytes,workers=a.workers)))
    except Exception as error:p.exit(1,'prepared range verification refused: '+type(error).__name__+'; preserve receipts and original files\n')


if __name__=='__main__':main()
