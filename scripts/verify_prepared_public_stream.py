#!/usr/bin/env python3
"""Download and hash every prepared-data byte without another full disk copy.

This is anonymous, fixed-revision, complete HTTP response verification. Original
prepared files must remain locally available and are rehashed before the download.
Downloaded response bytes are not retained; only full-file receipts are retained.
This supplies public byte-identity evidence only, never reconstruction or release
verification. No cache, Range request, sampling, implicit retry or credentials.
"""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import re
import time
from urllib.request import build_opener,ProxyHandler,Request

from huggingface_hub import HfApi
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,require_digest,verify_inventory,write_json
from publish_evidence_archive import REPO,existing,validate


def verify(plan_path,expected,revision,retained,output,*,api=None,opener=None):
    plan_path=Path(plan_path);output=Path(output);retained=Path(retained)
    require_digest(expected)
    if file_hash(plan_path)!=expected:raise EvidenceError('prepared plan differs from independent selection')
    plan=read_json(plan_path);validate(plan)
    if plan['kind']!='prepared':raise EvidenceError('streaming download is limited to prepared data; release checks require retained artifacts')
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
    # Empty proxy configuration, no auth/cookie handlers, no token or netrc use.
    opener=opener or build_opener(ProxyHandler({}))
    output.mkdir(parents=True,exist_ok=False)
    intent={'schema':'ovl.prepared-stream-download-intent.v1','plan_sha256':expected,'repo':REPO,'revision':revision,
            'prefix':plan['prefix'],'authentication':'anonymous-no-credentials','local_transport_cache':False,
            'response_bytes_retained':False,'original_files_rehashed':True,'retained_directory':str(retained.resolve()),
            'started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'intent.json',intent);receipts=[]
    for index,item in enumerate(plan['files']):
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{plan["prefix"]}/{item["path"]}'
        request=Request(url,headers={'Accept-Encoding':'identity','Cache-Control':'no-cache'})
        count=0;sha=hashlib.sha256();started=time.monotonic();last=started
        try:
            with opener.open(request,timeout=60) as response:
                if response.status!=200 or response.headers.get('Content-Encoding','identity') not in ('','identity'):
                    raise EvidenceError('full unencoded public response required')
                length=response.headers.get('Content-Length')
                if length is not None and (not length.isdecimal() or int(length)!=item['bytes']):
                    raise EvidenceError('public response length differs')
                while chunk:=response.read(min(1024**2,item['bytes']-count+1)):
                    count+=len(chunk)
                    if count>item['bytes']:raise EvidenceError('oversized public response')
                    sha.update(chunk)
                    now=time.monotonic()
                    if now-last>=30:
                        write_json(output/'progress.json',{'schema':'ovl.prepared-stream-download-progress.v1',
                            'file_index':index,'path':item['path'],'response_bytes_read':count,'complete_files':len(receipts),
                            'scope':'bounded actual response reads only; current file verification pending'})
                        last=now
            if count!=item['bytes'] or sha.hexdigest()!=item['sha256']:raise EvidenceError('downloaded prepared bytes differ')
        except Exception as error:
            write_json(output/'failure.json',{'schema':'ovl.prepared-stream-download-failure.v1','file_index':index,
                'path':item['path'],'response_bytes_read':count,'error_type':type(error).__name__,'complete_files':len(receipts),
                'result':'FAIL','scope':'original files preserved; partial response discarded, no complete inventory acceptance'})
            raise EvidenceError('prepared public response failed: '+type(error).__name__) from None
        receipt={'schema':'ovl.prepared-public-file-response.v1','result':'PASS','revision':revision,'url':url,'file':item,
                 'actual_bytes':count,'actual_sha256':sha.hexdigest(),'response_bytes_retained':False,
                 'elapsed_ms':int((time.monotonic()-started)*1000)}
        write_json(output/f'file-{index:03d}.json',receipt);receipts.append(receipt)
    result={'schema':'ovl.prepared-stream-download-verification.v1','result':'PASS','intent_sha256':digest(intent),
            'plan_sha256':expected,'repo':REPO,'revision':revision,'files':plan['files'],
            'bytes':sum(e['bytes'] for e in plan['files']),'file_receipts_sha256':digest(receipts),
            'finished_utc':datetime.now(timezone.utc).isoformat(),'response_bytes_retained':False,
            'scope':'all selected public HTTP response bytes freshly downloaded and hashed; original local files separately retained',
            'semantic_verification':'NOT_RUN','raw_reconstruction':'NOT_RUN','release_verification':'NOT_RUN'}
    write_json(output/'verification.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','retained','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('plan-sha256','revision'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    try:print(digest(verify(a.plan,a.plan_sha256,a.revision,a.retained,a.output)))
    except Exception as error:p.exit(1,'prepared streaming verification refused: '+type(error).__name__+'; preserve receipts and original files\n')


if __name__=='__main__':main()
