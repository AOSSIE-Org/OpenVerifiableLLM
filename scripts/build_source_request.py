#!/usr/bin/env python3
"""Build a source-signing request from a completed anonymous raw download.

The retained download receipt is operator evidence, not independent acceptance.
This builder rehashes the complete downloaded inventory and checks metadata parents;
it does not repeat decompression or run any transformation, signing or training.
"""
import argparse
from pathlib import Path
import re

from publish_raw_archive import validate_plan
from ovl_pipeline.canonical import EvidenceError, canonical, confined, digest, file_hash, read_json, verify_inventory, write_json
from ovl_pipeline.preparation import preparation_code, preparation_environment, validate_source_metadata
from ovl_pipeline.source_commitment import REVISION_SENTINEL
from ovl_pipeline.conversations import POLICY


def build(plan_path, download_directory, run_id, attempt_id, *, vocab_size=32000, sample_bytes=16_000_000):
    plan=read_json(plan_path);spec=validate_plan(plan)
    receipt=read_json(download_directory/'verification.json')
    if (receipt.get('schema')!='ovl.raw-download-verification.v1' or receipt.get('result')!='PASS'
        or receipt.get('plan_sha256')!=file_hash(plan_path) or receipt.get('repo')!=plan['repo']
        or receipt.get('files')!=plan['files'] or receipt.get('wikipedia_verified')!=plan['wikipedia_verified']
        or receipt.get('total_bytes')!=sum(e['bytes'] for e in plan['files'])
        or not re.fullmatch('[0-9a-f]{40}',receipt.get('revision',''))):
        raise EvidenceError('raw download observation does not bind this complete source plan')
    intent=read_json(download_directory/'download-intent.json')
    if (intent.get('repo')!=plan['repo'] or intent.get('revision')!=receipt['revision']
        or intent.get('plan_sha256')!=file_hash(plan_path) or intent.get('token') is not False
        or intent.get('force_download') is not True or intent.get('xet_disabled') is not True):
        raise EvidenceError('anonymous fresh-download intent differs')
    root=confined(download_directory/'downloaded',plan['prefix'])
    verify_inventory(root,plan['files'])
    wiki=root/'wikipedia';conv=root/'conversation'
    acquisition=read_json(conv/'acquisition.json',canonical_required=False)
    date=re.fullmatch('enwiki-([0-9]{8})-pages-articles.xml.bz2',spec.filename)[1]
    by_name={e['path']:e for e in plan['files']}
    def entries(prefix,names):
        return [{**by_name[prefix+n],'path':n} for n in sorted(names)]
    conversation_inventory=[{k:e[k] for k in ('path','bytes','sha256')} for e in acquisition['files']]
    splits={split:next(e['path'] for e in conversation_inventory if e['path'].startswith('data/'+split+'-')) for split in ('train','validation')}
    value={'schema':'ovl.source-preparation.v2','scope':'production-source-preparation',
        'run_id':run_id,'attempt_id':attempt_id,'source_revision':'0'*40,
        'wikipedia':{'date':date,'spec':spec.object(),'inventory':entries('wikipedia/',[spec.filename]),
            'official_status_sha256':file_hash(wiki/'dumpstatus.json'),
            'metadata_inventory':entries('wikipedia/',[f'enwiki-{date}-{suffix}' for suffix in ('index.html','md5sums.txt','sha1sums.txt')])},
        'conversation':{'repo':acquisition['repo'],'revision':acquisition['revision'],
            'inventory':conversation_inventory,'splits':splits},
        'recipe':{'extractor':'main-nonredirect-stripcode-v1','tokenizer_vocab_size':vocab_size,
            'tokenizer_sample_bytes':sample_bytes,'conversation_policy':POLICY},
        'code':preparation_code(),'environment':preparation_environment(),
        'acquisition_receipts':{'wikipedia':file_hash(wiki/(spec.filename+'.verified.json')),
            'conversation':file_hash(conv/'acquisition.json')},
        'archive':{'repo':plan['repo'],'revision':receipt['revision'],'prefix':plan['prefix'],
            'inventory':plan['files'],'retention_days_target':90,
            'retention_policy':'owner-preserve-best-effort-public-host-v1'}}
    validate_source_metadata(value,wiki,conv)
    value['source_revision']=REVISION_SENTINEL
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--download-directory',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--attempt-id',required=True)
    p.add_argument('--output-directory',type=Path,required=True)
    args=p.parse_args()
    result=build(args.plan,args.download_directory,args.run_id,args.attempt_id)
    name=result['run_id']+'-'+result['attempt_id']+'.json';output=args.output_directory/name
    if output.exists():raise EvidenceError('source request already exists; never overwrite an attempt')
    # Atomic no-overwrite publication; signer still independently validates the request.
    import os
    from ovl_pipeline.preparation_stages import sync_directory
    args.output_directory.mkdir(parents=True,exist_ok=True)
    import tempfile
    fd,pending=tempfile.mkstemp(prefix='.pending-source-',dir=args.output_directory)
    with os.fdopen(fd,'wb') as f:f.write(canonical(result));f.flush();os.fsync(f.fileno())
    os.link(pending,output,follow_symlinks=False)
    sync_directory(args.output_directory)
    os.unlink(pending)
    print(canonical({'result':'REQUEST_BUILT_NOT_ANCHORED','path':str(output),'request_sha256':digest(result),
        'complete_raw_inventory_rehashed':True,'decompression_repeated':False,'production_training_admission':'NOT_RUN'}).decode())

if __name__=='__main__':main()
