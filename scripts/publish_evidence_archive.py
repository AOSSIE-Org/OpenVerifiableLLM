#!/usr/bin/env python3
"""Immutable evidence-prefix transport; no semantic verification or training credit.

Existing public AOSSIE evidence repository only. A write-ahead intent and per-plan
OS lease prohibit automatic retry after uncertain creation. Recovery inspects the
already published prefix; it never overwrites existing files or rewrites history.
Consumers must separately perform the relevant signature/state/reconstruction
checks after full public downloads. Token values never enter records or arguments.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import os
from pathlib import Path
import re

from huggingface_hub import CommitOperationAdd,HfApi
from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,read_json,require_digest,verify_inventory,write_json
from ovl_pipeline.schema import fields,integer

REPO='AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence'


def validate(plan):
    fields(plan,'schema repo kind prefix subject_sha256 files','immutable publication plan')
    if plan['schema']!='ovl.evidence-publication-plan.v1' or plan['repo']!=REPO:
        raise EvidenceError('publication requires the existing approved evidence repository')
    require_digest(plan['subject_sha256']);kind=plan['kind'];root=plan['subject_sha256']
    if type(plan['prefix']) is not str:raise EvidenceError('invalid publication prefix')
    exact={'registration-anchor':('production-anchors/'+root,['registration.sigstore.json']),
           'progress-anchor':(None,['statement.json','statement.sigstore.json']),
           'checkpoint':(None,['checkpoint.json','state.json','state.safetensors'])}
    if kind=='prepared':
        if plan['prefix']!='prepared/'+root:raise EvidenceError('prepared publication prefix differs from root')
        names=None
    elif kind=='registration-packet':
        from ovl_pipeline.production_anchoring import PACKET_FILES
        if plan['prefix']!='production-registration/'+root:raise EvidenceError('registration packet prefix differs')
        names=sorted(PACKET_FILES)
    elif kind=='release-evidence':
        from ovl_pipeline.production_release import EVIDENCE
        if plan['prefix']!='release-evidence/'+root or digest(plan['files'])!=root:raise EvidenceError('release evidence prefix differs')
        names=EVIDENCE
    elif kind=='release-anchor':
        if plan['prefix']!='release-anchors/'+root:raise EvidenceError('release anchor prefix differs')
        names=['release.json','release.sigstore.json']
    elif kind=='python-runtime':
        if plan['prefix']!='runtime-python/'+root:raise EvidenceError('Python distribution prefix differs')
        names=['SHA256SUMS','acquisition.json','distribution.tar.gz','payloads.json']
    elif kind=='operational-evidence':
        if plan['prefix']!='operational-evidence/'+root:raise EvidenceError('operational evidence prefix differs')
        names=['checkpoint.json','retained-export-inventory.json','retained-exports.tar.gz']
    elif kind in exact:
        prefix,names=exact[kind]
        if prefix is not None and plan['prefix']!=prefix:raise EvidenceError('registration publication prefix differs')
        if kind=='progress-anchor' and not re.fullmatch(r'production-progress/[0-9a-f]{64}/progress-[0-9]{5}',plan['prefix']):
            raise EvidenceError('invalid progress publication prefix')
        if kind=='checkpoint' and not re.fullmatch(r'production-checkpoints/[0-9a-f]{64}/boundary-[0-9]{5}',plan['prefix']):
            raise EvidenceError('invalid checkpoint publication prefix')
    else:raise EvidenceError('unsupported publication kind')
    entries=plan['files']
    if type(entries) is not list or not 1<=len(entries)<=256:raise EvidenceError('bounded complete publication inventory required')
    paths=[]
    for e in entries:
        fields(e,'path bytes sha256','publication file');require_digest(e['sha256']);integer(e['bytes'],1,2**40,'publication bytes')
        confined(Path('.'),e['path']);paths.append(e['path'])
    if paths!=sorted(set(paths)):raise EvidenceError('publication inventory must be sorted and unique')
    if names is not None and paths!=names:raise EvidenceError('unexpected anchor/checkpoint files')
    if kind=='prepared' and ('preparation.json' not in paths or any(
            p!='preparation.json' and not re.fullmatch(r'(corpus|tokenizer|conversation-selection|wikipedia|conversation|conversation-validation)/[a-zA-Z0-9_.-]+',p) for p in paths)):
        raise EvidenceError('unexpected prepared-data publication paths')
    marker={'prepared':'preparation.json','progress-anchor':'statement.json','checkpoint':'checkpoint.json',
            'registration-packet':'registration.json','release-anchor':'release.json','python-runtime':'distribution.tar.gz',
            'operational-evidence':'checkpoint.json'}.get(kind)
    if marker and next(e['sha256'] for e in entries if e['path']==marker)!=root:
        raise EvidenceError('publication subject differs from exact manifest/statement bytes')


def lease(path):
    fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except Exception:os.close(fd);raise EvidenceError('another publisher holds this plan lease') from None
    return fd


def existing(api,plan,revision):
    names=api.list_repo_files(REPO,repo_type='dataset',revision=revision)
    return sorted(n for n in names if n==plan['prefix'] or n.startswith(plan['prefix']+'/'))


def upload(plan_path,staging,output,*,api=None):
    from ovl_pipeline.publication_pause import require_publication_open
    require_publication_open()
    plan=read_json(plan_path);validate(plan)
    if output.exists():raise EvidenceError('existing publication intent/result; reconcile rather than repeat upload')
    fd=lease(plan_path.with_name(plan_path.name+'.publication.lock'))
    try:
        if output.exists() or staging.is_symlink():raise EvidenceError('fresh output and regular staging required')
        verify_inventory(staging,plan['files'])
        actual=[]
        for p in staging.rglob('*'):
            if p.is_symlink() or not(p.is_dir() or p.is_file()):raise EvidenceError('nonregular staged evidence')
            if p.is_file():actual.append(p.relative_to(staging).as_posix())
        if sorted(actual)!=[e['path'] for e in plan['files']]:raise EvidenceError('staging contains unregistered files')
        from ovl_pipeline.publication_privacy import review_export
        review_export(plan,staging,plan_path.with_name(plan_path.name+'.review.json'))
        api=api or HfApi(endpoint='https://huggingface.co')
        info=api.repo_info(REPO,repo_type='dataset')
        if info.private or not re.fullmatch('[0-9a-f]{40}',info.sha):raise EvidenceError('existing public pinned parent required')
        if existing(api,plan,info.sha):raise EvidenceError('prefix already exists; adopt and verify, never overwrite')
        output.mkdir(parents=True,exist_ok=False)
        intent={'schema':'ovl.evidence-publication-intent.v1','plan_sha256':file_hash(plan_path),'plan':plan,
                'parent_revision':info.sha,'started_utc':datetime.now(timezone.utc).isoformat(),
                'publisher_code_sha256':file_hash(Path(__file__)),'semantic_verification':'NOT_RUN'}
        write_json(output/'intent.json',intent)
        commit=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
            commit_message='Archive immutable '+plan['kind']+' evidence '+plan['subject_sha256'][:16],
            operations=[CommitOperationAdd(path_in_repo=plan['prefix']+'/'+e['path'],path_or_fileobj=confined(staging,e['path'])) for e in plan['files']],num_threads=2)
        if not re.fullmatch('[0-9a-f]{40}',commit.oid):raise EvidenceError('provider returned invalid commit identity; reconcile intent')
        receipt={'schema':'ovl.evidence-publication.v1','result':'UPLOADED_NOT_DOWNLOAD_VERIFIED','repo':REPO,
                 'revision':commit.oid,'url':commit.commit_url,'intent_sha256':digest(intent),
                 'prefix':plan['prefix'],'finished_utc':datetime.now(timezone.utc).isoformat(),
                 'complete_download_verification':'NOT_RUN','semantic_verification':'NOT_RUN'}
        write_json(output/'upload.json',receipt);return receipt
    finally:os.close(fd)


def reconcile(plan_path,output,*,api=None):
    """Read-only recovery of one existing prefix; returns a pin for fresh download.

    Matching names and server metadata cannot prove byte identity. No recovered
    receipt admits execution; the complete fixed-revision download still must pass.
    """
    plan=read_json(plan_path);validate(plan)
    fd=lease(plan_path.with_name(plan_path.name+'.publication.lock'))
    try:
        intent=read_json(output/'intent.json')
        if intent['plan']!=plan or intent['plan_sha256']!=file_hash(plan_path):raise EvidenceError('publication intent differs from selected plan')
        api=api or HfApi(endpoint='https://huggingface.co');info=api.repo_info(REPO,repo_type='dataset')
        if info.private or not re.fullmatch('[0-9a-f]{40}',info.sha):raise EvidenceError('invalid public recovery revision')
        names=existing(api,plan,info.sha);expected=sorted(plan['prefix']+'/'+e['path'] for e in plan['files'])
        if names!=expected:raise EvidenceError('publication prefix absent/incomplete/altered; preserve intent and investigate, no automatic retry')
        receipt={'schema':'ovl.evidence-publication-recovery.v1','result':'FOUND_AWAITING_COMPLETE_DOWNLOAD','repo':REPO,
                 'revision':info.sha,'prefix':plan['prefix'],'intent_sha256':digest(intent),
                 'observed_utc':datetime.now(timezone.utc).isoformat(),'provider_mutation':'NOT_RUN',
                 'complete_download_verification':'NOT_RUN','semantic_verification':'NOT_RUN'}
        # Content-addressed observations do not replace the original creation record.
        write_json(output/('reconciliation-'+digest(receipt)+'.json'),receipt);return receipt
    finally:os.close(fd)


def download(plan_path,revision,output,*,api=None,fetch_file=None):
    """Download every selected byte anonymously; semantic checks remain separate."""
    from huggingface_hub import hf_hub_download
    plan=read_json(plan_path);validate(plan)
    if type(revision) is not str or not re.fullmatch('[0-9a-f]{40}',revision):raise EvidenceError('immutable public revision required')
    if output.exists():raise EvidenceError('complete download requires a fresh output directory')
    api=api or HfApi(endpoint='https://huggingface.co',token=False);fetch_file=fetch_file or hf_hub_download
    info=api.repo_info(REPO,repo_type='dataset',revision=revision)
    if info.private or info.sha!=revision:raise EvidenceError('selected public revision unavailable')
    if existing(api,plan,revision)!=sorted(plan['prefix']+'/'+e['path'] for e in plan['files']):
        raise EvidenceError('public archive file set differs from selected plan')
    output.mkdir(parents=True,exist_ok=False);target=output/'downloaded';target.mkdir();cache=output/'transport-cache'
    intent={'schema':'ovl.evidence-download-intent.v1','plan_sha256':file_hash(plan_path),'revision':revision,
            'repo':REPO,'prefix':plan['prefix'],'authentication':'anonymous-token-False','force_download':True,
            'started_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'intent.json',intent)
    for e in plan['files']:
        try:
            raw=Path(fetch_file(repo_id=REPO,repo_type='dataset',revision=revision,filename=plan['prefix']+'/'+e['path'],
                token=False,force_download=True,local_files_only=False,cache_dir=cache)).resolve(strict=True)
        except Exception as error:raise EvidenceError('public download failed: '+type(error).__name__) from None
        if not raw.is_relative_to(cache.resolve()) or not raw.is_file():raise EvidenceError('download returned a path outside its fresh transport cache')
        if raw.stat().st_size!=e['bytes'] or file_hash(raw)!=e['sha256']:raise EvidenceError('downloaded public file bytes differ')
        path=confined(target,e['path']);path.parent.mkdir(parents=True,exist_ok=True)
        # Two owner-controlled names for one completely downloaded regular file;
        # avoid a second full corpus copy solely for transport staging. No symlink,
        # cache hit or sampled read supplies download verification credit.
        os.link(raw,path,follow_symlinks=False)
        with path.open('rb') as f:os.fsync(f.fileno())
    verify_inventory(target,plan['files'])
    receipt={'schema':'ovl.evidence-download.v1','result':'PASS','scope':'complete-selected-public-file-byte-identity-only',
             'intent_sha256':digest(intent),'plan_sha256':file_hash(plan_path),'repo':REPO,'revision':revision,
             'files':plan['files'],'bytes':sum(e['bytes'] for e in plan['files']),
             'finished_utc':datetime.now(timezone.utc).isoformat(),'semantic_verification':'NOT_RUN'}
    write_json(output/'verification.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['upload','reconcile','download'])
    p.add_argument('--plan',required=True,type=Path);p.add_argument('--staging',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--revision')
    a=p.parse_args()
    try:
        if a.action=='upload' and a.staging is None:raise EvidenceError('upload requires staging directory')
        if a.action=='upload':result=upload(a.plan,a.staging,a.output)
        elif a.action=='download':result=download(a.plan,a.revision,a.output)
        else:result=reconcile(a.plan,a.output)
        from ovl_pipeline.canonical import canonical
        print(canonical(result).decode())
    except Exception as error:p.exit(1,'evidence publication refused: '+type(error).__name__+'; inspect preserved intent, no automatic retry\n')

if __name__=='__main__':main()
