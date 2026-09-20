#!/usr/bin/env python3
"""Publish each signed model once to a new AOSSIE repository; recover read-only.

No existing repository is overwritten or deleted. Uncertain commit results must
be reconciled, then independently downloaded and verified. Credentials are used
only by the local Hugging Face client and never enter payloads or receipts.
"""
import argparse
import os
from pathlib import Path
import re
import shutil

from huggingface_hub import CommitOperationAdd,HfApi
from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,inventory,read_json,verify_inventory,write_json
from ovl_pipeline.production_release import PAYLOAD,ReleasePublisherPolicy,validate,verify_payloads
from publish_evidence_archive import lease

NAMES=sorted(PAYLOAD+['release.json','release.sigstore.json'])


def public(info):
    if info.private or type(info.sha) is not str or not re.fullmatch('[0-9a-f]{40}',info.sha):
        raise EvidenceError('public immutable repository revision required')


def check_tree(directory,names):
    if directory.is_symlink():raise EvidenceError('regular owned model staging required')
    actual=[]
    for p in directory.rglob('*'):
        if p.is_symlink() or not(p.is_dir() or p.is_file()):raise EvidenceError('nonregular staged model payload')
        if p.is_file():actual.append(p.relative_to(directory).as_posix())
    if sorted(actual)!=sorted(names):raise EvidenceError('unregistered staged model payload')


def commit_pending(output,api):
    from ovl_pipeline.publication_pause import require_publication_open
    require_publication_open()
    intent=read_json(output/'intent.json');repo=intent['repo'];created=read_json(output/'created.json')
    if created['repo']!=repo or created['intent_sha256']!=digest(intent):raise EvidenceError('created repository differs from intent')
    if (output/'commit-intent.json').exists():raise EvidenceError('commit already attempted; read-only reconciliation required')
    stage=output/'staged';check_tree(stage,NAMES);verify_inventory(stage,intent['files'])
    info=api.repo_info(repo,repo_type='model');public(info)
    if info.sha!=created['revision'] or set(api.list_repo_files(repo,repo_type='model',revision=info.sha))- {'.gitattributes'}:
        raise EvidenceError('new repository changed before publication; preserve it and investigate')
    attempt={'schema':'ovl.model-commit-intent.v1','intent_sha256':digest(intent),'repo':repo,'parent_revision':info.sha}
    write_json(output/'commit-intent.json',attempt)
    committed=api.create_commit(repo,repo_type='model',parent_commit=info.sha,
        commit_message='Publish signed verifiable '+intent['phase']+' model '+intent['release_sha256'][:16],
        operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=confined(stage,n)) for n in NAMES],num_threads=2)
    if type(committed.oid) is not str or not re.fullmatch('[0-9a-f]{40}',committed.oid):
        raise EvidenceError('uncertain commit identity; reconcile without retry')
    result={'schema':'ovl.model-publication.v1','result':'UPLOADED_NOT_DOWNLOAD_VERIFIED','repo':repo,'revision':committed.oid,
            'release_sha256':intent['release_sha256'],'intent_sha256':digest(intent),'url':committed.commit_url,
            'complete_public_download_verification':'NOT_RUN','training_recomputed_by_publication':False}
    write_json(output/'upload.json',result);return result


def check_destinations(statement,output,*,api=None):
    value=validate(read_json(statement))
    if output.exists():raise EvidenceError('destination check needs fresh output')
    api=api or HfApi(endpoint='https://huggingface.co');observations={}
    for phase in ('base','chat'):
        repo=value['models'][phase]['repo'];exists=api.repo_exists(repo,repo_type='model')
        if type(exists) is not bool:raise EvidenceError('invalid destination availability response')
        observations[phase]={'repo':repo,'exists':exists}
    result={'schema':'ovl.model-destination-observation.v1','result':'AVAILABLE' if not any(v['exists'] for v in observations.values()) else 'UNAVAILABLE',
            'release_sha256':digest(value),'models':observations,'reservation':'NONE','provider_mutation':'NOT_RUN'}
    write_json(output,result)
    if result['result']!='AVAILABLE':raise EvidenceError('selected destination already exists; choose a new publication identity before signing')
    return result


def publish(phase,statement,bundle,policy,payloads,output,*,api=None):
    from ovl_pipeline.publication_pause import require_publication_open
    require_publication_open()
    if phase not in ('base','chat'):raise EvidenceError('explicit model phase required')
    if output.exists():raise EvidenceError('existing publication intent; adopt it rather than repeat creation')
    fd=lease(output.with_name(output.name+'.publication.lock'))
    try:
        if output.exists():raise EvidenceError('publication output already exists')
        for directory in payloads.values():check_tree(directory,PAYLOAD)
        r=read_json(payloads['base']/'registration.json')
        checked=verify_payloads(statement,bundle,policy,payloads,r);value=validate(read_json(statement));repo=value['models'][phase]['repo']
        api=api or HfApi(endpoint='https://huggingface.co')
        if api.repo_exists(repo,repo_type='model'):raise EvidenceError('model destination already exists; preserve it and reconcile original intent')
        output.mkdir(parents=True,exist_ok=False);stage=output/'staged';shutil.copytree(payloads[phase],stage)
        shutil.copyfile(statement,stage/'release.json');shutil.copyfile(bundle,stage/'release.sigstore.json')
        check_tree(stage,NAMES);verify_inventory(stage,value['models'][phase]['files'])
        if file_hash(stage/'release.json')!=policy.statement_sha256 or file_hash(stage/'release.sigstore.json')!=file_hash(bundle):
            raise EvidenceError('signed release changed while staging')
        intent={'schema':'ovl.model-publication-intent.v1','phase':phase,'repo':repo,'release_sha256':digest(value),
                'files':inventory(stage,NAMES),'payload_verification':checked,'publisher_code_sha256':file_hash(Path(__file__)),
                'creation_policy':'new-repository-only-exist_ok-false','automatic_retry':False}
        write_json(output/'intent.json',intent)
        api.create_repo(repo,repo_type='model',private=False,exist_ok=False)
        info=api.repo_info(repo,repo_type='model');public(info)
        if set(api.list_repo_files(repo,repo_type='model',revision=info.sha))- {'.gitattributes'}:
            raise EvidenceError('created repository is not empty; preserve intent and investigate')
        write_json(output/'created.json',{'repo':repo,'revision':info.sha,'intent_sha256':digest(intent)})
        return commit_pending(output,api)
    finally:os.close(fd)


def resume_unattempted_commit(output,*,api=None):
    """Only a recorded successful creation and no commit write-ahead may resume."""
    fd=lease(output.with_name(output.name+'.publication.lock'))
    try:return commit_pending(output,api or HfApi(endpoint='https://huggingface.co'))
    finally:os.close(fd)


def reconcile(output,*,api=None):
    fd=lease(output.with_name(output.name+'.publication.lock'))
    try:
        intent=read_json(output/'intent.json');attempt=read_json(output/'commit-intent.json')
        if attempt['intent_sha256']!=digest(intent) or attempt['repo']!=intent['repo']:raise EvidenceError('publication write-ahead differs')
        api=api or HfApi(endpoint='https://huggingface.co');repo=intent['repo'];info=api.repo_info(repo,repo_type='model');public(info)
        if sorted(n for n in api.list_repo_files(repo,repo_type='model',revision=info.sha) if n!='.gitattributes')!=NAMES:
            raise EvidenceError('uncertain publication absent or incomplete; no automatic retry')
        result={'schema':'ovl.model-publication-recovery.v1','result':'FOUND_AWAITING_COMPLETE_DOWNLOAD','repo':repo,'revision':info.sha,
                'release_sha256':intent['release_sha256'],'intent_sha256':digest(intent),'parent_revision':attempt['parent_revision'],
                'commit_continuity':'NOT_CHECKED; selected revision still requires complete signed-payload download verification','provider_mutation':'NOT_RUN',
                'complete_public_download_verification':'NOT_RUN','training_recomputed_by_publication':False}
        write_json(output/('reconciliation-'+digest(result)+'.json'),result);return result
    finally:os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['check-destinations','publish','resume-unattempted-commit','reconcile'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',choices=['base','chat'])
    for n in ('statement','bundle','policy','payloads'):p.add_argument('--'+n,type=Path)
    a=p.parse_args()
    try:
        if a.action=='check-destinations':
            if a.statement is None:raise EvidenceError('unsigned release inventory required')
            result=check_destinations(a.statement,a.output)
        elif a.action=='publish':
            if any(v is None for v in (a.phase,a.statement,a.bundle,a.policy,a.payloads)):raise EvidenceError('all signed release inputs required')
            result=publish(a.phase,a.statement,a.bundle,ReleasePublisherPolicy(**read_json(a.policy)),
                           {ph:a.payloads/ph for ph in ('base','chat')},a.output)
        elif a.action=='reconcile':result=reconcile(a.output)
        else:result=resume_unattempted_commit(a.output)
        from ovl_pipeline.canonical import canonical
        print(canonical(result).decode());return 0
    except Exception as error:
        # Provider errors may reflect credentials; local records retain safe state.
        p.exit(1,'model publication refused: '+type(error).__name__+'; inspect preserved intent, do not repeat creation\n')

if __name__=='__main__':raise SystemExit(main())
