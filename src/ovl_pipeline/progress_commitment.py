"""Append-only public checkpoint endorsement inputs for GitHub Actions.

Checks publisher/run ancestry and actual downloaded safe checkpoint bytes. This
signs operator assertions, never claims that CI reproduced numerical training.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import re
import shutil

from .anchoring import ISSUER,OWNER_ID,REPOSITORY,REPOSITORY_ID,PublisherPolicy,verify_anchor
from .canonical import EvidenceError,confined,digest,file_hash,read_json,require_digest,write_json
from .production_anchoring import object_at,verify_packet
from .production_chain import verify_chain
from .production_commitment import download_packet,validate_request as registration_request
from .production_identity import ProductionPublisherPolicy
from .progress_anchoring import PROGRESS_WORKFLOW,ProgressPublisherPolicy,statement,verify_prefix
from .schema import fields,integer
from .source_commitment import REF,actions_revision,fetch_metadata,git,tree
from .state import read_state,unpack

REQUEST_DIRECTORY='project/progress-commitments'


def select_request(root,environ):
    actions_revision(root,environ)
    if len(git(root,'rev-list','--parents','-n','1','HEAD').split())!=2:raise EvidenceError('progress request requires single-parent commit')
    lines=git(root,'diff-tree','--no-commit-id','--name-status','-r','--no-renames','HEAD','--',REQUEST_DIRECTORY).splitlines()
    if not lines:return None
    if len(lines)!=1 or not lines[0].startswith('A\t'):raise EvidenceError('one new append-only progress request required')
    name=lines[0].split('\t')[1]
    if not re.fullmatch(REQUEST_DIRECTORY+r'/[a-z0-9][a-z0-9-]+-boundary-[0-9]{5}\.json',name):raise EvidenceError('invalid progress request path')
    if len(git(root,'log','--full-history','--format=%H','HEAD','--',name).splitlines())!=1:raise EvidenceError('progress request identity reused')
    return name


def archive_shape(value,prefix,names,maximum):
    fields(value,'repo revision prefix inventory','public progress archive')
    if (type(value['repo']) is not str or not re.fullmatch(r'AOSSIE/openverifiable-[a-z0-9-]+-evidence',value['repo']) or
            type(value['revision']) is not str or not re.fullmatch('[0-9a-f]{40}',value['revision']) or value['prefix']!=prefix):
        raise EvidenceError('approved immutable progress archive required')
    inv=value['inventory']
    if type(inv) is not list or len(inv)!=len(names):raise EvidenceError('complete closed progress inventory required')
    for e in inv:
        fields(e,'path bytes sha256','progress archive file');require_digest(e['sha256']);integer(e['bytes'],1,maximum,'archive file bytes')
    if [e['path'] for e in inv]!=sorted(names) or sum(e['bytes'] for e in inv)>maximum:
        raise EvidenceError('unexpected or oversized progress archive inventory')


def validate_request(value):
    fields(value,'schema registration_request registration_anchor registration_policy envelopes previous_progress checkpoint_archive','progress signing request')
    if value['schema']!='ovl.progress-signing-request.v1':raise EvidenceError('unsupported progress signing request')
    registration_request(value['registration_request']);root=value['registration_request']['registration_sha256']
    policy=ProductionPublisherPolicy(**value['registration_policy']);policy.validate()
    if policy.statement_sha256!=root:raise EvidenceError('registration policy root differs')
    archive_shape(value['registration_anchor'],'production-anchors/'+root,['registration.sigstore.json'],2*1024**2)
    envelopes=value['envelopes'];previous=value['previous_progress']
    if type(envelopes) is not list or not 1<=len(envelopes)<=4096:raise EvidenceError('bounded nonempty progress prefix required')
    if type(previous) is not list or len(previous)!=len(envelopes)-1:raise EvidenceError('complete previous progress endorsements required')
    for i,v in enumerate(previous):
        fields(v,'archive policy','previous public progress')
        archive_shape(v['archive'],f'production-progress/{root}/progress-{i:05d}',
                      ['statement.json','statement.sigstore.json'],4*1024**2)
        ProgressPublisherPolicy(**v['policy']).validate()
    archive_shape(value['checkpoint_archive'],f'production-checkpoints/{root}/boundary-{len(envelopes)-1:05d}',
                  ['checkpoint.json','state.json','state.safetensors'],2*1024**3)
    return policy


def download_archive(archive,output,*,fetch=fetch_metadata,download=None):
    """Anonymous forced downloads at a fixed revision, followed by full hashes.

    Revalidate the closed archive type before network access. Cache is only a
    transport location: every file is forced from public storage and rehashed.
    """
    prefix=archive.get('prefix')
    if type(prefix) is not str:raise EvidenceError('invalid public archive prefix')
    if re.fullmatch(r'production-anchors/[0-9a-f]{64}',prefix):
        names=['registration.sigstore.json'];maximum=2*1024**2
    elif re.fullmatch(r'production-progress/[0-9a-f]{64}/progress-[0-9]{5}',prefix):
        names=['statement.json','statement.sigstore.json'];maximum=4*1024**2
    elif re.fullmatch(r'production-checkpoints/[0-9a-f]{64}/boundary-[0-9]{5}',prefix):
        names=['checkpoint.json','state.json','state.safetensors'];maximum=2*1024**3
    else:raise EvidenceError('unsupported public archive prefix')
    archive_shape(archive,prefix,names,maximum)
    if output.exists():raise EvidenceError('archive download must use a fresh directory')
    remote=tree(archive['repo'],archive['revision'],archive['prefix'],fetch)
    expected={archive['prefix']+'/'+e['path'] for e in archive['inventory']}
    if set(remote)!=expected:raise EvidenceError('public archive inventory differs')
    if download is None:
        from huggingface_hub import hf_hub_download
        download=hf_hub_download
    output.mkdir(parents=True,exist_ok=False)
    for e in archive['inventory']:
        name=archive['prefix']+'/'+e['path']
        if remote[name].get('size')!=e['bytes']:raise EvidenceError('public archive size differs')
        try:
            cached=Path(download(repo_id=archive['repo'],repo_type='dataset',revision=archive['revision'],filename=name,
                token=False,force_download=True,local_files_only=False,cache_dir=output.parent/(output.name+'-transport-cache')))
        except Exception as error:raise EvidenceError('public archive download failed: '+type(error).__name__) from None
        target=confined(output,e['path']);shutil.copyfile(cached,target)
        if target.stat().st_size!=e['bytes'] or file_hash(target)!=e['sha256']:
            raise EvidenceError('actual public download differs from selected bytes')
    return {'schema':'ovl.public-archive-download.v1','result':'PASS','archive_sha256':digest(archive),
            'repo':archive['repo'],'revision':archive['revision'],'files':archive['inventory'],'authentication':'anonymous-token-False',
            'force_download':True,'scope':'complete-selected-public-file-byte-identity'}


def generate(root,environ,output):
    revision=actions_revision(root,environ);name=select_request(root,environ)
    if name is None:raise EvidenceError('no new progress request')
    request=object_at(root,name);policy=validate_request(request)
    if output.exists():raise EvidenceError('preserve prior progress signing output')
    output.mkdir(parents=True,exist_ok=False)
    download_packet(request['registration_request'],output/'registration-packet')
    downloads=[download_archive(request['registration_anchor'],output/'registration-anchor')]
    source=PublisherPolicy(**request['registration_request']['source_policy'])
    registration_check=verify_packet(output/'registration-packet',output/'registration-anchor/registration.sigstore.json',
                                     policy,source,source_checkout=root,policy_origin='ci-self-generated')
    r=object_at(output/'registration-packet','registration.json');regroot=digest(r);envelopes=request['envelopes']
    verify_chain(r,regroot,envelopes,complete=False)
    index=len(envelopes)-1
    if Path(name).stem!=r['run_id']+'-'+r['attempt_id']+f'-boundary-{index:05d}':raise EvidenceError('progress filename differs from run/attempt/index')
    anchors=output/'progress';anchors.mkdir();policies=[]
    for i,v in enumerate(request['previous_progress']):
        downloads.append(download_archive(v['archive'],anchors/f'progress-{i:05d}'))
        policies.append(ProgressPublisherPolicy(**v['policy']))
    prior_check=None;previous=regroot
    if index:
        prior_check=verify_prefix(r,regroot,envelopes[:-1],anchors,policies,complete=False)
        previous=prior_check['closing_statement_sha256']
    value=statement(r,regroot,envelopes,request['checkpoint_archive'],previous)
    downloads.append(download_archive(request['checkpoint_archive'],output/'checkpoint'))
    metadata,tensors=read_state(output/'checkpoint',envelopes[-1]['body']['checkpoint'])
    if unpack(metadata['tree'],tensors).get('control')!=envelopes[-1]['body']['control']:
        raise EvidenceError('downloaded safe state control differs from signed boundary')
    current=anchors/f'progress-{index:05d}';current.mkdir();write_json(current/'statement.json',value)
    own=ProgressPublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,PROGRESS_WORKFLOW,ISSUER,REF,
        revision,digest(value),'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted');own.validate()
    write_json(output/'ci-progress-policies.json',[asdict(p) for p in [*policies,own]])
    write_json(output/'request.json',request)
    write_json(output/'ci-input-checks.json',{'registration':registration_check,'previous_progress':prior_check,'downloads':downloads,
        'scope':'publisher/run ancestry and actual downloaded safe checkpoint bytes; numerical training NOT_RUN',
        'request_uniqueness':'observed Git ancestry only; global anti-equivocation not established'})
    write_json(output/'current.json',{'index':index,'statement':f'progress/progress-{index:05d}/statement.json',
                                    'bundle':f'progress/progress-{index:05d}/statement.sigstore.json'})


def verify_output(output):
    request=read_json(output/'request.json');validate_request(request)
    r=object_at(output/'registration-packet','registration.json')
    policies=[ProgressPublisherPolicy(**v) for v in read_json(output/'ci-progress-policies.json')]
    return verify_prefix(r,digest(r),request['envelopes'],output/'progress',policies,complete=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--select',action='store_true');p.add_argument('--verify-output',action='store_true')
    p.add_argument('--output',type=Path,default=Path('anchor-progress'));a=p.parse_args()
    try:
        if a.select:print('present='+('true' if select_request(Path.cwd(),os.environ) else 'false'))
        elif a.verify_output:write_json(a.output/'ci-verification.json',verify_output(a.output))
        else:generate(Path.cwd(),os.environ,a.output)
    except Exception as error:p.exit(1,'progress endorsement refused: '+str(error)+'\n')

if __name__=='__main__':main()
