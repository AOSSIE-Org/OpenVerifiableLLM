"""Append-only Actions endorsement of a public production registration packet.

This authenticates the source anchor and checks report relationships, source code
and dependency pins. Endorsing operator reports does not verify their computations.
The production executor must additionally satisfy its live admission gates.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import re
import subprocess

from .anchoring import ISSUER,OWNER_ID,REPOSITORY,REPOSITORY_ID,PublisherPolicy
from .canonical import EvidenceError,confined,digest,file_hash,sha256,write_json
from .production_anchoring import PACKET_FILES,check_source_parents,object_at
from .production_identity import PRODUCTION_WORKFLOW,ProductionPublisherPolicy
from .schema import fields,integer
from .source_commitment import REF,actions_revision,fetch_metadata,git,tree

REQUEST_DIRECTORY='project/production-commitments'


def select_request(root,environ):
    actions_revision(root,environ)
    if len(git(root,'rev-list','--parents','-n','1','HEAD').split())!=2:
        raise EvidenceError('production signing requires single-parent commit')
    changes=git(root,'diff-tree','--no-commit-id','--name-status','-r','--no-renames','HEAD','--',REQUEST_DIRECTORY)
    if not changes:return None
    lines=changes.splitlines()
    if len(lines)!=1 or not lines[0].startswith('A\t'):raise EvidenceError('exactly one new append-only production request required')
    name=lines[0].split('\t')[1]
    if not re.fullmatch(REQUEST_DIRECTORY+r'/[a-z0-9][a-z0-9-]+\.json',name):raise EvidenceError('invalid production request filename')
    if len(git(root,'log','--full-history','--format=%H','HEAD','--',name).splitlines())!=1:raise EvidenceError('production request identity reused')
    touched=git(root,'diff-tree','--no-commit-id','--name-only','-r','--no-renames','HEAD').splitlines()
    if touched!=[name]:raise EvidenceError('signing commit must change only the append-only request')
    return name


def validate_request(request):
    fields(request,'schema registration_sha256 packet source_policy','production signing request')
    if request['schema']!='ovl.production-signing-request.v1':raise EvidenceError('unsupported production signing request')
    from .canonical import require_digest
    require_digest(request['registration_sha256'])
    p=request['packet'];fields(p,'repo revision prefix inventory','public production packet')
    if (type(p['repo']) is not str or not re.fullmatch(r'AOSSIE/openverifiable-[a-z0-9-]+-evidence',p['repo'])
            or type(p['revision']) is not str or not re.fullmatch('[0-9a-f]{40}',p['revision'])
            or type(p['prefix']) is not str or not re.fullmatch(r'production-registration/[0-9a-f]{64}',p['prefix'])):
        raise EvidenceError('packet must use pinned approved public archive')
    if p['prefix']!='production-registration/'+request['registration_sha256']:
        raise EvidenceError('packet prefix must bind registration root')
    inv=p['inventory']
    if type(inv) is not list or len(inv)!=len(PACKET_FILES):raise EvidenceError('complete bounded production packet required')
    names=[]
    for e in inv:
        fields(e,'path bytes sha256','packet file');require_digest(e['sha256'])
        integer(e['bytes'],1,2*1024*1024 if e['path']=='source-statement.sigstore.json' else 16*1024*1024,'packet file length')
        names.append(e['path'])
    if names!=sorted(PACKET_FILES):raise EvidenceError('packet files must be exact sorted closed inventory')
    if next(e['sha256'] for e in inv if e['path']=='registration.json')!=request['registration_sha256']:
        raise EvidenceError('request registration root mismatch')
    policy=PublisherPolicy(**request['source_policy']);policy.validate()
    return policy


def download_packet(request,output,*,fetch=fetch_metadata):
    validate_request(request)
    if output.exists():raise EvidenceError('packet download output must be fresh')
    p=request['packet'];remote=tree(p['repo'],p['revision'],p['prefix'],fetch)
    if set(remote)!={p['prefix']+'/'+e['path'] for e in p['inventory']}:raise EvidenceError('public packet inventory differs')
    output.mkdir(parents=True,exist_ok=False)
    for e in p['inventory']:
        if remote[p['prefix']+'/'+e['path']].get('size')!=e['bytes']:raise EvidenceError('public packet size differs')
        data=fetch(f"https://huggingface.co/datasets/{p['repo']}/resolve/{p['revision']}/{p['prefix']}/{e['path']}")
        if len(data)!=e['bytes'] or sha256(data)!=e['sha256']:raise EvidenceError('downloaded packet bytes differ')
        confined(output,e['path']).write_bytes(data)


def verify_code(root,registration):
    """Compare trusted checkout to the declared earlier pilot/kernel commit."""
    revision=registration['code_revision']
    if not re.fullmatch('[0-9a-f]{40}',revision):raise EvidenceError('invalid code revision')
    if subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor',revision,'HEAD'],capture_output=True).returncode:
        raise EvidenceError('production code revision must be an ancestor of signing commit')
    entries=[]
    listing=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','-z',revision,'--','src/ovl_pipeline','src/model.py']).decode()
    for line in filter(None,listing.split('\0')):
        meta,name=line.split('\t',1);mode,kind,oid=meta.split()
        if mode not in ('100644','100755') or kind!='blob':raise EvidenceError('code tree contains nonregular source')
        if not name.endswith('.py'):continue
        if name!='src/model.py' and '/' in name[len('src/ovl_pipeline/'):]:raise EvidenceError('unsupported nested production module')
        data=subprocess.check_output(['git','-C',str(root),'show',revision+':'+name])
        if not confined(root,name).is_file():raise EvidenceError('declared production source missing from checkout')
        if file_hash(confined(root,name))!=sha256(data):raise EvidenceError('production source differs from pilot commit')
        entries.append({'path':name.removeprefix('src/'),'sha256':sha256(data)})
    # Match training.code_root order (pipeline files then shared model.py).
    entries.sort(key=lambda e:(e['path']=='model.py',e['path']))
    from .training import code_root
    if not entries or digest(entries)!=registration['code_root'] or code_root()!=registration['code_root']:
        raise EvidenceError('declared production code root mismatch')
    lock='requirements/gpu.lock'
    raw=subprocess.check_output(['git','-C',str(root),'show',revision+':'+lock])
    if sha256(raw)!=registration['runtime']['dependency_lock_sha256'] or file_hash(confined(root,lock))!=sha256(raw):
        raise EvidenceError('GPU dependency lock differs from code commit')
    return {'result':'PASS','scope':'source-bytes-and-dependency-lock-only','code_revision':revision,'code_root':digest(entries)}


def generate(root,environ,output):
    revision=actions_revision(root,environ);name=select_request(root,environ)
    if name is None:raise EvidenceError('no new production signing request')
    request=object_at(root,name);source_policy=validate_request(request)
    if output.exists():raise EvidenceError('preserve prior signing output')
    output.mkdir(parents=True,exist_ok=False)
    download_packet(request,output/'packet')
    registration,checks=check_source_parents(output/'packet',source_policy,policy_origin='ci-self-generated')
    if Path(name).stem!=registration['run_id']+'-'+registration['attempt_id']:raise EvidenceError('request name differs from registered run/attempt')
    if digest(registration)!=request['registration_sha256']:raise EvidenceError('registration differs from signing request')
    # CPU preparation environment is available in this job. This independently
    # checks the frozen source contract; it does not rerun corpus transformations.
    from .preparation import validate_contract
    validate_contract(object_at(output/'packet','source-statement.json'))
    checks['code']=verify_code(root,registration)
    checks['request_uniqueness_scope']='observed Git ancestry only; global equivocation prevention NOT_ESTABLISHED'
    policy=ProductionPublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,PRODUCTION_WORKFLOW,ISSUER,REF,
        revision,digest(registration),'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    policy.validate()
    write_json(output/'ci-source-policy.json',asdict(source_policy))
    write_json(output/'ci-self-check-policy.json',asdict(policy))
    write_json(output/'ci-parent-checks.json',checks)
    write_json(output/'request.json',request)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--select',action='store_true')
    p.add_argument('--output',type=Path,default=Path('anchor-production'));a=p.parse_args()
    try:
        if a.select:print('present='+('true' if select_request(Path.cwd(),os.environ) else 'false'))
        else:generate(Path.cwd(),os.environ,a.output)
    except Exception as e:p.exit(1,f'production endorsement refused: {e}\n')


if __name__=='__main__':main()
