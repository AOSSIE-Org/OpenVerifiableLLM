#!/usr/bin/env python3
"""Off-pod checkpoint publication, exact Actions endorsement and verified ack.

One boundary per invocation. No provider lifecycle, no private run key and no
training advancement. The enclosing dispatcher copies the resulting verified
anchor/policy acknowledgement to the paused recorder. Ambiguous writes are
reconciled read-only; missing receipts never authorize advancement.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import publish_evidence_archive as transport
from ovl_pipeline.anchoring import ISSUER,OWNER_ID,REPOSITORY,REPOSITORY_ID,PublisherPolicy
from ovl_pipeline.canonical import EvidenceError,canonical,confined,digest,file_hash,inventory,read_json,verify_inventory,write_json
from ovl_pipeline.production_anchoring import object_at,verify_packet
from ovl_pipeline.production_chain import verify_chain
from ovl_pipeline.production_commitment import verify_code
from ovl_pipeline.production_identity import ProductionPublisherPolicy
from ovl_pipeline.progress_anchoring import PROGRESS_WORKFLOW,ProgressPublisherPolicy,statement,verify_prefix
from ovl_pipeline.progress_commitment import validate_request
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.source_commitment import REF
from ovl_pipeline.state import read_state,unpack

BRANCH=REF.removeprefix('refs/heads/')
REMOTE='https://github.com/'+REPOSITORY+'.git'


def command(args,*,cwd=None,timeout=120):
    try:r=subprocess.run(args,cwd=cwd,capture_output=True,timeout=timeout,check=True)
    except Exception as error:raise EvidenceError('external command failed: '+type(error).__name__) from None
    return r.stdout.decode().strip()


def save_once(path,value):
    if path.exists():
        if read_json(path)!=value:raise EvidenceError('preserved publisher intent differs; do not overwrite')
    else:write_json(path,value)


def published(plan,staging,output):
    """Preserve uncertain writes and adopt only a fully downloaded fixed revision."""
    plan_path=output/'plan.json';output.mkdir(exist_ok=True);save_once(plan_path,plan)
    upload=output/'publication'
    pin=output/'archive.json'
    if pin.exists():
        archive=read_json(pin)
        if {k:archive[k] for k in ('repo','prefix','inventory')}!={'repo':plan['repo'],'prefix':plan['prefix'],'inventory':plan['files']}:
            raise EvidenceError('saved archive selection differs from publication plan')
    else:
        if (upload/'upload.json').exists():receipt=read_json(upload/'upload.json')
        else:receipt=transport.reconcile(plan_path,upload) if upload.exists() else transport.upload(plan_path,staging,upload)
        archive={'repo':plan['repo'],'revision':receipt['revision'],'prefix':plan['prefix'],'inventory':plan['files']}
        save_once(pin,archive)
    fresh=output/('download-'+uuid.uuid4().hex)
    downloaded=transport.download(plan_path,archive['revision'],fresh)
    return archive,fresh/'downloaded',downloaded


def expected_policy(revision,value):
    policy=ProgressPublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,PROGRESS_WORKFLOW,ISSUER,REF,
                                  revision,digest(value),'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    policy.validate();return policy


def copy_anchor(source,destination):
    names=['statement.json','statement.sigstore.json']
    if source.is_symlink() or sorted(p.name for p in source.iterdir())!=names:
        raise EvidenceError('unexpected downloaded anchor files')
    for name in names:
        path=confined(source,name)
        if not path.is_file() or path.stat().st_size>2*1024**2:raise EvidenceError('invalid downloaded anchor file')
    destination.mkdir()
    for name in names:shutil.copyfile(confined(source,name),destination/name)


def request_commit(request,r,directory,*,execute=command):
    """Dedicated clone only; never stage the owner's working tree or force-push."""
    from ovl_pipeline.publication_pause import require_publication_open
    require_publication_open()
    validate_request(request);name=f"project/progress-commitments/{r['run_id']}-{r['attempt_id']}-boundary-{len(request['envelopes'])-1:05d}.json"
    if not directory.exists():directory.mkdir()
    clone=directory/'checkout';intent=directory/'intent.json';saved=directory/'commit.json'
    if not intent.exists():
        if clone.exists():raise EvidenceError('incomplete publisher clone requires inspection; preserved')
        execute(['git','clone','--no-checkout','--single-branch','--branch',BRANCH,REMOTE,str(clone)])
        execute(['git','checkout','--detach','origin/'+BRANCH],cwd=clone)
        if execute(['git','status','--porcelain'],cwd=clone):raise EvidenceError('dedicated publisher checkout must be clean')
        verify_code(clone,r)
        parent=execute(['git','rev-parse','HEAD'],cwd=clone)
        if execute(['git','log','--all','--format=%H','--',name],cwd=clone):raise EvidenceError('progress request identity was previously used')
        save_once(intent,{'schema':'ovl.progress-request-write-intent.v1','parent':parent,'request_sha256':digest(request),'path':name})
    selected=read_json(intent)
    if selected['request_sha256']!=digest(request) or selected['path']!=name:raise EvidenceError('request write intent changed')
    if not saved.exists():
        head=execute(['git','rev-parse','HEAD'],cwd=clone)
        if head!=selected['parent']:raise EvidenceError('unrecorded request commit requires read-only reconciliation')
        target=confined(clone,name)
        if target.exists():raise EvidenceError('unrecorded staged request requires read-only reconciliation')
        write_json(target,request);execute(['git','add','--',name],cwd=clone)
        changed=execute(['git','diff','--cached','--name-only'],cwd=clone)
        if changed!=name:raise EvidenceError('publisher would commit unrelated changes')
        execute(['git','-c','user.name=Rajat Roy','-c','user.email=135772548+ryoari@users.noreply.github.com',
                 'commit','-m',f"Commit public progress boundary {len(request['envelopes'])-1}"],cwd=clone)
        revision=execute(['git','rev-parse','HEAD'],cwd=clone)
        save_once(saved,{'revision':revision,'request_sha256':digest(request),'path':name})
    record=read_json(saved);revision=record['revision']
    if (record['path']!=name or record['request_sha256']!=digest(request) or not re.fullmatch('[0-9a-f]{40}',revision)
        or execute(['git','rev-parse','HEAD'],cwd=clone)!=revision
        or file_hash(confined(clone,name))!=digest(request)
        or execute(['git','show',revision+':'+name],cwd=clone)!=canonical(request).decode()
        or execute(['git','status','--porcelain'],cwd=clone)
        or execute(['git','rev-list','--parents','-n','1',revision],cwd=clone).split()!=[revision,selected['parent']]
        or execute(['git','diff-tree','--no-commit-id','--name-status','-r','--no-renames',revision],cwd=clone)!='A\t'+name):
        raise EvidenceError('saved request commit does not match selected content')
    push=directory/'push-attempt.json'
    if push.exists():
        save_once(push,{'revision':revision,'remote':REMOTE,'ref':REF})
        # A retry never blindly repeats a possibly successful push. Fetching and
        # proving ancestry is read-only with respect to the public repository.
        execute(['git','fetch','origin',BRANCH],cwd=clone)
        execute(['git','merge-base','--is-ancestor',revision,'origin/'+BRANCH],cwd=clone)
    else:
        save_once(push,{'revision':revision,'remote':REMOTE,'ref':REF})
        execute(['git','push','origin','HEAD:'+REF],cwd=clone)
    execute(['git','fetch','origin',BRANCH],cwd=clone)
    execute(['git','merge-base','--is-ancestor',revision,'origin/'+BRANCH],cwd=clone)
    save_once(directory/'public-commit.json',{'revision':revision,'url':'https://github.com/'+REPOSITORY+'/commit/'+revision})
    return revision


def actions_artifact(revision,output,deadline,*,execute=command,wall=time.time,sleep=time.sleep,progress=None):
    """Bounded wait on the exact pushed commit; no workflow rerun on failure."""
    import json
    integer(deadline,1,2**53-1,'publisher deadline');limit=time.monotonic()+max(0,deadline-wall())
    while wall()<deadline and time.monotonic()<limit:
        runs=json.loads(execute(['gh','run','list','--repo',REPOSITORY,'--workflow',PROGRESS_WORKFLOW,
                                '--commit',revision,'--limit','20','--json','databaseId,headSha,status,conclusion']))
        if len(runs)>1:raise EvidenceError('ambiguous progress Actions runs; no automatic selection')
        if runs:
            run=runs[0]
            if run['headSha']!=revision:raise EvidenceError('wrong Actions source identity')
            if progress is not None:
                # These finite successful step transitions are cost liveness,
                # never signer identity or computation verification. Unchanged
                # polls cannot create another event or extend the deadline.
                detail=json.loads(execute(['gh','api',f'repos/{REPOSITORY}/actions/runs/{run["databaseId"]}']))
                if (detail['id']!=run['databaseId'] or detail['run_attempt']!=1 or detail['head_sha']!=revision
                    or detail['head_branch']!=BRANCH or detail['path']!=PROGRESS_WORKFLOW or detail['event']!='push'):
                    raise EvidenceError('publication activity has wrong Actions identity')
                identity={'run_id':run['databaseId'],'revision':revision,'attempt':1}
                progress('actions-run-observed',identity)
                jobs=json.loads(execute(['gh','api',f'repos/{REPOSITORY}/actions/runs/{run["databaseId"]}/jobs?per_page=100']))
                if type(jobs.get('jobs')) is not list or jobs.get('total_count')!=len(jobs['jobs']) or len(jobs['jobs'])>1:
                    raise EvidenceError('ambiguous or truncated publication jobs')
                for job in jobs['jobs']:
                    if job['run_id']!=run['databaseId'] or job['head_sha']!=revision or job['name']!='endorse-progress':
                        raise EvidenceError('wrong publication job identity')
                    steps=job['steps']
                    if type(steps) is not list or len(steps)>32:raise EvidenceError('unbounded publication step list')
                    numbers=set()
                    for step in steps:
                        integer(step['number'],1,32,'publication step number')
                        if step['number'] in numbers:raise EvidenceError('duplicate publication step')
                        numbers.add(step['number'])
                        if step['status']=='completed' and step['conclusion']=='success':
                            progress(f'actions-step-{step["number"]:02d}',{**identity,'job_id':job['id'],'step_number':step['number'],'name':step['name']})
            if run['status']=='completed':
                if run['conclusion']!='success':raise EvidenceError('progress endorsement failed; preserve public run')
                detail=json.loads(execute(['gh','api',f'repos/{REPOSITORY}/actions/runs/{run["databaseId"]}']))
                if (detail['id']!=run['databaseId'] or detail['run_attempt']!=1 or detail['head_sha']!=revision
                    or detail['head_branch']!=BRANCH or detail['path']!=PROGRESS_WORKFLOW or detail['event']!='push'
                    or detail['status']!='completed' or detail['conclusion']!='success'):
                    raise EvidenceError('wrong workflow, source, branch or rerun Actions identity')
                name=f"pipeline-production-progress-{revision}-1"
                if output.exists():raise EvidenceError('Actions download requires fresh output')
                execute(['gh','run','download',str(run['databaseId']),'--repo',REPOSITORY,'--name',name,'--dir',str(output)],timeout=180)
                return {'run_id':run['databaseId'],'revision':revision,'attempt':1,'artifact_name':name}
        sleep(min(15,max(0,deadline-wall())))
    raise EvidenceError('publisher deadline reached; checkpoint remains preserved and unacknowledged')


def _publish(packet,bundle,production_policy,source_policy,source_checkout,config,chain_directory,
            previous_directory,previous_policies,output,deadline):
    integer(deadline,1,2**53-1,'publisher deadline')
    if time.time()>=deadline:raise EvidenceError('publisher deadline expired before publication')
    fields(config,'schema registration_request registration_anchor','progress dispatcher configuration')
    if config['schema']!='ovl.progress-dispatch.v1':raise EvidenceError('unsupported dispatcher configuration')
    verification=verify_packet(packet,bundle,production_policy,source_policy,source_checkout=source_checkout)
    r=object_at(packet,'registration.json');root=digest(r)
    if config['registration_request']['registration_sha256']!=root:raise EvidenceError('dispatcher selected another registration')
    from ovl_pipeline.production_commitment import validate_request as registration_request
    registration_request(config['registration_request'])
    if config['registration_request']['source_policy']!=asdict(source_policy):raise EvidenceError('dispatcher source policy differs from operator selection')
    verify_inventory(packet,config['registration_request']['packet']['inventory'])
    chain=read_json(confined(chain_directory,'chain.json'))
    fields(chain,'schema complete boundaries','recorded progress chain')
    if chain['schema']!='ovl.production-chain.v1' or type(chain['complete']) is not bool:raise EvidenceError('unsupported recorded chain')
    envelopes=chain['boundaries'];verify_chain(r,root,envelopes,complete=chain['complete'])
    index=len(envelopes)-1;body=envelopes[-1]['body']
    waiting=read_json(confined(chain_directory,'awaiting-anchor.json'))
    expected={'schema':'ovl.awaiting-public-progress.v1','registration_sha256':root,'index':index,
              'boundary_sha256':digest(envelopes[-1]),'checkpoint_path':body['checkpoint_path'],'checkpoint':body['checkpoint']}
    if waiting!=expected:raise EvidenceError('waiting checkpoint differs from selected chain')
    if len(previous_policies)!=index:raise EvidenceError('exact previous external policies required')
    prior=[];previous=root
    if index:
        checked=verify_prefix(r,root,envelopes[:-1],previous_directory,previous_policies,complete=False)
        previous=checked['closing_statement_sha256']
        for i,policy in enumerate(previous_policies):
            # Archive locators are operator-retained receipts, separate from the
            # signed anchor directory. Every prior anchor is verified above.
            entry=read_json(confined(output.parent,f'boundary-{i:05d}/ack.json'))
            if entry['policy']!=asdict(policy) or entry['boundary_sha256']!=digest(envelopes[i]):
                raise EvidenceError('previous publication acknowledgement ancestry differs')
            prior.append({'archive':entry['archive'],'policy':entry['policy']})
    checkpoint=confined(chain_directory,body['checkpoint_path'])
    if read_json(confined(checkpoint,'checkpoint.json'))!=body['checkpoint']:raise EvidenceError('checkpoint marker differs')
    md,ts=read_state(checkpoint,body['checkpoint'])
    if unpack(md['tree'],ts)['control']!=body['control']:raise EvidenceError('checkpoint control differs')
    output.mkdir(parents=True,exist_ok=True)
    save_once(output/'intent.json',{'schema':'ovl.progress-dispatch-intent.v1','registration_sha256':root,
              'boundary_sha256':digest(envelopes[-1]),'config_sha256':digest(config),'prior_policies_sha256':digest([asdict(p) for p in previous_policies]),
              'deadline_epoch':deadline})
    def activity(stage,identity):
        from publication_activity import emit
        emit(output/'activity',root,digest(envelopes[-1]),stage,identity,deadline)
    cpplan={'schema':'ovl.evidence-publication-plan.v1','repo':transport.REPO,'kind':'checkpoint',
            'prefix':f'production-checkpoints/{root}/{body["checkpoint_path"]}',
            'subject_sha256':digest(body['checkpoint']),'files':inventory(checkpoint,['checkpoint.json','state.json','state.safetensors'])}
    archive,downloaded,cpdownload=published(cpplan,checkpoint,output/'checkpoint')
    md,ts=read_state(downloaded,body['checkpoint'])
    if unpack(md['tree'],ts)['control']!=body['control']:raise EvidenceError('downloaded checkpoint control differs')
    activity('checkpoint-public-download-verified',{'archive':archive,'checkpoint':body['checkpoint']})
    request={'schema':'ovl.progress-signing-request.v1',**{k:config[k] for k in ('registration_request','registration_anchor')},
             'registration_policy':asdict(production_policy),'envelopes':envelopes,'previous_progress':prior,'checkpoint_archive':archive}
    validate_request(request)
    value=statement(r,root,envelopes,archive,previous)
    save_once(output/'expected-statement.json',value)
    revision=request_commit(request,r,output/'git-request')
    policy=expected_policy(revision,value);save_once(output/'operator-policy.json',asdict(policy))
    activity('request-public-commit-verified',{'revision':revision,'request_sha256':digest(request)})
    artifact=output/('actions-download-'+uuid.uuid4().hex)
    actions=actions_artifact(revision,artifact,deadline,progress=activity)
    anchors=output/('checked-prefix-'+uuid.uuid4().hex);anchors.mkdir()
    for i in range(index):copy_anchor(confined(previous_directory,f'progress-{i:05d}'),anchors/f'progress-{i:05d}')
    current=anchors/f'progress-{index:05d}'
    copy_anchor(confined(artifact,f'progress/progress-{index:05d}'),current)
    if read_json(current/'statement.json')!=value:raise EvidenceError('Actions statement differs from preselected operator expectation')
    local_check=verify_prefix(r,root,envelopes,anchors,[*previous_policies,policy],complete=False)
    activity('actions-anchor-signature-verified',{'policy':asdict(policy),'statement_sha256':digest(value)})
    plan={'schema':'ovl.evidence-publication-plan.v1','repo':transport.REPO,'kind':'progress-anchor',
          'prefix':f'production-progress/{root}/progress-{index:05d}','subject_sha256':digest(value),
          'files':inventory(current,['statement.json','statement.sigstore.json'])}
    public,downloaded,anchor_download=published(plan,current,output/'anchor')
    # Verify the actual public bytes, not merely the temporary Actions artifact.
    public_prefix=output/('public-prefix-'+uuid.uuid4().hex);public_prefix.mkdir()
    for i in range(index):copy_anchor(confined(previous_directory,f'progress-{i:05d}'),public_prefix/f'progress-{i:05d}')
    copy_anchor(downloaded,public_prefix/f'progress-{index:05d}')
    checked=verify_prefix(r,root,envelopes,public_prefix,[*previous_policies,policy],complete=False)
    activity('anchor-public-download-verified',{'archive':public,'policy':asdict(policy),'closing_statement_sha256':checked['closing_statement_sha256']})
    if time.time()>=deadline:raise EvidenceError('publisher deadline expired; preserve public evidence without acknowledging advancement')
    ack={'schema':'ovl.verified-public-progress-ack.v1','registration_sha256':root,'index':index,
         'boundary_sha256':digest(envelopes[-1]),'checkpoint_archive':archive,'archive':public,'policy':asdict(policy),
         'registration_check':verification,'actions':actions,'actions_prefix_check':local_check,
         'checkpoint_download':cpdownload,'anchor_download':anchor_download,'public_prefix_check':checked,
         'anchor_directory':str(public_prefix.resolve()),'training_replay':'NOT_RUN'}
    save_once(output/'ack.json',ack);return ack


def publish(packet,bundle,production_policy,source_policy,source_checkout,config,chain_directory,
            previous_directory,previous_policies,output,deadline):
    from ovl_pipeline.publication_pause import require_publication_open
    require_publication_open()
    output.mkdir(parents=True,exist_ok=True)
    fd=transport.lease(output/'.publisher.lock')
    try:
        if (output/'ack.json').exists():raise EvidenceError('completed publication acknowledgement already exists; verify and adopt it instead of publishing again')
        return _publish(packet,bundle,production_policy,source_policy,source_checkout,config,chain_directory,
                        previous_directory,previous_policies,output,deadline)
    finally:os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','registration-bundle','production-policy','source-policy','source-checkout','config','chain-directory',
                 'previous-directory','previous-policies','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--deadline',required=True,type=int);a=p.parse_args()
    try:
        result=publish(a.packet,a.registration_bundle,ProductionPublisherPolicy(**read_json(a.production_policy)),
            PublisherPolicy(**read_json(a.source_policy)),a.source_checkout,read_json(a.config),a.chain_directory,
            a.previous_directory,[ProgressPublisherPolicy(**v) for v in read_json(a.previous_policies)],a.output,a.deadline)
        print('Verified public progress acknowledgement '+digest(result))
    except Exception as error:p.exit(1,'progress dispatch refused: '+type(error).__name__+'; preserve output for reconciliation\n')

if __name__=='__main__':main()
