#!/usr/bin/env python3
"""Public precommitment publication; no provisioning or training authorization.

Check the source signature, complete parent reports and source bytes; archive and
freshly download the exact packet; append one registration request in a dedicated
clone; verify the exact Actions identity and both signatures after publishing the
bundle. Recovery adopts original writes and deadlines, never reissues ambiguous
pushes. This endorses operator reports, not their truth or independent replay.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import re
import shutil
import time

from ovl_pipeline.anchoring import ISSUER,OWNER_ID,REPOSITORY,REPOSITORY_ID,PublisherPolicy
from ovl_pipeline.canonical import EvidenceError,canonical,confined,digest,file_hash,inventory,read_json,write_json
from ovl_pipeline.production_anchoring import PACKET_FILES,check_source_parents,verify_packet
from ovl_pipeline.production_commitment import validate_request,verify_code
from ovl_pipeline.production_identity import PRODUCTION_WORKFLOW,ProductionPublisherPolicy
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.source_commitment import REF
from publish_progress_boundary import command,deadline_command,save_once,published
from publish_evidence_archive import REPO

BRANCH=REF.removeprefix('refs/heads/')
REMOTE='https://github.com/'+REPOSITORY+'.git'


def expected_policy(revision,registration):
    policy=ProductionPublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,PRODUCTION_WORKFLOW,ISSUER,REF,
        revision,digest(registration),'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    policy.validate();return policy


def request_commit(request,r,directory,*,execute=command):
    """Dedicated clone only; never stage the owner's working tree or force-push."""
    from ovl_pipeline.publication_pause import require_publication_open
    require_publication_open()
    validate_request(request);name=f"project/production-commitments/{r['run_id']}-{r['attempt_id']}.json"
    if not directory.exists():directory.mkdir()
    clone=directory/'checkout';intent=directory/'intent.json';saved=directory/'commit.json'
    if not intent.exists():
        if clone.exists():raise EvidenceError('incomplete publisher clone requires inspection; preserved')
        execute(['git','clone','--no-checkout','--single-branch','--branch',BRANCH,REMOTE,str(clone)])
        execute(['git','checkout','--detach','origin/'+BRANCH],cwd=clone)
        if execute(['git','status','--porcelain'],cwd=clone):raise EvidenceError('dedicated publisher checkout must be clean')
        verify_code(clone,r)
        parent=execute(['git','rev-parse','HEAD'],cwd=clone)
        if execute(['git','log','--all','--format=%H','--',name],cwd=clone):raise EvidenceError('production request identity was previously used')
        save_once(intent,{'schema':'ovl.production-request-write-intent.v1','parent':parent,'request_sha256':digest(request),'path':name})
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
                 'commit','-m','Commit public production registration'],cwd=clone,timeout=600)
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
        execute(['git','push','origin','HEAD:'+REF],cwd=clone,timeout=600)
    execute(['git','fetch','origin',BRANCH],cwd=clone)
    execute(['git','merge-base','--is-ancestor',revision,'origin/'+BRANCH],cwd=clone)
    save_once(directory/'public-commit.json',{'revision':revision,'url':'https://github.com/'+REPOSITORY+'/commit/'+revision})
    return revision


def actions_artifact(revision,output,deadline,*,execute=command,wall=time.time,sleep=time.sleep,progress=None):
    """Bounded wait on the exact pushed commit; no workflow rerun on failure."""
    import json
    integer(deadline,1,2**53-1,'publisher deadline');limit=time.monotonic()+max(0,deadline-wall())
    while wall()<deadline and time.monotonic()<limit:
        runs=json.loads(execute(['gh','run','list','--repo',REPOSITORY,'--workflow',PRODUCTION_WORKFLOW,
                                '--commit',revision,'--limit','20','--json','databaseId,headSha,status,conclusion']))
        if len(runs)>1:raise EvidenceError('ambiguous registration Actions runs; no automatic selection')
        if runs:
            run=runs[0]
            if run['headSha']!=revision:raise EvidenceError('wrong Actions source identity')
            if progress is not None:
                # These finite successful step transitions are cost liveness,
                # never signer identity or computation verification. Unchanged
                # polls cannot create another event or extend the deadline.
                detail=json.loads(execute(['gh','api',f'repos/{REPOSITORY}/actions/runs/{run["databaseId"]}']))
                if (detail['id']!=run['databaseId'] or detail['run_attempt']!=1 or detail['head_sha']!=revision
                    or detail['head_branch']!=BRANCH or detail['path']!=PRODUCTION_WORKFLOW or detail['event']!='push'):
                    raise EvidenceError('publication activity has wrong Actions identity')
                identity={'run_id':run['databaseId'],'revision':revision,'attempt':1}
                progress('actions-run-observed',identity)
                jobs=json.loads(execute(['gh','api',f'repos/{REPOSITORY}/actions/runs/{run["databaseId"]}/jobs?per_page=100']))
                if type(jobs.get('jobs')) is not list or jobs.get('total_count')!=len(jobs['jobs']) or len(jobs['jobs'])>1:
                    raise EvidenceError('ambiguous or truncated publication jobs')
                for job in jobs['jobs']:
                    if job['run_id']!=run['databaseId'] or job['head_sha']!=revision or job['name']!='endorse-registration':
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
                if run['conclusion']!='success':raise EvidenceError('registration endorsement failed; preserve public run')
                detail=json.loads(execute(['gh','api',f'repos/{REPOSITORY}/actions/runs/{run["databaseId"]}']))
                if (detail['id']!=run['databaseId'] or detail['run_attempt']!=1 or detail['head_sha']!=revision
                    or detail['head_branch']!=BRANCH or detail['path']!=PRODUCTION_WORKFLOW or detail['event']!='push'
                    or detail['status']!='completed' or detail['conclusion']!='success'):
                    raise EvidenceError('wrong workflow, source, branch or rerun Actions identity')
                name=f"pipeline-production-registration-{revision}-1"
                if output.exists():raise EvidenceError('Actions download requires fresh output')
                execute(['gh','run','download',str(run['databaseId']),'--repo',REPOSITORY,'--name',name,'--dir',str(output)],timeout=180)
                return {'run_id':run['databaseId'],'revision':revision,'attempt':1,'artifact_name':name}
        sleep(min(15,max(0,deadline-wall())))
    raise EvidenceError('publisher deadline reached; checkpoint remains preserved and unacknowledged')


def publish(packet,expected_registration,source_policy,source_checkout,output,deadline,*,progress=None):
    from ovl_pipeline.publication_pause import require_publication_open
    require_publication_open()
    integer(deadline,1,2**53-1,'registration publication deadline')
    if time.time()>=deadline:raise EvidenceError('registration publication deadline expired')
    if type(source_policy) is not PublisherPolicy:raise EvidenceError('independent source publisher policy required')
    r,source_check=check_source_parents(packet,source_policy,policy_origin='operator-reconstructed-from-source')
    if digest(r)!=expected_registration:raise EvidenceError('registration differs from external selection')
    verify_code(source_checkout,r)
    if any(p.is_symlink() for path in (packet,source_checkout,output) for p in [path,*path.absolute().parents]):
        raise EvidenceError('registration publication paths must be regular')
    output.mkdir(parents=True,exist_ok=True)
    identity={'schema':'ovl.registration-publication-selection.v1','registration_sha256':expected_registration,
        'packet_inventory':inventory(packet,sorted(PACKET_FILES)),'source_policy':asdict(source_policy),
        'source_checkout':str(source_checkout.resolve()),'deadline_epoch':deadline}
    save_once(output/'selection.json',identity)
    bounded_command=deadline_command(deadline,execute=command)
    plan={'schema':'ovl.evidence-publication-plan.v1','repo':REPO,'kind':'registration-packet',
        'prefix':'production-registration/'+expected_registration,'subject_sha256':expected_registration,
        'files':identity['packet_inventory']}
    archive,downloaded,receipt=published(plan,packet,output/'packet-publication',deadline=deadline)
    if time.time()>=deadline:raise EvidenceError('packet publication exceeded original deadline')
    if progress is not None:progress('checkpoint-public-download-verified',{'kind':'registration-packet','archive':archive})
    request={'schema':'ovl.production-signing-request.v1','registration_sha256':expected_registration,
        'packet':archive,'source_policy':asdict(source_policy)}
    validate_request(request);save_once(output/'request.json',request)
    revision=request_commit(request,r,output/'request-commit',execute=bounded_command)
    if progress is not None:progress('request-public-commit-verified',{'revision':revision,'request_sha256':digest(request)})
    policy=expected_policy(revision,r)
    save_once(output/'operator-selected-policy.json',asdict(policy))
    action_receipt=output/'actions.json';action_directory=output/'actions'
    if not action_receipt.exists():
        result=actions_artifact(revision,action_directory,deadline,execute=bounded_command,progress=progress)
        save_once(action_receipt,result)
    action=read_json(action_receipt)
    if (action.get('revision')!=revision or action.get('attempt')!=1
        or action.get('artifact_name')!=f'pipeline-production-registration-{revision}-1'):
        raise EvidenceError('saved Actions artifact identity differs')
    # CI policy is not a trust root. Recompute both signatures against the policy
    # selected above from the exact public request commit and owner identity.
    bundle=confined(action_directory,'registration.sigstore.json')
    checked=verify_packet(downloaded,bundle,policy,source_policy,
        policy_origin='operator-reconstructed-from-source',source_checkout=source_checkout)
    if progress is not None:progress('actions-anchor-signature-verified',{'policy':asdict(policy),'bundle_sha256':file_hash(bundle)})
    staging=output/'anchor-staging';staging.mkdir(exist_ok=True)
    dest=staging/'registration.sigstore.json'
    if dest.exists():
        if dest.is_symlink() or file_hash(dest)!=file_hash(bundle):raise EvidenceError('retained registration anchor changed')
    else:shutil.copyfile(bundle,dest)
    anchor_plan={'schema':'ovl.evidence-publication-plan.v1','repo':REPO,'kind':'registration-anchor',
        'prefix':'production-anchors/'+expected_registration,'subject_sha256':expected_registration,
        'files':inventory(staging,['registration.sigstore.json'])}
    if time.time()>=deadline:raise EvidenceError('original deadline forbids anchor publication')
    anchor,downloaded_anchor,anchor_receipt=published(anchor_plan,staging,output/'anchor-publication',deadline=deadline)
    final=verify_packet(downloaded,downloaded_anchor/'registration.sigstore.json',policy,source_policy,
        policy_origin='operator-reconstructed-from-source',source_checkout=source_checkout)
    if time.time()>=deadline:raise EvidenceError('registration publication completed after original deadline')
    if progress is not None:progress('anchor-public-download-verified',{'archive':anchor,'policy':asdict(policy)})
    result={'schema':'ovl.public-registration-publication.v1','result':'PASS','registration_sha256':expected_registration,
        'request_sha256':digest(request),'request_revision':revision,'production_policy':asdict(policy),
        'packet':archive,'registration_anchor':anchor,
        'scope':'public downloads, independently selected publisher identity and report parents; computation truth not established',
        'production_admission':'NOT_RUN','independent_third_party':False}
    # Observations are content addressed because a fresh signature/download check
    # can contain a later operator timestamp. The original selections never move.
    observation={'packet_download':receipt,'anchor_download':anchor_receipt,'signature_and_parent_checks':final}
    write_json(output/('verification-'+digest(observation)+'.json'),observation)
    save_once(output/'verified-registration.json',result)
    config={'schema':'ovl.progress-dispatch.v1','registration_request':request,'registration_anchor':anchor}
    save_once(output/'progress-publisher-config.json',config)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','source-policy','source-checkout','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--registration-sha256',required=True);p.add_argument('--deadline',required=True,type=int);a=p.parse_args()
    try:
        result=publish(a.packet,a.registration_sha256,PublisherPolicy(**read_json(a.source_policy)),
            a.source_checkout,a.output,a.deadline)
        print(canonical(result).decode());return 0
    except Exception as error:
        p.exit(1,'registration publication refused: '+type(error).__name__+'; preserve original writes and receipts\n')


if __name__=='__main__':raise SystemExit(main())
