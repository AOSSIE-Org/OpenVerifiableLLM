"""Public release endorsement of operator assertions, with exact ancestry checks.

The workflow rechecks signatures and actual final checkpoint bytes. It does not
perform raw reconstruction or CUDA replay and must not claim those computations.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import re

from .anchoring import ISSUER,OWNER_ID,REPOSITORY,REPOSITORY_ID,PublisherPolicy,verify_anchor
from .canonical import EvidenceError,confined,digest,read_json,parse_json,write_json
from .production_anchoring import object_at,verify_packet
from .production_commitment import download_packet
from .production_identity import ProductionPublisherPolicy
from .production_release import RELEASE_WORKFLOW,ReleasePublisherPolicy,validate,reports,model_repo
from .progress_anchoring import ProgressPublisherPolicy,verify_prefix
from .progress_commitment import download_archive
from .schema import fields
from .source_commitment import REF,actions_revision,git
from .state import read_state,unpack,tensor_digest

REQUEST_DIRECTORY='project/release-commitments'


def select_request(root,environ):
    actions_revision(root,environ)
    if len(git(root,'rev-list','--parents','-n','1','HEAD').split())!=2:raise EvidenceError('release request requires one parent')
    lines=git(root,'diff-tree','--no-commit-id','--name-status','-r','--no-renames','HEAD','--',REQUEST_DIRECTORY).splitlines()
    if not lines:return None
    if len(lines)!=1 or not lines[0].startswith('A\t'):raise EvidenceError('exactly one append-only release request required')
    name=lines[0].split('\t')[1]
    if not re.fullmatch(REQUEST_DIRECTORY+r'/[a-z0-9][a-z0-9-]+\.json',name):raise EvidenceError('invalid release request name')
    if len(git(root,'log','--full-history','--format=%H','HEAD','--',name).splitlines())!=1:raise EvidenceError('release request identity reused')
    touched=git(root,'diff-tree','--no-commit-id','--name-only','-r','--no-renames','HEAD').splitlines()
    if touched!=[name]:raise EvidenceError('signing commit must change only the append-only request')
    return name


def policies(context):
    """For CI's labelled self-check only. Consumers supply policies separately."""
    c=context['closing_request']
    return (ProductionPublisherPolicy(**c['registration_policy']),PublisherPolicy(**c['registration_request']['source_policy']),
            [ProgressPublisherPolicy(**v['policy']) for v in [*c['previous_progress'],context['final_progress']]])


def verify_parents(value,evidence,output,source_checkout,production_policy,source_policy,progress_policies,*,
                   complete_checkpoints=False,policy_origin='caller-supplied'):
    validate(value)
    if output.exists():raise EvidenceError('public ancestry verification needs a fresh output directory')
    from .canonical import verify_inventory
    verify_inventory(evidence,value['evidence']['inventory'])
    context=read_json(evidence/'context.json');fields(context,'schema closing_request final_progress','release navigation')
    if context['schema']!='ovl.release-context.v1':raise EvidenceError('unsupported release navigation')
    c=context['closing_request']
    selected=policies(context)
    if (type(production_policy) is not ProductionPublisherPolicy or type(source_policy) is not PublisherPolicy
        or type(progress_policies) is not list or any(type(p) is not ProgressPublisherPolicy for p in progress_policies)
        or asdict(production_policy)!=asdict(selected[0]) or asdict(source_policy)!=asdict(selected[1])
        or [asdict(p) for p in progress_policies]!=[asdict(p) for p in selected[2]]):
        raise EvidenceError('release ancestry policies differ from separate operator selection')
    if type(complete_checkpoints) is not bool:raise EvidenceError('explicit checkpoint download scope required')
    output.mkdir(parents=True,exist_ok=False)
    download_packet(c['registration_request'],output/'packet')
    downloads=[download_archive(c['registration_anchor'],output/'registration-anchor')]
    checked=verify_packet(output/'packet',output/'registration-anchor/registration.sigstore.json',production_policy,source_policy,
                          source_checkout=source_checkout,policy_origin=policy_origin)
    r=object_at(output/'packet','registration.json');root=digest(r)
    if (root!=value['registration_sha256'] or r['source_statement_sha256']!=value['source_statement_sha256']
        or r['preparation_sha256']!=value['preparation_sha256'] or r['code_revision']!=value['code']['revision']
        or r['code_root']!=value['code']['code_root']):raise EvidenceError('release registration/source/code parents differ')
    report_values=reports(r,evidence)
    for name in ('verification','reconstruction','replay','evaluation','exports'):
        if digest(report_values[name])!=value['report_roots'][name]:raise EvidenceError('release report parent differs')
    source=object_at(output/'packet','source-statement.json')
    expected_raw={'path_supplied_by':'caller','archive_repo':source['archive']['repo'],'archive_revision':source['archive']['revision'],
                  'archive_prefix':source['archive']['prefix'],'complete_inventory_sha256':digest(source['archive']['inventory']),
                  'all_local_bytes_rehashed':'PASS','public_anonymous_download_this_command':'NOT_RUN'}
    if report_values['verification']['raw_inputs']!=expected_raw:raise EvidenceError('raw-input provenance scope differs from source archive')
    prepared=object_at(output/'packet','preparation.json');validation=prepared['streams']['conversation-validation']
    if report_values['evaluation']['stream_sha256']!=digest(validation):raise EvidenceError('evaluation stream is not registered held-out data')
    for phase in ('base','chat'):
        if (value['models'][phase]['repo']!=model_repo(r,phase,value['publication_id'])
            or report_values['evaluation']['models'][phase]['targets']!=validation['targets']
            or report_values['replay'][phase+'_model_root']!=value['models'][phase]['model_root']):
            raise EvidenceError('release evaluation/model mapping differs from registered reports')
    progress=output/'progress';progress.mkdir();envelopes=c['envelopes']
    for i,entry in enumerate([*c['previous_progress'],context['final_progress']]):
        downloads.append(download_archive(entry['archive'],progress/f'progress-{i:05d}'))
    anchors=verify_prefix(r,root,envelopes,progress,progress_policies,complete=True)
    checkpoints=output/'chain';checkpoints.mkdir();endpoints={};downloaded_indices=[]
    for i,env in enumerate(envelopes):
        body=env['body']
        if not complete_checkpoints and body['kind'] not in ('base','final'):continue
        statement=read_json(progress/f'progress-{i:05d}/statement.json')
        downloads.append(download_archive(statement['archive'],checkpoints/body['checkpoint_path']))
        md,ts=read_state(checkpoints/body['checkpoint_path'],body['checkpoint']);state=unpack(md['tree'],ts)
        if state['control']!=body['control']:raise EvidenceError('downloaded checkpoint control differs')
        downloaded_indices.append(i)
        if body['kind'] in ('base','final'):
            phase='base' if body['kind']=='base' else 'chat';actual=tensor_digest(state['model'])
            if actual!=value['models'][phase]['model_root']:raise EvidenceError('released model root differs from public phase checkpoint')
            endpoints[phase]={'boundary_sha256':digest(env),'checkpoint_state_root':body['checkpoint']['state_root'],'model_root':actual}
    if set(endpoints)!={'base','chat'}:raise EvidenceError('missing both final model checkpoint downloads')
    # A partial checkpoint download cannot masquerade as a full replay input.
    if complete_checkpoints:
        write_json(checkpoints/'chain.json',{'schema':'ovl.production-chain.v1','complete':True,'boundaries':envelopes})
    result={'schema':'ovl.release-ancestry-verification.v1','result':'PASS','release_sha256':digest(value),
            'registration':checked,'progress':anchors,'downloads':downloads,'final_models':endpoints,
            'checkpoint_indices_downloaded':downloaded_indices,'complete_checkpoints_downloaded':complete_checkpoints,
            'operator_assertion_relationships':'PASS','scope':'publisher-report-ancestry-and-actual-checkpoint-byte-identity',
            'raw_reconstruction_performed':False,'training_replay_performed':False,'independent_third_party':False,
            'policy_origin':policy_origin}
    write_json(output/'verification.json',result);return r,result


def selected_parent_policies(root,request_name):
    """Read the independently committed policy file from the request's parent."""
    path='project/release-policies/'+Path(request_name).name
    try:value=parse_json(git(root,'show','HEAD^:'+path).encode(),canonical_required=True)
    except Exception:raise EvidenceError('missing independently committed release parent policies') from None
    fields(value,'schema production source progress','preselected release parent policies')
    if value['schema']!='ovl.release-parent-policies.v1':raise EvidenceError('unsupported release parent policy selection')
    pp=ProductionPublisherPolicy(**value['production']);sp=PublisherPolicy(**value['source'])
    progress=[ProgressPublisherPolicy(**v) for v in value['progress']]
    for p in [pp,sp,*progress]:
        p.validate()
        try:git(root,'merge-base','--is-ancestor',p.source_revision,'HEAD^')
        except Exception:raise EvidenceError('selected publisher revision is not a prior ancestor') from None
    return pp,sp,progress,{'path':path,'sha256':digest(value),'selection_revision':git(root,'rev-parse','HEAD^')}


def generate(root,environ,output):
    revision=actions_revision(root,environ);name=select_request(root,environ)
    if name is None:raise EvidenceError('no new release request')
    request=object_at(root,name);fields(request,'schema release','release signing request')
    if request['schema']!='ovl.release-signing-request.v1':raise EvidenceError('unsupported release signing request')
    value=validate(request['release'])
    if output.exists():raise EvidenceError('release signing output must be fresh')
    output.mkdir(parents=True,exist_ok=False)
    pp,sp,progress,selection=selected_parent_policies(root,name)
    download=download_archive(value['evidence'],output/'evidence')
    r,checked=verify_parents(value,output/'evidence',output/'parents',root,pp,sp,progress,policy_origin='repository-selected-before-release-request')
    if Path(name).stem!=r['run_id']+'-'+r['attempt_id']+'-'+value['publication_id']:raise EvidenceError('release filename differs from registered attempt')
    own=ReleasePublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,RELEASE_WORKFLOW,ISSUER,REF,revision,digest(value),
                              'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted');own.validate()
    write_json(output/'release.json',value);write_json(output/'ci-self-check-policy.json',asdict(own))
    write_json(output/'ci-input-checks.json',{'evidence_download':download,'parents':checked,
               'parent_policy_selection':selection,'request_uniqueness_scope':'one publication identity in observed Git ancestry; global equivocation prevention NOT_ESTABLISHED',
               'scope':'actual public checkpoint integrity and operator report relationships; raw reconstruction and CUDA replay NOT_RUN'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--select',action='store_true');p.add_argument('--verify-output',action='store_true')
    p.add_argument('--output',type=Path,default=Path('anchor-release'));a=p.parse_args()
    try:
        if a.select:print('present='+('true' if select_request(Path.cwd(),os.environ) else 'false'))
        elif a.verify_output:
            own=ReleasePublisherPolicy(**read_json(a.output/'ci-self-check-policy.json'))
            receipt=verify_anchor(a.output/'release.json',a.output/'release.sigstore.json',own,policy_origin='ci-self-generated')
            write_json(a.output/'ci-signature-check.json',receipt)
        else:generate(Path.cwd(),os.environ,a.output)
    except Exception as error:p.exit(1,'release endorsement refused: '+str(error)+'\n')

if __name__=='__main__':main()
