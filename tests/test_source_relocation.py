"""Real tiny preparation; explicit synthetic endorsement doubles, no gate credit."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import verify_source_relocation as relocation
from test_preparation import inputs
from test_prepared_verification import prepared
from ovl_pipeline.anchoring import PublisherPolicy,REPOSITORY,WORKFLOW,ISSUER,REPOSITORY_ID,OWNER_ID
from ovl_pipeline.canonical import EvidenceError,digest,write_json


@pytest.fixture
def objects(inputs,prepared):
    original=inputs[0];new=deepcopy(original)
    new['attempt_id']='relocated';new['source_revision']='2'*40;new['archive']['revision']='3'*40
    prep=prepared[1]
    stages=['corpus','tokenizer','wikipedia','conversation-selection','conversation','conversation-validation']
    observation={'schema':'ovl.preparation-execution.v1','status':'PASS',
        'source_commitment_sha256':digest(original),'resume_requested':False,
        'stages_executed_this_run':stages,'stages_adopted_from_local_cache':[]}
    report={'result':'PASS','scope':'complete-source-preparation','preparation_sha256':digest(prep),
        'source_commitment_sha256':digest(original),'full_reconstruction_compared':True,
        'stages_executed_this_run':stages,'stages_adopted_from_local_cache':[],
        'execution_observation_sha256':digest(observation)}
    cp={'schema':'ovl.complete-clean-reconstruction-checkpoint.v2','result':'PASS',
        'independent_third_party':False,'whole_job_elapsed_ms':1,'report':report}
    return original,new,prep,cp,observation


def check(objects,**pins):
    _,_,prepared,checkpoint,_=objects
    return relocation.relationships(*objects,**{
        'preparation_sha256':digest(prepared),'reconstruction_checkpoint_sha256':digest(checkpoint),**pins})


def test_locations_change_while_execution_parents_remain_original(objects):
    before=deepcopy(objects);result=check(objects)
    assert objects==before
    assert result['original_source_statement_sha256']==digest(objects[0])
    assert result['locator_source_statement_sha256']==digest(objects[1])
    assert result['execution_source_parent']=='original_source_statement_sha256'
    assert all('/resolve/'+'3'*40+'/' in row['url'] for row in result['selected_endorsed_raw_locations'])
    assert [{k:v for k,v in row.items() if k!='url'} for row in result['selected_endorsed_raw_locations']]==objects[0]['archive']['inventory']


@pytest.mark.parametrize('field',['attempt_id','source_revision','archive.revision'])
def test_each_required_new_identity_must_differ(objects,field):
    original,new=objects[:2]
    if field=='archive.revision':new['archive']['revision']=original['archive']['revision']
    else:new[field]=original[field]
    with pytest.raises(EvidenceError,match='distinct assertion'):check(objects)


@pytest.mark.parametrize('mutation',['recipe','code','environment','inventory','conversation','prefix','run-id'])
def test_extra_source_changes_are_not_relocation(objects,mutation):
    new=objects[1]
    if mutation=='recipe':new['recipe']['tokenizer_sample_bytes']+=1
    elif mutation=='code':new['code'][0]['sha256']='f'*64
    elif mutation=='environment':new['environment']['python']='0.0.0'
    elif mutation=='inventory':new['archive']['inventory'][0]['sha256']='f'*64
    elif mutation=='conversation':new['conversation']['revision']='f'*40
    elif mutation=='prefix':new['archive']['prefix']='raw/other'
    else:new['run_id']='other'
    with pytest.raises(EvidenceError):check(objects)


@pytest.mark.parametrize('mutation',['prepared-parent','report-parent','observation-parent','cached-stage','missing-stage','checkpoint-pin','preparation-pin'])
def test_reparenting_or_incomplete_old_execution_rejected(objects,mutation):
    original,new,prepared,cp,obs=objects
    pins={}
    if mutation=='prepared-parent':prepared['source_commitment_sha256']=digest(new)
    elif mutation=='report-parent':cp['report']['source_commitment_sha256']=digest(new)
    elif mutation=='observation-parent':obs['source_commitment_sha256']=digest(new);cp['report']['execution_observation_sha256']=digest(obs)
    elif mutation=='cached-stage':cp['report']['stages_adopted_from_local_cache']=['corpus']
    elif mutation=='missing-stage':cp['report']['stages_executed_this_run']=cp['report']['stages_executed_this_run'][:-1]
    elif mutation=='checkpoint-pin':pins['reconstruction_checkpoint_sha256']='f'*64
    else:pins['preparation_sha256']='f'*64
    with pytest.raises(EvidenceError):check(objects,**pins)


def arguments(objects,tmp_path):
    original,new,prepared,cp,obs=objects
    args={}
    for key,value in [('original_statement',original),('relocated_statement',new),('preparation',prepared),
                      ('reconstruction_checkpoint',cp),('execution_observation',obs)]:
        args[key]=tmp_path/(key+'.json');write_json(args[key],value)
    for key,value in [('original',original),('relocated',new)]:
        args[key+'_bundle']=tmp_path/(key+'.bundle')
        args[key+'_policy']=PublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,WORKFLOW,ISSUER,
            'refs/heads/feat/verifiable-wikipedia-pipeline',value['source_revision'],digest(value),
            'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    return {**args,'preparation_sha256':digest(prepared),'reconstruction_checkpoint_sha256':digest(cp)}


def test_two_endorsements_required_and_no_execution_credit(objects,tmp_path,monkeypatch):
    args=arguments(objects,tmp_path);calls=[]
    def synthetic_endorsement(statement,bundle,policy):
        calls.append(policy)
        return {'result':'PASS','statement_sha256':policy.statement_sha256,'scope':'explicit synthetic signature double'}
    monkeypatch.setattr(relocation,'verify_anchor',synthetic_endorsement)
    report=relocation.verify(**args)
    assert calls==[args['original_policy'],args['relocated_policy']]
    for key in ['complete_raw_download','prepared_payload_download','data_reconstruction','training_replay','production_admission']:
        assert report[key]=='NOT_RUN'
    assert report['assertion_truth_established'] is False


def test_missing_second_endorsement_fails(objects,tmp_path,monkeypatch):
    args=arguments(objects,tmp_path)
    def synthetic_endorsement(statement,bundle,policy):
        if policy==args['relocated_policy']:raise EvidenceError('missing second endorsement')
        return {'result':'PASS','statement_sha256':policy.statement_sha256}
    monkeypatch.setattr(relocation,'verify_anchor',synthetic_endorsement)
    with pytest.raises(EvidenceError,match='missing second'):relocation.verify(**args)


def test_wrong_external_identity_rejected_before_signature_call(objects,tmp_path,monkeypatch):
    args=arguments(objects,tmp_path);args['relocated_policy']=replace(args['relocated_policy'],repository='unapproved/repository')
    monkeypatch.setattr(relocation,'verify_anchor',lambda *a:pytest.fail('invalid policy reached verifier'))
    with pytest.raises(EvidenceError,match='unsupported external'):relocation.verify(**args)


def test_valid_policy_shape_cannot_change_statement_signing_revision(objects,tmp_path,monkeypatch):
    args=arguments(objects,tmp_path)
    args['relocated_policy']=replace(args['relocated_policy'],source_revision='9'*40)
    monkeypatch.setattr(relocation,'verify_anchor',lambda statement,bundle,policy:
        {'result':'PASS','statement_sha256':policy.statement_sha256,'scope':'explicit synthetic signature double'})
    with pytest.raises(EvidenceError,match='selected certificate revision'):relocation.verify(**args)


def test_statement_change_after_endorsement_rejected(objects,tmp_path,monkeypatch):
    args=arguments(objects,tmp_path)
    def synthetic_endorsement(statement,bundle,policy):
        if policy==args['relocated_policy']:
            changed=deepcopy(objects[0]);changed['attempt_id']='changed';write_json(args['original_statement'],changed)
        return {'result':'PASS','statement_sha256':policy.statement_sha256}
    monkeypatch.setattr(relocation,'verify_anchor',synthetic_endorsement)
    with pytest.raises(EvidenceError,match='source assertion changed'):relocation.verify(**args)
