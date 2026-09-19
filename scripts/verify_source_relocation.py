#!/usr/bin/env python3
"""Check separately endorsed source locations without replacing execution parents.

Both policies are selected by the caller independently of the downloaded objects.
The later source assertion is used only for current raw-source locations. It is
not substituted into the original preparation, reconstruction or registration.
This check does not execute transformations or verify complete raw downloads.
The publisher signature endorses the entire supplemental source assertion; its
location-only role is assigned by this checker, not enforced globally by signing.
"""
import argparse
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote

from ovl_pipeline.anchoring import PublisherPolicy, verify_anchor
from ovl_pipeline.canonical import EvidenceError, canonical, digest, read_json, require_digest
from ovl_pipeline.preparation import validate_contract
from assemble_computation_evidence import reconstruction


def relationships(original, relocated, prepared, checkpoint, observation, *,
                  preparation_sha256, reconstruction_checkpoint_sha256):
    """Exact object relationships only; no signature or execution claim."""
    require_digest(preparation_sha256);require_digest(reconstruction_checkpoint_sha256)
    validate_contract(original);validate_contract(relocated)
    if (original['source_revision']==relocated['source_revision']
            or original['attempt_id']==relocated['attempt_id']
            or original['archive']['revision']==relocated['archive']['revision']):
        raise EvidenceError('relocation requires distinct assertion, signing and archive identities')
    normalized=deepcopy(relocated)
    normalized['source_revision']=original['source_revision']
    normalized['attempt_id']=original['attempt_id']
    normalized['archive']['revision']=original['archive']['revision']
    if normalized!=original:
        raise EvidenceError('relocation changed source inventory, recipe, code, environment or other committed content')
    if (digest(prepared)!=preparation_sha256
            or prepared.get('schema')!='ovl.complete-preparation.v1'
            or prepared.get('source_commitment_sha256')!=digest(original)
            or prepared.get('code')!=original['code']
            or prepared.get('environment')!=original['environment']
            or prepared.get('validation_used_for_training') is not False):
        raise EvidenceError('original prepared parent identity changed')
    if digest(checkpoint)!=reconstruction_checkpoint_sha256:
        raise EvidenceError('reconstruction checkpoint differs from independently selected digest')
    report=reconstruction(checkpoint,observation,{
        'source_statement_sha256':digest(original),'preparation_sha256':preparation_sha256})
    archive=relocated['archive']
    return {
        'original_source_statement_sha256':digest(original),
        'locator_source_statement_sha256':digest(relocated),
        'original_preparation_sha256':preparation_sha256,
        'original_reconstruction_checkpoint_sha256':reconstruction_checkpoint_sha256,
        'original_reconstruction_report_sha256':digest(report),
        'original_execution_observation_sha256':digest(observation),
        'raw_inventory_sha256':digest(original['archive']['inventory']),
        'selected_endorsed_raw_locations':[
            {**entry,'url':f"https://huggingface.co/datasets/{archive['repo']}/resolve/{archive['revision']}/"+quote(archive['prefix']+'/'+entry['path'],safe='/')}
            for entry in archive['inventory']],
        'execution_source_parent':'original_source_statement_sha256',
        'supplemental_statement_use_in_this_check':'raw-location-endorsement-only',
        'supplemental_signature_scope':'entire-source-assertion; this checker does not restrict other consumers',
        'location_availability':'NOT_RUN',
    }


def verify(*,original_statement,original_bundle,original_policy,relocated_statement,
           relocated_bundle,relocated_policy,preparation,preparation_sha256,
           reconstruction_checkpoint,reconstruction_checkpoint_sha256,execution_observation):
    for policy in (original_policy,relocated_policy):
        if type(policy) is not PublisherPolicy:
            raise EvidenceError('independently selected exact source publisher policies required')
        policy.validate()
    original_check=verify_anchor(original_statement,original_bundle,original_policy)
    relocated_check=verify_anchor(relocated_statement,relocated_bundle,relocated_policy)
    original=read_json(original_statement);relocated=read_json(relocated_statement)
    for statement,policy,check in ((original,original_policy,original_check),
                                   (relocated,relocated_policy,relocated_check)):
        if (digest(statement)!=policy.statement_sha256
                or digest(statement)!=check['statement_sha256']
                or statement['source_revision']!=policy.source_revision):
            raise EvidenceError('source assertion changed or differs from selected certificate revision')
    relation=relationships(original,relocated,read_json(preparation),read_json(reconstruction_checkpoint),
                           read_json(execution_observation),preparation_sha256=preparation_sha256,
                           reconstruction_checkpoint_sha256=reconstruction_checkpoint_sha256)
    return {
        'schema':'ovl.source-location-continuity-check.v1','result':'PASS',
        'scope':'two-publisher-endorsements-and-exact-source-location-relationships-only',
        'original_endorsement':original_check,'locator_endorsement':relocated_check,
        'relationships':relation,'old_signed_objects_modified':False,
        'policy_origin':'caller-supplied-for-each-endorsement',
        'independent_policy_selection_cryptographically_established':False,
        'assertion_truth_established':False,'complete_raw_download':'NOT_RUN',
        'prepared_payload_download':'NOT_RUN','data_reconstruction':'NOT_RUN',
        'training_replay':'NOT_RUN','production_admission':'NOT_RUN',
        'independent_third_party':False,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original-statement','original-bundle','original-policy','relocated-statement',
                 'relocated-bundle','relocated-policy','preparation','reconstruction-checkpoint','execution-observation'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preparation-sha256',required=True,
                        help='Independently selected digest of the original complete preparation manifest')
    parser.add_argument('--reconstruction-checkpoint-sha256',required=True,
                        help='Independently selected digest of the original checkpoint object, not its nested report')
    args=vars(parser.parse_args())
    try:
        for key in ('original_policy','relocated_policy'):
            args[key]=PublisherPolicy(**read_json(args[key]))
        result=verify(**args)
    except Exception as error:
        print(canonical({'result':'FAIL','reason':str(error)}).decode());return 1
    print(canonical(result).decode());return 0


if __name__=='__main__':
    raise SystemExit(main())
