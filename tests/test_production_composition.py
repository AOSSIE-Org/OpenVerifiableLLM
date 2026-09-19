"""Separate invocation report links; CPU numerical work and explicit identity doubles."""
from copy import deepcopy
from pathlib import Path
import pytest
from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
from test_production_release import setup
from ovl_pipeline import production_release as release
from ovl_pipeline.canonical import EvidenceError, digest, inventory, read_json, write_json


def composed(report):
    value = read_json(report / 'verification.json')
    replay = read_json(report / 'replay.json')
    reconstruction = value['reconstruction']
    # Explicit process observation substitute: setup executes a real complete
    # tiny CPU trajectory but deliberately does not launch a CUDA child.
    process = {'schema':'ovl.audited-runtime-process.v1','launch_sha256':'a'*64,'exit_code':0,
               'scope':'external package audit and constrained target-process launch; not model verification by itself'}
    value.update(schema='ovl.complete-computation-verification.v2', locally_recomputed=False,
                 performed_by='project-operator-separate-recorded-executions', replay_process=process)
    value['execution'] = {
        'schema':'ovl.separate-computation-executions.v1',
        'mode':'earlier-complete-clean-reconstruction-and-later-continuous-full-replay',
        'reconstruction':{'report_sha256':digest(reconstruction),
                          'execution_observation_sha256':reconstruction['execution_observation_sha256'],
                          'public_evidence':{'url':'https://raw.githubusercontent.com/AOSSIE-Org/OpenVerifiableLLM/'+'1'*40+'/project/evidence/synthetic/reconstruction.json','sha256':'b'*64}},
        'replay':{'report_sha256':digest(replay),'audited_process_sha256':digest(process),
                  'launch_sha256':process['launch_sha256'],'terminal_sha256':'c'*64,
                  'retention_sha256':'d'*64,'public_evidence':{'url':'https://huggingface.co/datasets/AOSSIE/openverifiable-synthetic-evidence/resolve/'+'2'*40+'/replay.json','sha256':'e'*64}},
        'assembly':{'raw_inventory_sha256':value['raw_inputs']['complete_inventory_sha256'],
                    'prepared_manifest_sha256':reconstruction['preparation_sha256'],
                    'record_inventory_sha256':'f'*64,'replay_inventory_sha256':'1'*64,
                    'safe_states_compared':len(replay['comparisons'])+len(replay['recovery_checkpoints']),
                    'raw_transformations_executed':False,'numerical_updates_executed':0}}
    return value


def test_separate_invocations_preserve_all_existing_release_requirements(cpu_runtime, inputs, prepared, tmp_path, monkeypatch):
    registration, report, archive, payloads = setup(inputs, prepared, tmp_path, monkeypatch)
    original = {name:read_json(report / (name+'.json')) for name in ('verification','reconstruction','replay','evaluation')}
    good = composed(report)
    write_json(report / 'verification.json', good)
    assert release.reports(registration, report)['verification']['locally_recomputed'] is False
    archive['inventory']=inventory(report,release.EVIDENCE);archive['prefix']='release-evidence/'+digest(archive['inventory'])
    new_payload=release.prepare_payloads(registration,report,tmp_path/'export',Path(__file__).parents[1],
        tmp_path/'composed-payload',source_statement=inputs[0],evidence_archive=archive)
    assert release.build(registration,report,archive,new_payload)['claims']['independent_third_party'] is False
    for phase in ('base','chat'):
        assert 'separate recorded executions' in (new_payload[phase]/'README.md').read_text()
    for damage in ('claim-new-execution','restore-prover','sample-replay','cache-reconstruction',
                   'missing-stage','wrong-observation','wrong-launch','failed-process','no-public-evidence',
                   'missing-state','wrong-preparation','new-transformations','new-updates','changed-raw'):
        values = deepcopy(original); value = deepcopy(good); replay = values['replay']; rec = values['reconstruction']
        execution = value['execution']
        if damage == 'claim-new-execution': value['locally_recomputed'] = True
        elif damage == 'restore-prover': replay['prover_checkpoints_restored'] = True
        elif damage == 'sample-replay': replay['updates_recomputed']['wikipedia'] -= 1
        elif damage == 'cache-reconstruction': rec['stages_adopted_from_local_cache'] = ['corpus']
        elif damage == 'missing-stage': rec['stages_executed_this_run'].pop()
        elif damage == 'wrong-observation': execution['reconstruction']['execution_observation_sha256'] = '0'*64
        elif damage == 'wrong-launch': execution['replay']['launch_sha256'] = '0'*64
        elif damage == 'failed-process':
            value['replay_process']['exit_code'] = 1
            execution['replay']['audited_process_sha256'] = digest(value['replay_process'])
        elif damage == 'no-public-evidence': del execution['replay']['public_evidence']
        elif damage == 'missing-state': execution['assembly']['safe_states_compared'] -= 1
        elif damage == 'wrong-preparation': execution['assembly']['prepared_manifest_sha256'] = '0'*64
        elif damage == 'new-transformations': execution['assembly']['raw_transformations_executed'] = True
        elif damage == 'new-updates': execution['assembly']['numerical_updates_executed'] = 1
        else: execution['assembly']['raw_inventory_sha256'] = '0'*64
        # Rebind ordinary report hashes: adversaries must still fail the actual
        # coverage, process and honest invocation-scope conditions.
        value['reconstruction'] = rec; value['replay_report_sha256'] = digest(replay)
        execution['reconstruction']['report_sha256'] = digest(rec)
        execution['replay']['report_sha256'] = digest(replay)
        for name, data in [('verification',value),('replay',replay),('reconstruction',rec)]:
            write_json(report / (name+'.json'), data)
        with pytest.raises(EvidenceError): release.reports(registration, report)


@pytest.mark.parametrize('url',[
    'https://raw.githubusercontent.com/AOSSIE-Org/OpenVerifiableLLM/main/project/evidence/report.json',
    'https://raw.githubusercontent.com/attacker/OpenVerifiableLLM/'+'1'*40+'/project/evidence/report.json',
    'https://huggingface.co/datasets/AOSSIE/openverifiable-test-evidence/resolve/'+'1'*40+'/../report.json',
    'file:///private/report.json',
])
def test_public_execution_reference_requires_immutable_approved_destination(url):
    from ovl_pipeline.production_composition import public_reference
    with pytest.raises(EvidenceError):public_reference({'url':url,'sha256':'a'*64})
