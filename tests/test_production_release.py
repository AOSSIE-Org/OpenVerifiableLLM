"""Final inventory integrity and scope; explicit publisher/CUDA substitutes."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import pytest
from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
from test_production_verify import setup as full_setup
from test_progress_commitment import request as progress_request
from ovl_pipeline import production_release as m
from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy,PROGRESS_WORKFLOW
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json


def setup(inputs,prepared,tmp_path,monkeypatch):
    r,full,calls,run,exports=full_setup(inputs,prepared,tmp_path,monkeypatch);result=run()
    report=tmp_path/'reports';report.mkdir()
    for name,path in [('verification',full/'verification.json'),('reconstruction',full/'reconstruction.json'),
                      ('replay',full/'numerical-replay/verification.json'),('evaluation',full/'evaluation.json'),('exports',exports/'export.json')]:
        write_json(report/(name+'.json'),read_json(path))
    fixture=tmp_path/'request-fixture';fixture.mkdir();request=progress_request(fixture);root=digest(r)
    request['registration_request']['registration_sha256']=root
    request['registration_request']['packet']['prefix']='production-registration/'+root
    for e in request['registration_request']['packet']['inventory']:
        if e['path']=='registration.json':e['sha256']=root
    request['registration_policy']['statement_sha256']=root
    request['registration_anchor']['prefix']='production-anchors/'+root
    # full_setup's actual numerical boundaries are exposed through its test
    # authentication adapter. These are real Ed25519/tensor states, fake parents.
    from ovl_pipeline.production_replay import authenticate
    _,envelopes,_=authenticate();request['envelopes']=envelopes
    def archive(prefix,names):return {'repo':'AOSSIE/openverifiable-synthetic-evidence','revision':'1'*40,'prefix':prefix,
                                      'inventory':[{'path':n,'bytes':2,'sha256':'a'*64} for n in names]}
    def progress(i):
        p={**request['registration_policy'],'workflow':PROGRESS_WORKFLOW,'statement_sha256':'a'*64}
        return {'archive':archive(f'production-progress/{root}/progress-{i:05d}',['statement.json','statement.sigstore.json']),'policy':p}
    request['previous_progress']=[progress(i) for i in range(len(envelopes)-1)]
    last=envelopes[-1]['body'];cp=tmp_path/'checkpoints'/last['checkpoint_path']
    request['checkpoint_archive']={'repo':'AOSSIE/openverifiable-synthetic-evidence','revision':'1'*40,
        'prefix':f'production-checkpoints/{root}/{last["checkpoint_path"]}',
        'inventory':inventory(cp,['checkpoint.json','state.json','state.safetensors'])}
    write_json(report/'context.json',{'schema':'ovl.release-context.v1','closing_request':request,'final_progress':progress(len(envelopes)-1)})
    files=inventory(report,m.EVIDENCE)
    archive={'repo':'AOSSIE/openverifiable-synthetic-evidence','revision':'2'*40,'prefix':'release-evidence/'+digest(files),'inventory':files}
    payload=m.prepare_payloads(r,report,exports,Path(__file__).resolve().parents[1],tmp_path/'payload',source_statement=inputs[0],evidence_archive=archive)
    return r,report,archive,payload


def test_final_inventory_binds_both_safe_models_and_honest_operator_scope(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    r,report,archive,payload=setup(inputs,prepared,tmp_path,monkeypatch);value=m.build(r,report,archive,payload)
    assert set(value['models'])=={'base','chat'}
    assert value['claims']['consumer_training_recomputed'] is False and value['claims']['independent_third_party'] is False
    assert value['models']['base']['files']==inventory(payload['base'],m.PAYLOAD)
    assert 'not independent third-party verification' in (payload['chat']/'README.md').read_text()


@pytest.mark.parametrize('damage',['signature-is-replay','third-party','bad-model-root','tag','foreign-repo','unknown-schema','missing-payload','wrong-report-parent'])
def test_release_manifest_refuses_wrong_identity_or_overstated_claims(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    r,report,archive,payload=setup(inputs,prepared,tmp_path,monkeypatch);value=m.build(r,report,archive,payload)
    if damage=='signature-is-replay':value['claims']['consumer_training_recomputed']=True
    elif damage=='third-party':value['claims']['independent_third_party']=True
    elif damage=='bad-model-root':value['models']['chat']['model_root']='bad'
    elif damage=='tag':value['evidence']['revision']='main'
    elif damage=='foreign-repo':value['models']['base']['repo']='unapproved/model'
    elif damage=='unknown-schema':value['schema']='future'
    elif damage=='missing-payload':value['models']['base']['files'].pop()
    else:value['report_roots']['replay']='0'*64
    with pytest.raises(EvidenceError):m.validate(value)


@pytest.mark.parametrize('damage',['subset','cache','restore-prover','replay-gap','scope','foreign-root','evaluation-count','extra-private-file'])
def test_rehashed_release_evidence_cannot_hide_missing_work_or_payload(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    r,report,archive,payload=setup(inputs,prepared,tmp_path,monkeypatch)
    if damage=='extra-private-file':(payload['base']/'seed.key').write_bytes(b'never publish even synthetic extra')
    else:
        rec=read_json(report/'reconstruction.json');replay=read_json(report/'replay.json');v=read_json(report/'verification.json');ev=read_json(report/'evaluation.json')
        if damage=='subset':replay['updates_recomputed']['wikipedia']-=1
        elif damage=='cache':rec['stages_adopted_from_local_cache']=['corpus']
        elif damage=='restore-prover':replay['prover_checkpoints_restored']=True
        elif damage=='replay-gap':replay['comparisons'].pop()
        elif damage=='scope':replay['scope']='sampled-segment'
        elif damage=='foreign-root':replay['registration_sha256']='0'*64
        else:ev['models']['chat']['targets']-=1
        v['reconstruction']=rec;v['replay_report_sha256']=digest(replay);v['evaluation_sha256']=digest(ev)
        for name,value in [('reconstruction',rec),('replay',replay),('verification',v),('evaluation',ev)]:write_json(report/(name+'.json'),value)
        archive['inventory']=inventory(report,m.EVIDENCE);archive['prefix']='release-evidence/'+digest(archive['inventory'])
    with pytest.raises(EvidenceError):m.build(r,report,archive,payload)


def test_separate_publication_identity_preserves_training_and_selects_new_destinations(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    r,report,archive,payload=setup(inputs,prepared,tmp_path,monkeypatch);first=m.build(r,report,archive,payload)
    other=m.prepare_payloads(r,report,tmp_path/'export',Path(__file__).resolve().parents[1],tmp_path/'new-payload',
                             source_statement=inputs[0],evidence_archive=archive,publication='release-2')
    second=m.build(r,report,archive,other,publication='release-2')
    assert first['registration_sha256']==second['registration_sha256']==digest(r)
    assert first['report_roots']==second['report_roots'] and first['evidence']==second['evidence']
    for phase in ('base','chat'):
        assert first['models'][phase]['model_root']==second['models'][phase]['model_root']
        assert first['models'][phase]['repo']!=second['models'][phase]['repo']
        assert second['models'][phase]['repo'].endswith('-release-2-'+phase)
    # Changing only an inventory's label cannot silently rename existing payloads.
    with pytest.raises(EvidenceError):m.build(r,report,archive,other,publication='unknown')


def test_model_cards_pin_sources_evidence_and_demonstration_scope(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    r,report,archive,payload=setup(inputs,prepared,tmp_path,monkeypatch)
    for phase in ('base','chat'):
        card=(payload[phase]/'README.md').read_text();sources=(payload[phase]/'MODEL_SOURCES.md').read_text()
        assert archive['revision']+'/'+archive['prefix'] in card
        assert inputs[0]['conversation']['revision'] in sources
        assert str(inputs[0]['archive']['retention_days_target'])+'-day' in card
        assert '--max-new-tokens 32' in card and 'internal model consistency only' in card
        assert ('- oasst1\n' in card)==(phase=='chat')


@pytest.mark.parametrize('damage',['attested-computation','early-public-verification','verified-export','truthy-list'])
def test_rehashed_operator_reports_cannot_overstate_scope(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    r,report,archive,payload=setup(inputs,prepared,tmp_path,monkeypatch)
    path=report/('exports.json' if damage=='verified-export' else 'verification.json');value=read_json(path)
    if damage=='attested-computation':value['attested_by']='publisher-is-computation'
    elif damage=='early-public-verification':value['public_release_download_verification']='PASS'
    elif damage=='verified-export':value['status']='TRAINING_VERIFIED_BY_EXPORT'
    else:value['locally_recomputed']=['signature']
    write_json(path,value);archive['inventory']=inventory(report,m.EVIDENCE);archive['prefix']='release-evidence/'+digest(archive['inventory'])
    with pytest.raises(EvidenceError):m.build(r,report,archive,payload)
