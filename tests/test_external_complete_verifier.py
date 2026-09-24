"""Actual complete tiny transformations/trajectory/exports; explicit identity/CUDA substitutes."""
from dataclasses import asdict
from pathlib import Path
import shutil
import pytest
from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
from test_production_export import setup as export_setup
import verify_complete as m
from ovl_pipeline import preparation,production_replay,production_export,runtime_launch
from ovl_pipeline.anchoring import PublisherPolicy,REPOSITORY,REPOSITORY_ID,OWNER_ID,WORKFLOW,ISSUER
from ovl_pipeline.production_identity import ProductionPublisherPolicy,PRODUCTION_WORKFLOW
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json


def setup(inputs,prepared,tmp_path,monkeypatch,damage=None):
    # Explicit identity/subprocess adapters; actual isolated child tested separately.
    monkeypatch.setattr(m,'driver_identity',lambda p,**kw:{'explicit_test_double':'synthetic source context'})
    monkeypatch.setattr(m,'check_replay_launch',lambda *a:None)
    monkeypatch.setattr(m,'reconstruct',lambda checkout,statement,bundle,policy,raw,output,expected,execution:
        preparation.prepare_committed(statement,bundle,policy,raw/'wikipedia',raw/'conversation',output,
                                      expected_preparation=expected,resume=False))
    source,wiki,conversation=inputs
    r,envelopes,chain,streams,exports,export_run=export_setup(prepared,tmp_path,monkeypatch);export_run()
    packet=tmp_path/'packet';packet.mkdir()
    for name,value in [('registration.json',r),('source-statement.json',source),('preparation.json',prepared[1]),
                       ('source-statement.sigstore.json',{'explicit-publisher-test-double':True})]:write_json(packet/name,value)
    sp=PublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,WORKFLOW,ISSUER,'refs/heads/feat/verifiable-wikipedia-pipeline',
                       source['source_revision'],digest(source),'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    pp=ProductionPublisherPolicy(**{**asdict(sp),'workflow':PRODUCTION_WORKFLOW,'statement_sha256':digest(r)})
    monkeypatch.setattr(preparation,'verify_anchor',lambda *a,**k:{'result':'PASS','statement_sha256':digest(source),'explicit-test-double':True})
    raw=tmp_path/'raw';raw.mkdir();shutil.copytree(wiki,raw/'wikipedia');shutil.copytree(conversation,raw/'conversation')
    for name in ('README.md','LICENSES.md'):(raw/name).write_bytes(b'raw')
    output=tmp_path/'full';calls=[]
    def launch(lock,wheels,venv,code,out,module,args,**kwargs):
        calls.append(args);assert module=='ovl_pipeline.production_export' and args[0]=='replay-check'
        selected=dict(zip(args[1::2],args[2::2]));actual_streams={p:Path(selected['--'+p+'-stream']) for p in ('wikipedia','conversation')}
        assert actual_streams=={p:output/'reconstructed'/p for p in actual_streams}
        if damage=='missing-replay':return {'exit_code':0,'explicit-launcher-test-double':True}
        result=production_replay.replay(packet,None,pp,sp,tmp_path,chain,tmp_path/'progress',[],actual_streams,Path(selected['--output']))
        if damage=='partial-replay':
            result['updates_recomputed']['wikipedia']=0;write_json(Path(selected['--output'])/'verification.json',result)
        if damage=='child-exit':return {'exit_code':1,'explicit-launcher-test-double':'killed after writing numerical PASS'}
        checked=production_export.verify_replayed_exports(exports,r,result)
        write_json(Path(selected['--output'])/'exports-verification.json',checked)
        return {'exit_code':0,'explicit-launcher-test-double':'CPU execution in test process only'}
    monkeypatch.setattr(runtime_launch,'launch',launch)
    if damage=='raw-byte':(raw/'wikipedia'/source['wikipedia']['spec']['filename']).write_bytes(b'altered')
    if damage in ('adopted-stage','missing-stage'):
        real=preparation.prepare_committed
        def changed(*a,**k):
            value=real(*a,**k)
            if damage=='adopted-stage':value['stages_adopted_from_local_cache']=['corpus']
            else:value['stages_executed_this_run'].pop()
            return value
        monkeypatch.setattr(preparation,'prepare_committed',changed)
    runtime={n:tmp_path/n for n in ('lock','wheels','venv','source','interpreter_archive','interpreter_root')}
    runtime['source']=tmp_path/'src';runtime['source'].mkdir()
    runtime.update(allowed_generated={},interpreter_sha256='1'*64)
    def run():return m.full(packet,packet/'source-statement.sigstore.json',pp,sp,tmp_path,chain,tmp_path/'progress',[],raw,exports,output,runtime)
    return r,output,calls,run,exports


def test_complete_reconstruction_replay_export_and_all_validation_targets(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    r,out,calls,run,exports=setup(inputs,prepared,tmp_path,monkeypatch)
    result=run();assert result['result']=='PASS' and result['locally_recomputed'] is True and result['independent_third_party'] is False
    assert result['reconstruction']['stages_executed_this_run']==m.STAGES and not result['reconstruction']['stages_adopted_from_local_cache']
    assert len(calls)==1 and result['public_release_download_verification']=='NOT_RUN'
    assert result['raw_inputs']['path_supplied_by']=='caller' and result['raw_inputs']['public_anonymous_download_this_command']=='NOT_RUN'
    assert result['raw_inputs']['all_local_bytes_rehashed']=='PASS'
    evaluation=read_json(out/'evaluation.json')
    expected=prepared[1]['streams']['conversation-validation']['targets']
    assert all(v['targets']==expected for v in evaluation['models'].values())
    assert len(evaluation['models']['chat']['demonstrations'])==2
    assert 'OR_TOKEN_LOSS' in evaluation['factual_accuracy']
    with pytest.raises(EvidenceError,match='fresh output'):run()


@pytest.mark.parametrize('damage',['raw-byte','adopted-stage','missing-stage','partial-replay','missing-replay'])
def test_full_profile_cannot_accept_partial_cached_or_missing_work(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    r,out,calls,run,exports=setup(inputs,prepared,tmp_path,monkeypatch,damage)
    with pytest.raises((EvidenceError,OSError)):run()
    assert not(out/'verification.json').exists()
    if damage in ('raw-byte','adopted-stage','missing-stage'):assert not calls


def test_evaluation_refuses_a_sample_even_with_correct_stream_manifest(prepared,tmp_path,monkeypatch):
    from ovl_pipeline import data
    r,envelopes,chain,streams,exports,export_run=export_setup(prepared,tmp_path,monkeypatch);report=export_run()
    actual=data.batches
    def sampled(*args):yield next(actual(*args))
    monkeypatch.setattr(data,'batches',sampled)
    with pytest.raises(EvidenceError,match='coverage incomplete'):
        m.evaluate(exports,r,prepared[0],{p:report['exports'][p]['model_root'] for p in ('base','chat')})


def test_failed_child_exit_cannot_consume_its_saved_pass_report(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    r,out,calls,run,exports=setup(inputs,prepared,tmp_path,monkeypatch,'child-exit')
    def must_not_consume(*a,**kw):raise AssertionError('parent consumed failed child report')
    monkeypatch.setattr(production_export,'verify_replayed_exports',must_not_consume)
    with pytest.raises(EvidenceError,match='process failed'):run()
    assert read_json(out/'numerical-replay/verification.json')['result']=='PASS'
    assert not(out/'verification.json').exists()
