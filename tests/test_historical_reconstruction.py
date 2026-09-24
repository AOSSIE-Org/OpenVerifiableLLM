"""Real fresh child transformations with explicit synthetic signature/source adapters.

No GPU/real publisher credit. Original historical source admission is also tested
against retained signed evidence outside these network-free synthetic tests.
"""
from dataclasses import asdict,replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

import historical_reconstruction as m
from test_preparation import inputs
from ovl_pipeline import anchoring,preparation
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,inventory,read_json,write_json
from ovl_pipeline.anchoring import PublisherPolicy,REPOSITORY,REPOSITORY_ID,OWNER_ID,WORKFLOW,ISSUER


@pytest.fixture
def historical(inputs,tmp_path,monkeypatch):
    source,wiki,conversation=inputs
    current=Path(preparation.__file__).resolve().parents[2]
    kernel=tmp_path/'synthetic-kernel'
    for name in m.PREPARATION_PATHS:
        dest=kernel/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(current/name,dest)
    # This unsigned fixture is a closed synthetic source inventory. No production
    # allowlist or signature implementation is changed outside this test process.
    with (kernel/'src/ovl_pipeline/anchoring.py').open('a') as f:
        f.write('\n# Synthetic signature adapter for isolated fixture only.\n'
                'def verify_anchor(statement,bundle,policy):\n'
                '    from .canonical import digest,read_json\n'
                '    assert read_json(bundle,canonical_required=False)=={"synthetic":True}\n'
                '    assert digest(read_json(statement))==policy.statement_sha256\n'
                '    return {"result":"PASS","statement_sha256":policy.statement_sha256}\n')
    source['code']=inventory(kernel,m.PREPARATION_PATHS)
    monkeypatch.setattr(preparation,'preparation_code',lambda:source['code'])
    expected=preparation.build_prepared(source,wiki,conversation,tmp_path/'reference')
    root=digest(source)
    policy=PublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,WORKFLOW,ISSUER,
        'refs/heads/feat/verifiable-wikipedia-pipeline','0'*40,root,'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    monkeypatch.setattr(anchoring,'verify_anchor',lambda *a:{'result':'PASS','statement_sha256':root})
    monkeypatch.setattr(m,'PREPARATION_SNAPSHOTS',{root:'1'*40})
    def materialize(checkout,revision,entries,target):
        assert revision=='1'*40 and entries==source['code']
        shutil.copytree(kernel,target)
    monkeypatch.setattr(m,'_materialize',materialize)
    raw=tmp_path/'raw';raw.mkdir();shutil.copytree(wiki,raw/'wikipedia');shutil.copytree(conversation,raw/'conversation')
    statement=tmp_path/'statement.json';bundle=tmp_path/'bundle.json'
    write_json(statement,source);bundle.write_text('{\n  \"synthetic\": true\n}\n')
    args=(current,statement,bundle,policy,raw,tmp_path/'rebuilt',expected,tmp_path/'execution')
    return args,kernel


def test_real_child_rebuilds_all_stages_and_ignores_hostile_pythonpath(historical,monkeypatch,tmp_path):
    args,kernel=historical
    hostile=tmp_path/'hostile';hostile.mkdir();(hostile/'ovl_pipeline.py').write_text('raise RuntimeError("shadow")')
    monkeypatch.setenv('PYTHONPATH',str(hostile));monkeypatch.setenv('SYNTHETIC_SECRET','not-for-child')
    report=m.reconstruct(*args)
    assert report['result']=='PASS' and report['full_reconstruction_compared'] is True
    assert report['stages_executed_this_run']==['corpus','tokenizer','wikipedia','conversation-selection','conversation','conversation-validation']
    assert report['stages_adopted_from_local_cache']==[]
    assert read_json(args[5]/'preparation.json')==args[6]
    process=read_json(args[-1]/'process.json');result=read_json(args[-1]/'result.json')
    assert process['exit_code']==0 and process['pid']==result['pid']!=os.getpid()
    with pytest.raises(EvidenceError,match='fresh'):m.reconstruct(*args)


@pytest.mark.parametrize('damage',['signature','unknown','environment','raw','expected','preexisting'])
def test_reconstruction_rejects_incorrect_inputs(historical,monkeypatch,damage):
    args,kernel=historical
    if damage=='signature':
        def reject(*a):raise EvidenceError('synthetic invalid signature')
        monkeypatch.setattr(anchoring,'verify_anchor',reject)
    elif damage=='unknown':monkeypatch.setattr(m,'PREPARATION_SNAPSHOTS',{})
    elif damage=='environment':monkeypatch.setenv('TOKENIZERS_PARALLELISM','true')
    elif damage=='raw':next((args[4]/'wikipedia').glob('*.bz2')).write_bytes(b'changed')
    elif damage=='expected':args[6]['corpus']['record_count']+=1
    else:args[5].mkdir()
    with pytest.raises(EvidenceError):m.reconstruct(*args)
    assert not (args[-1]/'result.json').exists()


@pytest.mark.parametrize('damage',['nonzero','wrong-nonce','wrong-pid','changed-kernel'])
def test_saved_pass_or_disconnected_child_cannot_pass(historical,monkeypatch,damage):
    args,kernel=historical
    real=m.subprocess.Popen
    class Child:
        def __init__(self,*a,**kw):self.child=real(*a,**kw);self.pid=self.child.pid
        def wait(self):
            code=self.child.wait();assert code==0
            path=args[-1]/'result.json';v=read_json(path)
            if damage=='nonzero':return 1
            if damage=='wrong-nonce':v['nonce']='0'*64
            elif damage=='wrong-pid':v['pid']+=1
            else:(args[-1]/'kernel/src/ovl_pipeline/data.py').write_text('# changed')
            write_json(path,v);return 0
    monkeypatch.setattr(m.subprocess,'Popen',Child)
    with pytest.raises(EvidenceError):m.reconstruct(*args)


def test_driver_identity_requires_same_frozen_import_root(tmp_path):
    current=Path(preparation.__file__).resolve().parents[2]
    identity=m.driver_identity(current)
    assert [v['path'] for v in identity['files']]==sorted(m.DRIVER_FILES)
    with pytest.raises(EvidenceError,match='selected frozen'):m.driver_identity(tmp_path)


@pytest.mark.parametrize('damage',[None,'exit','module','arguments','source','session','progress','runtime'])
def test_launch_session_binding_rejects_saved_or_partial_results(tmp_path,damage):
    out=tmp_path/'output';gpu=out/'gpu-launch';numerical=out/'numerical-replay'
    gpu.mkdir(parents=True);numerical.mkdir()
    lock=tmp_path/'lock';lock.write_text('synthetic lock')
    runtime={'source':tmp_path/'src','lock':lock,'interpreter_sha256':'1'*64}
    registration={'code_root':'2'*64};envelopes=[{'synthetic':True}];arguments=['--output',str(numerical)]
    launch={'module':'ovl_pipeline.production_export','arguments':['replay-check',*arguments],
            'source':str(runtime['source'].resolve()),'dependency_lock_sha256':file_hash(lock),
            'interpreter_origin':{'archive_sha256':'1'*64},'wheel_manifest_sha256':'3'*64}
    process={'exit_code':0,'launch_sha256':digest(launch)}
    session={'registration_sha256':digest(registration),'code_root':registration['code_root'],
             'chain_sha256':digest(envelopes),'prover_state_restored':False,'resume_supported':False,
             'environment':{'compatible':{'installed_wheel_audit':{'dependency_lock_sha256':file_hash(lock),
                'wheel_payloads_sha256':'3'*64,'interpreter_origin':{'archive_sha256':'1'*64}}}}}
    replay={'session_sha256':digest(session),'chain_sha256':digest(envelopes),'comparisons':[{'synthetic':True}]}
    progress={'session_sha256':digest(session),'complete':True,'comparisons':replay['comparisons']}
    if damage=='exit':process['exit_code']=1
    elif damage=='module':launch['module']='ovl_pipeline.production_replay'
    elif damage=='arguments':launch['arguments']=['export']
    elif damage=='source':launch['source']=str(tmp_path/'other')
    elif damage=='session':session['resume_supported']=True
    elif damage=='progress':progress['complete']=False
    elif damage=='runtime':session['environment']['compatible']['installed_wheel_audit']['wheel_payloads_sha256']='4'*64
    if damage in ('module','arguments','source'):process['launch_sha256']=digest(launch)
    if damage=='runtime':replay['session_sha256']=progress['session_sha256']=digest(session)
    for name,value in [('launch.json',launch),('process.json',process)]:write_json(gpu/name,value)
    for name,value in [('session.json',session),('progress.json',progress)]:write_json(numerical/name,value)
    if damage:
        with pytest.raises(EvidenceError):m.check_replay_launch(out,process,registration,envelopes,runtime,arguments,replay)
    else:m.check_replay_launch(out,process,registration,envelopes,runtime,arguments,replay)


def test_alternate_child_source_rejected_before_any_execution(tmp_path):
    selected=tmp_path/'selected';(selected/'src').mkdir(parents=True)
    alternate=tmp_path/'alternate';(alternate/'ovl_pipeline').mkdir(parents=True)
    source=Path(preparation.__file__).resolve().parent
    shutil.copyfile(source/'runtime_bootstrap.py',alternate/'ovl_pipeline/runtime_bootstrap.py')
    marker=tmp_path/'unauthorized-execution'
    (alternate/'ovl_pipeline/production_export.py').write_text(
        'from pathlib import Path\nPath('+repr(str(marker))+').write_text("executed zero updates")\n')
    with pytest.raises(EvidenceError,match='child source'):
        m.admit_runtime_source(selected,{'source':alternate})
    assert not marker.exists()
    m.admit_runtime_source(selected,{'source':selected/'src'})


@pytest.mark.parametrize('name',['verify_complete','verify_release_complete'])
def test_mixed_driver_copies_cannot_misattribute_bytes(tmp_path,monkeypatch,name):
    from types import SimpleNamespace
    current=Path(preparation.__file__).resolve().parents[2]
    copied=tmp_path/(name+'.py');copied.write_text('# different executable driver\n')
    with pytest.raises(EvidenceError,match='executing driver'):
        m.driver_identity(current,entrypoint=copied)
    monkeypatch.setitem(sys.modules,name,SimpleNamespace(__file__=str(copied)))
    with pytest.raises(EvidenceError,match='mixed verification driver'):
        m.driver_identity(current,entrypoint=Path(m.__file__).with_name(name+'.py'))
