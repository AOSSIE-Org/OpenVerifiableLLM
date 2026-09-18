"""Durable local publication service; explicit systemd/publisher doubles only here."""
from pathlib import Path
import copy
import subprocess
import sys
import time
import pytest
import persistent_publication as m
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,write_json


def fixture(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir()
    for name in ('src','scripts','requirements'):(project/name).mkdir()
    (project/'scripts/persistent_publication.py').write_text('# explicit fixture source')
    (project/'src/module.py').write_text('# explicit fixture numerical source')
    (project/'requirements/publisher.lock').write_text('# explicit fixture dependency')
    monkeypatch.setattr(m,'ROOT',project)
    args={}
    for name in m.PARAMETERS[:-1]:
        p=tmp_path/'inputs'/name;p.mkdir(parents=True)
        (p/'data').write_bytes(name.encode());args[name]=str(p)
    args['output']=str(tmp_path/'published/boundary-00000')
    spec=m.selection('a'*64,'b'*64,1000,args,python=Path(sys.executable).resolve())
    state=tmp_path/'state';units=tmp_path/'units';calls=[];dropins=[''];foreign=[False];uncertain=[False]
    def execute(argv):
        calls.append(argv)
        if argv[2]=='show':
            values={'LoadState':'loaded','ActiveState':'active','SubState':'running','MainPID':'123',
                    'ExecMainCode':'0','ExecMainStatus':'0','Result':'success','InvocationID':'1'*32,
                    'FragmentPath':str(units/argv[3]) if not foreign[0] else '/foreign.service','DropInPaths':dropins[0]}
            return '\n'.join(k+'='+v for k,v in values.items())
        if argv[2]=='start' and uncertain[0]:raise subprocess.TimeoutExpired(argv,20)
        return ''
    def run():return m.start_or_adopt(spec,digest(spec),state,execute=execute,wall=lambda:100,unit_directory=units)
    return spec,state,units,calls,dropins,foreign,uncertain,run


def test_selection_rechecks_all_complete_inputs_and_executable_sources(tmp_path,monkeypatch):
    spec,*_=fixture(tmp_path,monkeypatch)
    assert m.validate(spec,digest(spec))[0]==m.ROOT
    (Path(spec['arguments']['packet'])/'unselected').write_text('new')
    with pytest.raises(EvidenceError,match='completeness'):m.validate(spec,digest(spec))


@pytest.mark.parametrize('damage',['source','dependency','new-source','python','input','symlink'])
def test_changed_selection_fails_before_service_start(tmp_path,monkeypatch,damage):
    spec,state,units,calls,_,_,_,run=fixture(tmp_path,monkeypatch)
    if damage=='source':(m.ROOT/'src/module.py').write_text('different')
    elif damage=='dependency':(m.ROOT/'requirements/publisher.lock').write_text('different')
    elif damage=='new-source':(m.ROOT/'scripts/new.py').write_text('new')
    elif damage=='python':spec['python_sha256']='c'*64
    elif damage=='input':(Path(spec['arguments']['chain-directory'])/'data').write_text('different')
    else:(Path(spec['arguments']['packet'])/'alias').symlink_to('/etc/passwd')
    with pytest.raises(EvidenceError):run()
    assert not calls


@pytest.mark.parametrize('uncertain_start',[False,True])
def test_start_once_adoption_read_only_after_success_or_uncertain_start(tmp_path,monkeypatch,uncertain_start):
    spec,state,units,calls,_,_,uncertain,run=fixture(tmp_path,monkeypatch)
    uncertain[0]=uncertain_start
    if uncertain_start:
        with pytest.raises(subprocess.TimeoutExpired):run()
    else:run()
    old=read_json(state/'start-fence.json');assert old['runtime_seconds']==870
    before=len(calls);run()
    assert all(c[2]=='show' for c in calls[before:])
    assert sum(c[2]=='start' for c in calls)==1
    assert read_json(state/'start-fence.json')==old
    unit=(state/'unit.service').read_text()
    assert 'RuntimeMaxSec=870' in unit and 'RemainAfterExit=no' in unit and 'KillMode=control-group' in unit
    assert 'Restart=no' in unit and '-I -B' in unit


@pytest.mark.parametrize('damage',['unit','fence','unit-copy','foreign','drop-in','selection'])
def test_adoption_rejects_rewritten_service_identity_without_restart(tmp_path,monkeypatch,damage):
    spec,state,units,calls,dropins,foreign,_,run=fixture(tmp_path,monkeypatch);result=run();before=len(calls)
    if damage=='unit':(units/result['unit']).write_text('[Service]\nExecStart=/bin/true\n')
    elif damage=='unit-copy':(state/'unit.service').write_text('wrong')
    elif damage=='fence':
        v=read_json(state/'start-fence.json');v['requested_epoch']+=1;write_json(state/'start-fence.json',v)
    elif damage=='foreign':foreign[0]=True
    elif damage=='drop-in':dropins[0]='/tmp/override.conf'
    else:spec['deadline_epoch']+=1
    with pytest.raises(EvidenceError):run()
    assert not any(c[2]=='start' for c in calls[before:])


@pytest.mark.parametrize('mode',['foreign','drop-in','preexisting'])
def test_foreign_unit_never_started(tmp_path,monkeypatch,mode):
    spec,state,units,calls,dropins,foreign,_,run=fixture(tmp_path,monkeypatch)
    if mode=='foreign':foreign[0]=True
    elif mode=='drop-in':dropins[0]='/tmp/override.conf'
    else:
        units.mkdir();(units/('ovllm-publication-'+digest(spec)+'.service')).write_text('foreign')
    with pytest.raises(EvidenceError):run()
    assert not any(c[2]=='start' for c in calls)


@pytest.mark.parametrize('which',['state-in-input','input-in-state','output-in-input','state-in-output'])
def test_mutable_paths_cannot_contaminate_selected_evidence(tmp_path,monkeypatch,which):
    spec,state,units,calls,_,_,_,_=fixture(tmp_path,monkeypatch)
    if which=='state-in-input':state=Path(spec['arguments']['packet'])/'state'
    elif which=='input-in-state':state=tmp_path/'inputs'
    elif which=='output-in-input':spec['arguments']['output']=spec['arguments']['packet']+'/output'
    else:state=Path(spec['arguments']['output'])/'state'
    with pytest.raises(EvidenceError,match='overlap'):
        m.start_or_adopt(spec,digest(spec),state,execute=lambda a:calls.append(a),wall=lambda:100,unit_directory=units)
    assert not calls


def test_expired_start_does_not_renew_deadline(tmp_path,monkeypatch):
    spec,state,units,calls,_,_,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(EvidenceError,match='insufficient'):
        m.start_or_adopt(spec,digest(spec),state,execute=lambda a:calls.append(a),wall=lambda:980,unit_directory=units)
    assert not calls


@pytest.mark.parametrize('value',['/tmp/a b','/tmp/a%','/tmp/a/../b','/tmp//a','relative','/tmp/a\nX=y'])
def test_unit_injection_paths_rejected(value):
    with pytest.raises(EvidenceError):m.path(value)
