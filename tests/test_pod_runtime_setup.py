"""Bootstrap refusal tests; full public archive/wheel installation is separate evidence."""
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_runtime_setup as m
from ovl_pipeline.canonical import file_hash,inventory,write_json


def selection(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'module.py').write_bytes(b'operator selected source')
    wheels=tmp_path/'wheels';wheels.mkdir()
    for name in ('packaging','rfc8785'):(wheels/(name+'-1.0-py3-none-any.whl')).write_bytes(b'explicit invalid wheel; no import may reach it')
    value={'schema':'ovl.offline-runtime-setup.v1','source_root':'source','source_files':inventory(source,['module.py']),
           'dependency_lock':'lock','interpreter_archive':'python.tar.gz','interpreter_sha256':'a'*64,'wheels':'wheels',
           'bootstrap_wheels':[{'path':p.name,'sha256':file_hash(p)} for p in sorted(wheels.iterdir())]}
    config=tmp_path/'config.json';write_json(config,value);return config,value,source,wheels


@pytest.mark.parametrize('fault',['config-digest','source-bytes','unlisted-source','source-symlink','wheel-bytes','wheel-symlink','duplicate-wheel','source-traversal','unchecked-bytecode'])
def test_unauthorized_bootstrap_inputs_refused_before_import(tmp_path,fault,monkeypatch):
    config,value,source,wheels=selection(tmp_path);expected=file_hash(config)
    if fault=='config-digest':expected='0'*64
    elif fault=='source-bytes':(source/'module.py').write_bytes(b'changed source')
    elif fault=='unlisted-source':(source/'extra.py').write_bytes(b'extra')
    elif fault=='source-symlink':
        (source/'module.py').unlink();(source/'module.py').symlink_to(config)
    elif fault=='wheel-bytes':next(wheels.iterdir()).write_bytes(b'changed')
    elif fault=='wheel-symlink':
        target=next(wheels.iterdir());target.unlink();target.symlink_to(config)
    elif fault=='duplicate-wheel':value['bootstrap_wheels'][1]=value['bootstrap_wheels'][0]
    elif fault=='unchecked-bytecode':
        import py_compile,importlib.util
        poison=tmp_path/'poison.py';poison.write_text('raise RuntimeError("must never execute")\n')
        target=Path(importlib.util.cache_from_source(str(source/'module.py')));target.parent.mkdir()
        py_compile.compile(str(poison),cfile=str(target),invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    else:value['source_files'][0]['path']='../outside'
    if fault in ('duplicate-wheel','source-traversal'):write_json(config,value);expected=file_hash(config)
    original=list(sys.path)
    with pytest.raises(ValueError):m.selected(config,expected,tmp_path)
    assert sys.path==original


def test_bootstrap_cli_requires_isolated_no_site_parent(tmp_path):
    r=subprocess.run([sys.executable,str(Path(m.__file__)),'setup','--config',str(tmp_path/'none'),'--config-sha256','0'*64,
                      '--inputs',str(tmp_path),'--runtime',str(tmp_path/'runtime'),'--output',str(tmp_path/'output')],capture_output=True,text=True)
    assert r.returncode!=0 and 'requires -I -S' in r.stderr and not(tmp_path/'runtime').exists()


@pytest.mark.parametrize('layout',['same','nested','symlink','existing-output'])
def test_runtime_layout_preserves_existing_and_separates_evidence(tmp_path,layout):
    runtime=tmp_path/'runtime';output=tmp_path/'output'
    if layout=='same':output=runtime
    elif layout=='nested':output=runtime/'evidence'
    elif layout=='symlink':runtime.symlink_to(tmp_path)
    else:output.mkdir();(output/'sole-evidence').write_bytes(b'preserve')
    with pytest.raises(ValueError):m.layout(runtime,output)
    if layout=='existing-output':assert (output/'sole-evidence').read_bytes()==b'preserve'


def test_bootstrap_helper_digest_must_belong_to_its_own_lock_row(tmp_path):
    config,value,source,wheels=selection(tmp_path)
    for p in wheels.iterdir():p.write_bytes(('distinct '+p.name).encode())
    value['bootstrap_wheels']=[{'path':p.name,'sha256':file_hash(p)} for p in sorted(wheels.iterdir())]
    a,b=value['bootstrap_wheels'];lock=source/'lock'
    lock.write_text('packaging==1.0 --hash=sha256:'+b['sha256']+'\nrfc8785==1.0 --hash=sha256:'+a['sha256']+'\n')
    value['source_files']=inventory(source,['lock','module.py']);write_json(config,value)
    original=list(sys.path)
    with pytest.raises(ValueError,match='own pure-Python lock row'):m.selected(config,file_hash(config),tmp_path)
    assert sys.path==original


def test_setup_and_delegated_python_exclude_unaudited_bytecode(tmp_path):
    import os,py_compile
    package=tmp_path/'module';package.mkdir();probe=package/'probe.py'
    probe.write_text('print("EVIL")\n')
    py_compile.compile(str(probe),doraise=True,invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
    probe.write_text('print("CLEAN")\n')
    code='import sys;sys.path.insert(0,'+repr(str(package))+');import probe'
    plain={k:v for k,v in os.environ.items() if not k.startswith('PYTHON')}
    before=subprocess.run([sys.executable,'-B','-c',code],env=plain,capture_output=True,text=True,check=True)
    assert before.stdout.strip()=='EVIL'
    env={**plain,'PYTHONDONTWRITEBYTECODE':'1'}
    command=m.isolated_install_command(Path(sys.executable),'venv',tmp_path,env)
    delegated='import subprocess,sys;subprocess.run([sys.executable,"-c",'+repr(code)+'],check=True)'
    # Substitute a deterministic probe for the selected stdlib tool, preserving
    # exactly the setup startup flags and delegated environment.
    after=subprocess.run([*command[:-2],'-c',delegated],env=env,capture_output=True,text=True,check=True)
    assert after.stdout.strip()=='CLEAN'
    assert not list(Path(env['PYTHONPYCACHEPREFIX']).rglob('*.pyc'))
