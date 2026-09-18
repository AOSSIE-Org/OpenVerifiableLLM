from pathlib import Path
import json
import os
import py_compile
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from ovl_pipeline import runtime_launch
from ovl_pipeline.canonical import EvidenceError,read_json,write_json
from test_runtime_audit import fixture


def setup(tmp_path):
    lock,wheels,paths=fixture(tmp_path);venv=tmp_path/'venv';(venv/'bin').mkdir(parents=True)
    (venv/'bin/python').symlink_to(sys.executable);site=venv/'lib/python3.12/site-packages';site.parent.mkdir(parents=True);paths['site'].rename(site)
    src=Path(__file__).resolve().parents[1]/'src'
    return lock,wheels,venv,src,tmp_path/'launch'


def test_parent_audits_before_target_and_sets_fresh_import_environment(tmp_path,monkeypatch):
    args=setup(tmp_path);calls=[]
    monkeypatch.setenv('PYTHONPATH','untrusted');monkeypatch.setenv('LD_PRELOAD','untrusted')
    def execute(command,env,check):
        calls.append(command);assert '-s' in command and '-S' in command and '-P' in command
        assert 'PYTHONPATH' not in env and 'LD_PRELOAD' not in env and env['PYTHONHASHSEED']=='0'
        assert read_json(args[-1]/'installed-audit.json')['result']=='PASS'
        assert list((args[-1]/'pycache').iterdir())==[]
        return SimpleNamespace(returncode=0)
    r=runtime_launch.launch(*args,'ovl_pipeline',['fixture','--help'],execute=execute)
    assert r['exit_code']==0 and len(calls)==1
    with pytest.raises(EvidenceError,match='fresh'):runtime_launch.launch(*args,'ovl_pipeline',[],execute=execute)
    assert len(calls)==1


def test_bad_installed_bytes_prevent_any_target_execution(tmp_path):
    args=setup(tmp_path);(args[2]/'lib/python3.12/site-packages/example/__init__.py').write_bytes(b'tampered')
    def forbidden(*a,**k):raise AssertionError('must not start')
    with pytest.raises(EvidenceError,match='differs'):runtime_launch.launch(*args,'ovl_pipeline',[],execute=forbidden)
    assert not args[-1].exists()


def test_target_failure_preserves_audit_and_exit_receipt(tmp_path):
    args=setup(tmp_path)
    with pytest.raises(EvidenceError,match='target process failed'):
        runtime_launch.launch(*args,'ovl_pipeline',[],execute=lambda *a,**k:SimpleNamespace(returncode=7))
    assert read_json(args[-1]/'process.json')['exit_code']==7
    assert (args[-1]/'wheel-payloads.json').is_file()


def test_gpu_launch_admission_rejects_unconstrained_process(monkeypatch):
    monkeypatch.delenv('OVL_AUDITED_RUNTIME_LAUNCH',raising=False)
    with pytest.raises(EvidenceError,match='external audited launcher'):runtime_launch.current_launch()


def test_real_bootstrap_ignores_hostile_source_bytecode_and_site_hooks(tmp_path):
    src=tmp_path/'src';site=tmp_path/'site';cache=tmp_path/'cache'
    for p in (src,site,cache):p.mkdir()
    probe=src/'probe.py';probe.write_text('print("EVIL!")\n');py_compile.compile(str(probe),doraise=True)
    stamp=probe.stat();probe.write_text('print("CLEAN")\n');os.utime(probe,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    # Prove the source-side cached code is actually dangerous when used normally.
    env={k:v for k,v in os.environ.items() if not k.startswith('PYTHON')};env['PYTHONHASHSEED']='0'
    bad=subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,'+repr(str(src))+');import probe'],env=env,capture_output=True,text=True,check=True)
    assert bad.stdout.strip()=='EVIL!'
    sentinel=tmp_path/'hook-ran';(site/'evil.pth').write_text('import pathlib; pathlib.Path('+repr(str(sentinel))+').touch()\n')
    record=tmp_path/'launch.json';write_json(record,{'schema':'ovl.audited-runtime-launch.v1','source':str(src),'site':str(site),
        'pycache_prefix':str(cache),'module':'probe','arguments':[]})
    bootstrap=Path(runtime_launch.__file__).with_name('runtime_bootstrap.py')
    good=subprocess.run([sys.executable,'-s','-S','-P','-X','pycache_prefix='+str(cache),str(bootstrap),
        '--source',str(src),'--site',str(site),'--launch-record',str(record),'--module','probe','--'],env=env,capture_output=True,text=True,check=True)
    assert good.stdout.strip()=='CLEAN' and not sentinel.exists()


def test_bootstrap_rejects_launch_record_argument_substitution(tmp_path):
    src=tmp_path/'src';site=tmp_path/'site';cache=tmp_path/'cache'
    for p in (src,site,cache):p.mkdir()
    record=tmp_path/'launch.json';write_json(record,{'schema':'ovl.audited-runtime-launch.v1','source':str(src),'site':str(site),
        'pycache_prefix':str(cache),'module':'probe','arguments':['original']})
    env={k:v for k,v in os.environ.items() if not k.startswith('PYTHON')};env['PYTHONHASHSEED']='0'
    result=subprocess.run([sys.executable,'-s','-S','-P','-X','pycache_prefix='+str(cache),str(Path(runtime_launch.__file__).with_name('runtime_bootstrap.py')),
        '--source',str(src),'--site',str(site),'--launch-record',str(record),'--module','probe','--','changed'],env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'differs from audited parent record' in result.stderr


def test_python_origin_damage_stops_before_target_start(tmp_path):
    from test_python_origin import archive
    from ovl_pipeline import python_origin
    args=setup(tmp_path);tar,sha=archive(tmp_path);root=tmp_path/'public-python'
    python_origin.extract(tar,sha,root)
    link=args[2]/'bin/python';link.unlink();link.symlink_to(root/'python/bin/python3.12')
    (root/'python/lib/python3.12/example.py').write_bytes(b'altered executable source')
    with pytest.raises(EvidenceError,match='installed Python bytes differ'):
        runtime_launch.launch(*args,'ovl_pipeline',[],interpreter_archive=tar,interpreter_sha256=sha,interpreter_root=root,
                             execute=lambda *a,**k:pytest.fail('unverified interpreter must not start'))
    assert not args[-1].exists()


def test_gpu_refuses_wheel_only_audit_before_any_cuda_initialization(monkeypatch):
    from ovl_pipeline import gpu
    monkeypatch.setattr(gpu.torch,'__version__','2.14.0+cu130');monkeypatch.setattr(gpu.torch.version,'cuda','13.0')
    monkeypatch.setattr(runtime_launch,'current_launch',lambda:{'interpreter_origin':None})
    monkeypatch.setattr(gpu.torch.cuda,'init',lambda:pytest.fail('origin gate must precede CUDA'))
    with pytest.raises(EvidenceError,match='public interpreter payloads'):
        gpu.configure({'schema':'ovl.gpu-kernel.v1','precision':'fp32'})
