"""Fresh-process verifier import closure; does not emulate CUDA qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from ovl_pipeline import gpu
from ovl_pipeline.canonical import EvidenceError


def test_pilot_initialization_and_production_imports_keep_complete_library_closure():
    code='''import json
from ovl_pipeline import gpu_pilot,gpu
before=gpu.mapped_libraries()
modules=gpu.verification_runtime_imports()
selected=gpu.mapped_libraries()
from ovl_pipeline import initialization,production_record,production_replay
from sigstore.models import Bundle,ClientTrustConfig
from sigstore.verify import Verifier,policy
after=gpu.mapped_libraries()
print(json.dumps({'modules':modules,'before':before,'selected':selected,'after':after}))
'''
    result=subprocess.run([sys.executable,'-B','-c',code],env={**os.environ,'PYTHONPATH':str(Path(__file__).parents[1]/'src')},
                          check=True,capture_output=True,text=True,timeout=60)
    value=json.loads(result.stdout)
    assert value['modules']==['sigstore.models','sigstore.verify','sigstore.verify.policy']
    assert value['selected']==value['after']
    assert all(item in value['selected'] for item in value['before'])
    assert any('_pydantic_core' in item['name'] for item in value['selected'])
    assert any('_rust.' in item['name'] for item in value['selected'])
    assert all(len(item['sha256'])==64 and item['bytes']>0 for item in value['selected'])


def test_wrong_verifier_version_fails_before_importing_extensions(monkeypatch):
    monkeypatch.setattr(gpu.importlib.metadata,'version',lambda name:'0.0.0')
    def forbidden(name):raise AssertionError('wrong runtime may not be preloaded')
    monkeypatch.setattr(gpu.importlib,'import_module',forbidden)
    with pytest.raises(EvidenceError,match='verification runtime version'):gpu.verification_runtime_imports()


def test_missing_verifier_extension_is_not_silently_omitted(monkeypatch):
    monkeypatch.setattr(gpu.importlib.metadata,'version',lambda name:'4.5.0')
    def missing(name):raise ImportError('explicit missing verifier dependency')
    monkeypatch.setattr(gpu.importlib,'import_module',missing)
    with pytest.raises(ImportError,match='missing verifier dependency'):gpu.verification_runtime_imports()
