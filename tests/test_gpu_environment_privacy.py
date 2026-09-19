"""Synthetic device metadata only; no CUDA qualification or hardware attestation."""
from types import SimpleNamespace
import pytest
from ovl_pipeline import gpu,runtime_launch
from ovl_pipeline.canonical import EvidenceError


def configured(monkeypatch,reply):
    calls=[]
    def query(argv,**kw):
        assert argv==['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv,noheader,nounits']
        calls.append(argv);return reply
    monkeypatch.setattr(gpu,'_configured',True)
    monkeypatch.setattr(gpu.subprocess,'check_output',query)
    monkeypatch.setattr(gpu,'verification_runtime_imports',lambda:['synthetic-verifier'])
    monkeypatch.setattr(gpu.torch.cuda,'get_device_properties',lambda _:SimpleNamespace(name='Synthetic GPU',major=12,minor=0,total_memory=32*1024**3,multi_processor_count=1,warp_size=32))
    monkeypatch.setattr(gpu.torch.backends.cudnn,'version',lambda:123)
    monkeypatch.setattr(gpu.importlib.metadata,'distributions',lambda:[])
    monkeypatch.setattr(gpu,'mapped_libraries',lambda:[{'name':'libcuda.so','sha256':'a'*64}])
    monkeypatch.setattr(gpu,'host_runtime',lambda:{'synthetic':'initialization fingerprint'})
    monkeypatch.setattr(gpu,'flags',lambda:dict(gpu.REQUIRED_FLAGS))
    monkeypatch.setattr(gpu,'workspaces',lambda:dict(gpu.REQUIRED_WORKSPACES))
    monkeypatch.setattr(runtime_launch,'current_launch',lambda:{'synthetic':'audited-runtime'})
    return calls


def test_public_fingerprint_keeps_required_compatibility_without_device_identifier(monkeypatch):
    calls=configured(monkeypatch,'0, Synthetic GPU, 999.1\n')
    value=gpu.environment({'schema':'ovl.gpu-kernel.v1','precision':'bf16'})
    assert len(calls)==1
    assert set(value)=={'compatible','reproducibility_validation','production_admission'}
    c=value['compatible']
    assert c['driver_version']=='999.1' and c['gpu']['compute_capability']==[12,0]
    assert c['flags']==gpu.REQUIRED_FLAGS and c['blas_workspaces']==gpu.REQUIRED_WORKSPACES
    assert c['loaded_numerical_libraries']==[{'name':'libcuda.so','sha256':'a'*64}]
    assert c['installed_wheel_audit']=={'synthetic':'audited-runtime'}
    assert c['initialization_host']=={'synthetic':'initialization fingerprint'}
    assert value['reproducibility_validation']==value['production_admission']=='NOT_RUN'


@pytest.mark.parametrize('reply',['','0, Synthetic GPU, 999.1\n1, Synthetic GPU, 999.1\n','0, Synthetic GPU, synthetic-device-identifier, 999.1\n'])
def test_missing_multiple_or_unrequested_inventory_fields_fail_closed(monkeypatch,reply):
    calls=configured(monkeypatch,reply)
    with pytest.raises(EvidenceError,match='one physical GPU'):
        gpu.environment({'schema':'ovl.gpu-kernel.v1','precision':'bf16'})
    assert len(calls)==1
