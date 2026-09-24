"""Existing release fixtures exercised through the new complete driver.

Real synthetic transformations, CPU replay and downloaded payload checks; explicit
unsigned host/identity/GPU adapters. Isolated preparation process tested separately.
"""
import pytest
from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
import test_release_download as original
import verify_complete
import verify_release_complete
from ovl_pipeline import preparation


@pytest.mark.parametrize('incompatible',[False,True])
def test_external_release_full_path(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,incompatible):
    monkeypatch.setattr(original.m,'verify',verify_release_complete.verify)
    monkeypatch.setattr(verify_release_complete,'driver_identity',lambda p,**kw:{'explicit_test_double':'synthetic source context'})
    monkeypatch.setattr(verify_release_complete,'admit_runtime_source',lambda *a:None)
    monkeypatch.setattr(verify_complete,'admit_runtime_source',lambda *a:None)
    monkeypatch.setattr(verify_complete,'driver_identity',lambda p,**kw:{'explicit_test_double':'synthetic source context'})
    monkeypatch.setattr(verify_complete,'check_replay_launch',lambda *a:None)
    monkeypatch.setattr(verify_complete,'reconstruct',lambda checkout,statement,bundle,policy,raw,output,expected,execution:
        preparation.prepare_committed(statement,bundle,policy,raw/'wikipedia',raw/'conversation',output,
                                      expected_preparation=expected,resume=False))
    original.test_full_mode_calls_actual_complete_transformations_and_continuous_cpu_fixture_replay(
        cpu_runtime,inputs,prepared,tmp_path,monkeypatch,incompatible)
