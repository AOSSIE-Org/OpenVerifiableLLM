from pathlib import Path
import hashlib
import os
import subprocess
import sys

import pytest

from ovl_pipeline import canonical, runtime_launch
from ovl_pipeline.canonical import file_hash, host_cpu_model, host_executable_path
import pod_diagnostic
import pod_runtime_setup
import pod_verifier_setup


def test_host_executable_path_resolves_to_existing_file():
    exe = host_executable_path()
    assert isinstance(exe, Path)
    assert exe.is_file()
    assert exe.stat().st_size > 0
    assert file_hash(exe) == file_hash(Path(sys.executable).resolve())


def test_host_executable_path_consistent_across_scripts():
    diag_exe = pod_diagnostic.executable_path()
    runtime_exe = pod_runtime_setup.executable_path()
    verifier_exe = pod_verifier_setup.executable_path()

    assert diag_exe.is_file()
    assert runtime_exe.is_file()
    assert verifier_exe.is_file()
    assert file_hash(diag_exe) == file_hash(host_executable_path())
    assert file_hash(runtime_exe) == file_hash(host_executable_path())
    assert file_hash(verifier_exe) == file_hash(host_executable_path())


def test_host_executable_path_prefers_proc_when_present(tmp_path, monkeypatch):
    fake_proc = tmp_path / "proc_self_exe"
    fake_proc.write_bytes(b"mock binary content")

    def mock_path(p):
        if str(p) == "/proc/self/exe":
            return fake_proc
        return Path(p)

    monkeypatch.setattr(canonical, "Path", mock_path)
    res = canonical.host_executable_path()
    assert res == fake_proc


def test_host_cpu_model_returns_string_or_none():
    model = host_cpu_model()
    assert model is None or isinstance(model, str)
    if model is not None:
        assert len(model.strip()) > 0


def test_host_cpu_model_falls_back_when_proc_absent(tmp_path, monkeypatch):
    non_existent = tmp_path / "absent_cpuinfo"

    def mock_path(p):
        if str(p) == "/proc/cpuinfo":
            return non_existent
        return Path(p)

    monkeypatch.setattr(canonical, "Path", mock_path)
    model = canonical.host_cpu_model()
    assert model is None or isinstance(model, str)


def test_pod_diagnostic_records_actual_bootstrap_executable_hash(tmp_path):
    source = tmp_path / "payload"
    with source.open("wb") as f:
        f.truncate(8 * 1024**2)

    def smi(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0, b"NVIDIA GeForce RTX 5090, Fixture UUID, 580.65.06, 1, 00:00\n", b""
        )

    output = tmp_path / "diagnostic_out"
    report = pod_diagnostic.run(source, output, "NVIDIA GeForce RTX 5090", execute=smi)
    assert report["result"] == "OBSERVED_NOT_ADMITTED"
    assert report["bootstrap_executable_sha256"] == file_hash(host_executable_path())
    assert (output / "diagnostic.json").is_file()
