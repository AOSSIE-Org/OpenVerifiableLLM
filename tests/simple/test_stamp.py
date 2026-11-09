import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simple import stamp


def test_corrupted_existing_timestamps_are_skipped(tmp_path):
    run = tmp_path / "run"
    timestamps = run / "timestamps"
    timestamps.mkdir(parents=True)

    entry = {"index": 1, "kind": "base"}

    damaged_files = [timestamps / f"entry-00001.{name}.tsr" for name in stamp.TSA]

    for path in damaged_files:
        path.write_bytes(b"corrupted timestamp")

    with patch.object(stamp.urllib.request, "urlopen") as mock_urlopen:
        stamp.stamp_entry(run, entry)

    mock_urlopen.assert_not_called()

    for path in damaged_files:
        assert path.read_bytes() == b"corrupted timestamp"


def test_system_ca_prefers_system_bundle(monkeypatch):
    monkeypatch.setattr(
        stamp.Path,
        "is_file",
        lambda self: self.as_posix() == "/etc/ssl/cert.pem",
    )

    def fail_import(name, *args, **kwargs):
        if name == "certifi":
            raise AssertionError("certifi should not be imported")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fail_import)

    assert stamp.system_ca() == "/etc/ssl/cert.pem"


def test_system_ca_uses_certifi_fallback(monkeypatch):
    monkeypatch.setattr(stamp.Path, "is_file", lambda self: False)

    fake_certifi = SimpleNamespace(where=lambda: "/fake/certifi-ca.pem")
    monkeypatch.setitem(sys.modules, "certifi", fake_certifi)

    assert stamp.system_ca() == "/fake/certifi-ca.pem"


def test_system_ca_returns_none_without_certifi(monkeypatch):
    monkeypatch.setattr(stamp.Path, "is_file", lambda self: False)

    original_import = __import__

    def fail_certifi_import(name, *args, **kwargs):
        if name == "certifi":
            raise ImportError("certifi unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_certifi_import)

    assert stamp.system_ca() is None


def test_digicert_verification_uses_system_ca(monkeypatch, tmp_path):
    monkeypatch.setattr(stamp, "system_ca", lambda: "/fake/system-ca.pem")

    run_mock = Mock(
        side_effect=[
            SimpleNamespace(stdout="Verification: OK"),
            SimpleNamespace(stdout="Time stamp: Nov 9, 2026"),
        ]
    )
    monkeypatch.setattr(stamp.subprocess, "run", run_mock)

    ok, timestamp = stamp.verify_reply(
        tmp_path / "reply.tsr",
        "digicert",
        ["-digest", "abc123"],
        (None, None),
    )

    assert ok is True
    assert timestamp == "Nov 9, 2026"

    verify_command = run_mock.call_args_list[0].args[0]
    assert verify_command[-2:] == ["-CAfile", "/fake/system-ca.pem"]
    assert "-untrusted" not in verify_command


def test_freetsa_verification_keeps_its_certificate_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(
        stamp,
        "system_ca",
        lambda: (_ for _ in ()).throw(AssertionError("FreeTSA must not use system_ca")),
    )

    run_mock = Mock(
        side_effect=[
            SimpleNamespace(stdout="Verification: OK"),
            SimpleNamespace(stdout="Time stamp: Nov 9, 2026"),
        ]
    )
    monkeypatch.setattr(stamp.subprocess, "run", run_mock)

    ca = tmp_path / "cacert.pem"
    tsa = tmp_path / "tsa.crt"

    ok, timestamp = stamp.verify_reply(
        tmp_path / "reply.tsr",
        "freetsa",
        ["-digest", "abc123"],
        (ca, tsa),
    )

    assert ok is True
    assert timestamp == "Nov 9, 2026"

    verify_command = run_mock.call_args_list[0].args[0]
    assert verify_command[-4:] == [
        "-CAfile",
        str(ca),
        "-untrusted",
        str(tsa),
    ]


def test_digicert_verification_fails_cleanly_without_ca_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(stamp, "system_ca", lambda: None)
    run_mock = Mock()
    monkeypatch.setattr(stamp.subprocess, "run", run_mock)

    result = stamp.verify_reply(
        tmp_path / "reply.tsr",
        "digicert",
        ["-digest", "abc123"],
        (None, None),
    )

    assert result == (False, "CA bundle unavailable")
    run_mock.assert_not_called()
