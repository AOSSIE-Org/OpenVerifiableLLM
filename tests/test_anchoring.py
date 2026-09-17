"""Local policy/admission tests. A live Sigstore positive test is separately archived."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib.util
import subprocess

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID, ObjectIdentifier
from pyasn1.codec.der.encoder import encode
from pyasn1.type.char import UTF8String
import pytest
from sigstore.errors import VerificationError

from ovl_pipeline.anchoring import ISSUER, REPOSITORY, WORKFLOW, PublisherPolicy, verify_anchor
from ovl_pipeline.canonical import EvidenceError, canonical, digest, read_json


def configured(value):
    return PublisherPolicy("ovl.publisher-policy.v1", REPOSITORY, WORKFLOW, ISSUER,
                           "refs/heads/feat/verifiable-wikipedia-pipeline", "a" * 40,
                           digest(value), "sigstore-production-tuf")


def certificate(policy, overrides=None):
    """Self-signed test leaf for policy only; cannot pass full Sigstore trust."""
    overrides = overrides or {}
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-only")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(1)
            .not_valid_before(now - timedelta(minutes=1)).not_valid_after(now + timedelta(minutes=1))
            .add_extension(x509.SubjectAlternativeName([x509.UniformResourceIdentifier(overrides.get("identity", policy.identity))]), False))
    values = {"1": policy.issuer, "6": policy.ref,
              "12": f"https://github.com/{policy.repository}", "13": policy.source_revision}
    for suffix, value in values.items():
        value = overrides.get(suffix, value)
        if value is None:
            continue
        encoded = encode(UTF8String(value)) if suffix in ("12", "13") else value.encode()
        cert = cert.add_extension(x509.UnrecognizedExtension(ObjectIdentifier("1.3.6.1.4.1.57264.1." + suffix), encoded), False)
    return cert.sign(key, hashes.SHA256())


def test_exact_certificate_policy():
    p = configured({"test": True})
    p.certificate_policy().verify(certificate(p))
    for change in [{"identity": p.identity + "-other"}, {"1": "https://evil.example"},
                   {"6": "refs/heads/main"}, {"12": "https://github.com/attacker/OpenVerifiableLLM"},
                   {"13": "b" * 40}, {"13": None}]:
        with pytest.raises((VerificationError, x509.ExtensionNotFound)):
            p.certificate_policy().verify(certificate(p, change))


@pytest.mark.parametrize("change", [
    {"schema": "future-policy"}, {"repository": "attacker/OpenVerifiableLLM"},
    {"issuer": "https://evil.example"}, {"workflow": ".github/workflows/evil.yml"},
    {"ref": "refs/pull/101/merge"}, {"source_revision": "a" * 39},
    {"statement_sha256": "a" * 63}, {"trust_root": "bundle-supplied"},
])
def test_policy_rejects_unknown_or_unapproved(change):
    with pytest.raises(EvidenceError):
        replace(configured({}), **change).validate()


def test_missing_forged_and_changed_bundle_fail(tmp_path):
    statement, bundle = tmp_path / "statement.json", tmp_path / "bundle.json"
    value = {"schema": "test", "claimed_result": "PASS"}
    statement.write_bytes(canonical(value))
    p = configured(value)
    with pytest.raises(EvidenceError):
        verify_anchor(statement, bundle, p)
    for data in [b'{}', b'{"mediaType":"evil","mediaType":"evil"}',
                 canonical({"mediaType": "application/vnd.dev.sigstore.bundle.v0.3+json"})]:
        bundle.write_bytes(data)
        with pytest.raises(EvidenceError):
            verify_anchor(statement, bundle, p)
    statement.write_bytes(canonical({"claimed_result": "EVIL"}))
    with pytest.raises(EvidenceError, match="selected digest"):
        verify_anchor(statement, bundle, p)


def test_symlink_and_oversized_statements_rejected(tmp_path):
    statement, bundle = tmp_path / "statement.json", tmp_path / "bundle.json"
    bundle.write_text('{}')
    statement.symlink_to(bundle)
    with pytest.raises(EvidenceError, match="regular"):
        verify_anchor(statement, bundle, configured({}))
    statement.unlink()
    statement.write_bytes(b' ' * (16 * 1024 * 1024 + 1))
    with pytest.raises(EvidenceError, match="size"):
        verify_anchor(statement, bundle, configured({}))


def test_smoke_construction_bound_to_clean_push_checkout(tmp_path):
    source = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("identity_smoke", source / "scripts/anchor_identity_smoke.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    names = [WORKFLOW, "scripts/anchor_identity_smoke.py", "src/ovl_pipeline/anchoring.py",
             "src/ovl_pipeline/canonical.py", "requirements/anchoring.in", "requirements/anchoring.lock"]
    for n in names:
        out = tmp_path / n; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((source / n).read_bytes())
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL).decode().strip()
    git("init"); git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "test")
    env = {"GITHUB_REPOSITORY": REPOSITORY, "GITHUB_REF": "refs/heads/feat/verifiable-wikipedia-pipeline",
           "GITHUB_SHA": git("rev-parse", "HEAD"), "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1"}
    for key, value in [("GITHUB_SHA", "b" * 40), ("GITHUB_EVENT_NAME", "pull_request"),
                       ("GITHUB_REPOSITORY", "attacker/project"), ("GITHUB_RUN_ID", "")]:
        with pytest.raises(EvidenceError):
            mod.generate(tmp_path, {**env, key: value})
    original = (tmp_path / WORKFLOW).read_bytes()
    (tmp_path / WORKFLOW).write_bytes(original + b"# changed\n")
    with pytest.raises(EvidenceError, match="modifications"):
        mod.generate(tmp_path, env)
    (tmp_path / WORKFLOW).write_bytes(original)
    mod.generate(tmp_path, env)
    s = read_json(tmp_path / "anchor-smoke/statement.json")
    p = PublisherPolicy(**read_json(tmp_path / "anchor-smoke/ci-self-check-policy.json"))
    p.validate()
    assert p.statement_sha256 == digest(s) and s["source_revision"] == env["GITHUB_SHA"]
    assert s["production_admission"] == "NOT_RUN" and s["training_replay"] == "NOT_RUN"
