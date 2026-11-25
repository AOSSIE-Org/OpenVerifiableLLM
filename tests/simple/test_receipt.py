"""CPU unit tests for signed check receipts (simple/receipt.py).

Tests key generation, DSSE sealing, in-toto statement verification,
and failure modes: wrong key, tampered fields, wrong segment for round,
mismatched release anchor, and malformed inputs.
"""
import base64
import json
from pathlib import Path
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simple"))
import receipt  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MAIN_ANCHOR = FIXTURES / "simple-run-anchor.json"
FORK_ANCHOR = FIXTURES / "fork-anchor.json"
ROUNDS_FILE = FIXTURES / "drand_quicknet_rounds.json"

ROUNDS_DATA = json.loads(ROUNDS_FILE.read_text(encoding="utf-8"))
ROUNDS_MAP = {r: v["signature"] for r, v in ROUNDS_DATA["rounds"].items()}
SAMPLE_ROUND = 33803213
SAMPLE_SIG = ROUNDS_MAP[str(SAMPLE_ROUND)]


def make_key(path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes_raw()
    path.write_bytes(raw)
    path.with_name(path.name + ".pub").write_bytes(key.public_key().public_bytes_raw())
    return key


def test_keygen_creates_keys_and_refuses_overwrite(tmp_path):
    key_path = tmp_path / "test.key"
    res = receipt.main(["keygen", "--out", str(key_path), "--allow-in-repo"])
    assert res == 0
    assert key_path.exists()
    assert len(key_path.read_bytes()) == 32
    pub_path = tmp_path / "test.key.pub"
    assert pub_path.exists()
    assert len(pub_path.read_bytes()) == 32

    # Refuse overwrite via API and CLI
    with pytest.raises(receipt.Invalid, match="already exists"):
        receipt.keygen(str(key_path), allow_in_repo=True)
    assert receipt.main(["keygen", "--out", str(key_path), "--allow-in-repo"]) == 1


def test_keygen_refuses_inside_git_without_flag(tmp_path):
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    key_path = tmp_path / "leak.key"
    with pytest.raises(receipt.Invalid, match="inside git repo"):
        receipt.keygen(str(key_path), allow_in_repo=False)
    assert receipt.main(["keygen", "--out", str(key_path)]) == 1


def test_sign_and_verify_valid_receipt(tmp_path):
    key_file = tmp_path / "checker.key"
    key = make_key(key_file)

    env = receipt.sign_receipt(
        key=key,
        anchor_path=MAIN_ANCHOR,
        round_num=SAMPLE_ROUND,
        signature_hex=SAMPLE_SIG,
        outcome="MATCH"
    )

    res = receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)
    assert res["verdict"] == "VALID"
    assert res["beacon_verified"] is True
    assert res["round"] == SAMPLE_ROUND
    assert res["results"][0]["outcome"] == "MATCH"
    assert len(res["segments"]) == 1


def test_verify_rejects_wrong_key(tmp_path):
    key_a = make_key(tmp_path / "a.key")
    key_b = make_key(tmp_path / "b.key")

    env = receipt.sign_receipt(
        key=key_a,
        anchor_path=MAIN_ANCHOR,
        round_num=SAMPLE_ROUND,
        signature_hex=SAMPLE_SIG,
        outcome="MATCH"
    )

    # Tamper: replace checker public_key with key_b, keeping key_a's signature
    payload = json.loads(base64.b64decode(env["payload"]))
    payload["predicate"]["checker"]["public_key"] = receipt.public_hex(key_b)
    payload["predicate"]["checker"]["key_fp"] = receipt.fingerprint(receipt.public_hex(key_b))
    tampered_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    env["payload"] = base64.b64encode(tampered_bytes).decode("ascii")
    env["signatures"][0]["keyid"] = receipt.fingerprint(receipt.public_hex(key_b))

    with pytest.raises(receipt.Invalid, match="signature verification failed"):
        receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)


def test_verify_rejects_changed_payload_field(tmp_path):
    key = make_key(tmp_path / "checker.key")
    env = receipt.sign_receipt(
        key=key,
        anchor_path=MAIN_ANCHOR,
        round_num=SAMPLE_ROUND,
        signature_hex=SAMPLE_SIG,
        outcome="MISMATCH"
    )

    # Tamper with payload: change outcome to MATCH without updating signature
    payload = json.loads(base64.b64decode(env["payload"]))
    payload["predicate"]["results"][0]["outcome"] = "MATCH"
    tampered_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    env["payload"] = base64.b64encode(tampered_bytes).decode("ascii")

    with pytest.raises(receipt.Invalid, match="signature verification failed"):
        receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)


def test_verify_rejects_wrong_segment_for_round(tmp_path):
    key = make_key(tmp_path / "checker.key")
    pk_hex = receipt.public_hex(key)
    key_fp = receipt.fingerprint(pk_hex)

    rel = receipt.release_of(MAIN_ANCHOR)
    n = rel["chain_entries"] - 1
    rand_bytes = receipt.parse_randomness(SAMPLE_SIG)
    derived_segs = receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 1)

    # Deliberately sign for a different segment
    wrong_seg = (derived_segs[0] + 5) % n
    with pytest.raises(receipt.Invalid, match="do not match derived segments"):
        receipt.sign_receipt(
            key=key,
            anchor_path=MAIN_ANCHOR,
            round_num=SAMPLE_ROUND,
            signature_hex=SAMPLE_SIG,
            segments=[wrong_seg]
        )

    # Construct envelope with wrong segment directly, signed by key
    challenge = {
        "beacon_chain_hash": receipt.BEACON_CHAIN_HASH,
        "round": SAMPLE_ROUND,
        "signature": SAMPLE_SIG,
        "selection": receipt.SELECTION,
        "j": 1,
        "m": 1,
        "k": 1,
        "previous_outcome": None
    }
    stmt = receipt.make_statement(
        rel,
        challenge,
        [{"segment": [wrong_seg, wrong_seg + 1], "outcome": "MATCH"}],
        {"key_fp": key_fp, "public_key": pk_hex}
    )
    env = receipt.seal(key, receipt.STATEMENT_TYPE, stmt)

    with pytest.raises(receipt.Invalid, match="do not match derived segments"):
        receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)


def test_verify_rejects_mismatched_anchor(tmp_path):
    key = make_key(tmp_path / "checker.key")
    env = receipt.sign_receipt(
        key=key,
        anchor_path=MAIN_ANCHOR,
        round_num=SAMPLE_ROUND,
        signature_hex=SAMPLE_SIG,
        outcome="MATCH"
    )

    with pytest.raises(receipt.Invalid, match="receipt release metadata does not match"):
        receipt.verify_receipt(env, FORK_ANCHOR, rounds=ROUNDS_MAP)


def test_sign_rejects_invalid_bounds(tmp_path):
    key = make_key(tmp_path / "checker.key")
    with pytest.raises(receipt.Invalid, match="invalid bounds"):
        receipt.sign_receipt(
            key=key,
            anchor_path=MAIN_ANCHOR,
            round_num=SAMPLE_ROUND,
            signature_hex=SAMPLE_SIG,
            m=0
        )


def test_verify_rejects_unknown_outcome(tmp_path):
    key = make_key(tmp_path / "checker.key")
    pk_hex = receipt.public_hex(key)
    key_fp = receipt.fingerprint(pk_hex)
    rel = receipt.release_of(MAIN_ANCHOR)
    n = rel["chain_entries"] - 1
    rand_bytes = receipt.parse_randomness(SAMPLE_SIG)
    seg = receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 1)[0]

    challenge = {
        "beacon_chain_hash": receipt.BEACON_CHAIN_HASH,
        "round": SAMPLE_ROUND,
        "signature": SAMPLE_SIG,
        "selection": receipt.SELECTION,
        "j": 1,
        "m": 1,
        "k": 1,
        "previous_outcome": None
    }
    stmt = receipt.make_statement(
        rel,
        challenge,
        [{"segment": [seg, seg + 1], "outcome": "BOGUS"}],
        {"key_fp": key_fp, "public_key": pk_hex}
    )
    env = receipt.seal(key, receipt.STATEMENT_TYPE, stmt)
    with pytest.raises(receipt.Invalid, match="unknown outcome"):
        receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)


def test_verify_rejects_bool_parameters(tmp_path):
    key = make_key(tmp_path / "checker.key")
    pk_hex = receipt.public_hex(key)
    key_fp = receipt.fingerprint(pk_hex)
    rel = receipt.release_of(MAIN_ANCHOR)
    n = rel["chain_entries"] - 1
    rand_bytes = receipt.parse_randomness(SAMPLE_SIG)
    seg = receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 1)[0]

    challenge = {
        "beacon_chain_hash": receipt.BEACON_CHAIN_HASH,
        "round": SAMPLE_ROUND,
        "signature": SAMPLE_SIG,
        "selection": receipt.SELECTION,
        "j": True,
        "m": 1,
        "k": 1,
        "previous_outcome": None
    }
    stmt = receipt.make_statement(
        rel,
        challenge,
        [{"segment": [seg, seg + 1], "outcome": "MATCH"}],
        {"key_fp": key_fp, "public_key": pk_hex}
    )
    env = receipt.seal(key, receipt.STATEMENT_TYPE, stmt)
    with pytest.raises(receipt.Invalid, match="must be integers"):
        receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)


def test_verify_rejects_wrong_beacon_chain_hash(tmp_path):
    key = make_key(tmp_path / "checker.key")
    pk_hex = receipt.public_hex(key)
    key_fp = receipt.fingerprint(pk_hex)
    rel = receipt.release_of(MAIN_ANCHOR)
    n = rel["chain_entries"] - 1
    rand_bytes = receipt.parse_randomness(SAMPLE_SIG)
    seg = receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 1)[0]

    challenge = {
        "beacon_chain_hash": "0" * 64,
        "round": SAMPLE_ROUND,
        "signature": SAMPLE_SIG,
        "selection": receipt.SELECTION,
        "j": 1,
        "m": 1,
        "k": 1,
        "previous_outcome": None
    }
    stmt = receipt.make_statement(
        rel,
        challenge,
        [{"segment": [seg, seg + 1], "outcome": "MATCH"}],
        {"key_fp": key_fp, "public_key": pk_hex}
    )
    env = receipt.seal(key, receipt.STATEMENT_TYPE, stmt)
    with pytest.raises(receipt.Invalid, match="beacon_chain_hash"):
        receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)


def test_verify_rejects_unverified_or_grinded_randomness(tmp_path):
    """Regression: grinding random 48-byte hex with round=1 to hit segment 5 must NOT return VALID."""
    key = make_key(tmp_path / "checker.key")
    pk_hex = receipt.public_hex(key)
    key_fp = receipt.fingerprint(pk_hex)
    rel = receipt.release_of(MAIN_ANCHOR)
    n = rel["chain_entries"] - 1

    # Grind a fake 48-byte signature until it selects segment 5
    fake_sig = None
    for i in range(500):
        candidate = (i.to_bytes(48, "big")).hex()
        rand_bytes = receipt.parse_randomness(candidate)
        if receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 1) == [5]:
            fake_sig = candidate
            break
    assert fake_sig is not None

    env = receipt.sign_receipt(
        key=key,
        anchor_path=MAIN_ANCHOR,
        round_num=1,
        signature_hex=fake_sig,
        outcome="MATCH"
    )

    # Default verification fails closed when no verified round is provided
    with pytest.raises(receipt.Invalid, match="not verified"):
        receipt.verify_receipt(env, MAIN_ANCHOR)

    # Verification against real known rounds also fails closed
    with pytest.raises(receipt.Invalid, match="not verified"):
        receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)

    # Even if the round number exists in ROUNDS_MAP, mismatched signature fails
    env_real_round = receipt.sign_receipt(
        key=key,
        anchor_path=MAIN_ANCHOR,
        round_num=SAMPLE_ROUND,
        signature_hex=fake_sig,
        outcome="MATCH"
    )
    with pytest.raises(receipt.Invalid, match="does not match verified beacon"):
        receipt.verify_receipt(env_real_round, MAIN_ANCHOR, rounds=ROUNDS_MAP)

    # With allow_unverified_beacon=True, verdict is WELL_FORMED (never VALID)
    res = receipt.verify_receipt(env, MAIN_ANCHOR, allow_unverified_beacon=True)
    assert res["verdict"] == "WELL_FORMED"
    assert res["beacon_verified"] is False


def test_verify_rejects_arbitrary_32_byte_randomness(tmp_path):
    key = make_key(tmp_path / "checker.key")
    with pytest.raises(receipt.Invalid, match="expected 48-byte BLS12-381 G1 signature"):
        receipt.sign_receipt(
            key=key,
            anchor_path=MAIN_ANCHOR,
            round_num=SAMPLE_ROUND,
            signature_hex="aa" * 32,
            outcome="MATCH"
        )


def test_sign_requires_report_or_explicit_outcome(tmp_path):
    key = make_key(tmp_path / "checker.key")
    with pytest.raises(receipt.Invalid, match="must provide --report or an explicit --outcome"):
        receipt.sign_receipt(
            key=key,
            anchor_path=MAIN_ANCHOR,
            round_num=SAMPLE_ROUND,
            signature_hex=SAMPLE_SIG
        )


def test_sign_with_local_report(tmp_path):
    key = make_key(tmp_path / "checker.key")
    pk_hex = receipt.public_hex(key)
    key_fp = receipt.fingerprint(pk_hex)

    rel = receipt.release_of(MAIN_ANCHOR)
    n = rel["chain_entries"] - 1
    rand_bytes = receipt.parse_randomness(SAMPLE_SIG)
    derived = receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 1)[0]

    report_file = tmp_path / "local-verifications.jsonl"
    report_entry = {
        "time": 1730000000,
        "environment": {"gpu": "NVIDIA GeForce RTX 5090", "torch": "2.8.0+cu128"},
        "chain_head": rel["chain_head_sha256"],
        "checks": [{"segment": [derived, derived + 1], "match": True}],
        "result": "PASS"
    }
    report_file.write_text(json.dumps(report_entry) + "\n", encoding="utf-8")

    env = receipt.sign_receipt(
        key=key,
        anchor_path=MAIN_ANCHOR,
        round_num=SAMPLE_ROUND,
        signature_hex=SAMPLE_SIG,
        report_path=report_file
    )

    verified = receipt.verify_receipt(env, MAIN_ANCHOR, rounds=ROUNDS_MAP)
    assert verified["verdict"] == "VALID"
    assert verified["beacon_verified"] is True
    assert verified["results"][0]["outcome"] == "MATCH"

    # Now verify with a mismatched check in report
    bad_report = tmp_path / "bad.jsonl"
    bad_report.write_text(json.dumps({
        "checks": [{"segment": [derived + 1, derived + 2], "match": True}]
    }) + "\n", encoding="utf-8")
    with pytest.raises(receipt.Invalid, match="report checked segments"):
        receipt.sign_receipt(
            key=key,
            anchor_path=MAIN_ANCHOR,
            round_num=SAMPLE_ROUND,
            signature_hex=SAMPLE_SIG,
            report_path=bad_report
        )


def test_derive_deterministic():
    rel = receipt.release_of(MAIN_ANCHOR)
    n = rel["chain_entries"] - 1
    rand_bytes = receipt.parse_randomness(SAMPLE_SIG)
    key_fp = "0" * 64

    segs1 = receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 1)
    segs2 = receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 1)
    assert segs1 == segs2
    assert len(segs1) == 1
    assert 0 <= segs1[0] < n

    # Multi-segment draw
    multi = receipt.select(receipt.seed_for(rand_bytes, rel["chain_head_sha256"], key_fp, 1), n, 3)
    assert len(multi) == 3
    assert len(set(multi)) == 3
    assert multi == sorted(multi)


def test_cli_roundtrip(tmp_path):
    key_path = tmp_path / "cli.key"
    receipt_path = tmp_path / "receipt.json"

    # 1. keygen
    r1 = subprocess.run([
        sys.executable, str(ROOT / "simple" / "receipt.py"),
        "keygen", "--out", str(key_path), "--allow-in-repo"
    ], capture_output=True, text=True, check=True)
    key_info = json.loads(r1.stdout)
    assert "key_fp" in key_info

    # 2. derive
    r2 = subprocess.run([
        sys.executable, str(ROOT / "simple" / "receipt.py"),
        "derive", "--anchor", str(MAIN_ANCHOR),
        "--key-fp", key_info["key_fp"],
        "--round", str(SAMPLE_ROUND),
        "--signature", SAMPLE_SIG,
        "--quiet"
    ], capture_output=True, text=True, check=True)
    seg_str = r2.stdout.strip()
    assert seg_str.isdigit()

    # 3. sign
    subprocess.run([
        sys.executable, str(ROOT / "simple" / "receipt.py"),
        "sign", "--key", str(key_path),
        "--anchor", str(MAIN_ANCHOR),
        "--round", str(SAMPLE_ROUND),
        "--signature", SAMPLE_SIG,
        "--outcome", "MATCH",
        "--out", str(receipt_path)
    ], capture_output=True, text=True, check=True)
    assert receipt_path.exists()

    # 4. verify with --rounds returns VALID
    r4 = subprocess.run([
        sys.executable, str(ROOT / "simple" / "receipt.py"),
        "verify", str(receipt_path),
        "--anchor", str(MAIN_ANCHOR),
        "--rounds", str(ROUNDS_FILE)
    ], capture_output=True, text=True, check=True)
    v_out = json.loads(r4.stdout)
    assert v_out["verdict"] == "VALID"
    assert v_out["beacon_verified"] is True
    assert v_out["checker_key_fp"] == key_info["key_fp"]

    # 4b. verify without --rounds fails closed by default
    r4_unverified = subprocess.run([
        sys.executable, str(ROOT / "simple" / "receipt.py"),
        "verify", str(receipt_path),
        "--anchor", str(MAIN_ANCHOR)
    ], capture_output=True, text=True)
    assert r4_unverified.returncode == 1
    assert "not verified" in r4_unverified.stderr

    # 4c. verify with --allow-unverified-beacon returns WELL_FORMED (not VALID)
    r4_well = subprocess.run([
        sys.executable, str(ROOT / "simple" / "receipt.py"),
        "verify", str(receipt_path),
        "--anchor", str(MAIN_ANCHOR),
        "--allow-unverified-beacon"
    ], capture_output=True, text=True, check=True)
    w_out = json.loads(r4_well.stdout)
    assert w_out["verdict"] == "WELL_FORMED"
    assert w_out["beacon_verified"] is False

    # 5. verify rejects corrupt JSON cleanly (exit 1, no traceback)
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("{not json", encoding="utf-8")
    r5 = subprocess.run([
        sys.executable, str(ROOT / "simple" / "receipt.py"),
        "verify", str(bad_file),
        "--anchor", str(MAIN_ANCHOR)
    ], capture_output=True, text=True)
    assert r5.returncode == 1
    assert "FAIL:" in r5.stderr
    assert "Traceback" not in r5.stderr

    # 6. derive rejects invalid key-fp cleanly
    r6 = subprocess.run([
        sys.executable, str(ROOT / "simple" / "receipt.py"),
        "derive", "--anchor", str(MAIN_ANCHOR),
        "--key-fp", "not-hex",
        "--round", str(SAMPLE_ROUND),
        "--signature", SAMPLE_SIG
    ], capture_output=True, text=True)
    assert r6.returncode == 1
    assert "FAIL:" in r6.stderr
    assert "Traceback" not in r6.stderr
