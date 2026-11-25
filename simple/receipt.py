"""Signed check receipts with drand-selected segments (roadmap item 6, format v0).

A check result today is a self-reported line in local-verifications.jsonl or an issue comment.
A receipt instead is an in-toto Statement v1 inside a DSSE envelope, signed with the checker's
Ed25519 key, binding the release anchor, the chain head, the drand beacon round, and the drawn
segment.

Commands:
  1. keygen    generate an Ed25519 keypair (~/.ovl/checker.key and ~/.ovl/checker.key.pub)
  2. derive    recompute the assigned segment(s) for a given round, key and check index j
  3. sign      sign a check outcome into an in-toto DSSE envelope receipt
  4. verify    verify the receipt signature, payload integrity, anchor binding and segment match

Examples:
  python simple/receipt.py keygen --out ~/.ovl/checker.key
  python simple/receipt.py derive --anchor RUN/anchor.json --key-fp KEY_FP --round ROUND --signature SIG_HEX
  python simple/receipt.py sign --key ~/.ovl/checker.key --anchor RUN/anchor.json --round ROUND \\
      --signature SIG_HEX --report RUN/local-verifications.jsonl [--out receipt.json]
  python simple/receipt.py verify receipt.json --anchor RUN/anchor.json --rounds ROUNDS.json

Limitations of v0 (Milestone 1):
  - Receipts live with the release record on Hugging Face alongside verifications.jsonl, not in git.
  - This first milestone implements the receipt format, Ed25519 DSSE signing/verification, and
    segment derivation. Key registration and live drand BLS verification follow in Milestone 2;
    until then, `verify` requires `--rounds` (a verified round->signature map) to return VALID,
    or `--allow-unverified-beacon` to report WELL_FORMED (internal consistency only).
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

STATEMENT_TYPE = "application/vnd.in-toto+json"
RECEIPT_SCHEMA = "ovllm.check-receipt/v0"
PREDICATE_TYPE = "https://github.com/AOSSIE-Org/OpenVerifiableLLM/check-receipt/v0"
SELECTION = "ovllm-select-v0"
BEACON_CHAIN_HASH = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971"
OUTCOMES = ("MATCH", "MISMATCH", "WRONG_STACK", "ERROR", "ABANDONED")
MAX_M = 64


class Invalid(Exception):
    """Raised when a receipt or key fails validation."""


# ---------- encoding and signatures ----------

def canonical(obj):
    """Deterministic UTF-8 JSON byte representation for envelope digests."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def envelope_id(env):
    """SHA-256 digest of the canonical envelope."""
    return hashlib.sha256(canonical(env)).hexdigest()


def pae(payload_type, payload):
    """DSSE v1 Pre-Authentication Encoding."""
    t = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def load_key(path):
    """Load a raw 32-byte Ed25519 private seed."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    try:
        raw = Path(path).expanduser().read_bytes()
    except OSError as e:
        raise Invalid(f"cannot read key {path}: {e.strerror}") from None
    if len(raw) != 32:
        raise Invalid(f"{path} is not a raw 32-byte Ed25519 key (length={len(raw)})")
    return Ed25519PrivateKey.from_private_bytes(raw)


def public_hex(key):
    """Raw 32-byte public key as hex."""
    return key.public_key().public_bytes_raw().hex()


def fingerprint(public_key_hex):
    """Fingerprint key_fp is SHA-256 of the raw public key bytes."""
    return hashlib.sha256(bytes.fromhex(public_key_hex)).hexdigest()


def seal(key, payload_type, body):
    """Produce a DSSE envelope with a single Ed25519 signature."""
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sig = key.sign(pae(payload_type, payload))
    pk_hex = public_hex(key)
    return {
        "payloadType": payload_type,
        "payload": base64.b64encode(payload).decode("ascii"),
        "signatures": [{
            "keyid": fingerprint(pk_hex),
            "sig": base64.b64encode(sig).decode("ascii")
        }]
    }


def open_envelope(env, payload_type, public_key_hex):
    """Verify a single-signature DSSE envelope and return its decoded payload dict."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    if not isinstance(env, dict) or set(env) != {"payloadType", "payload", "signatures"}:
        raise Invalid("not a DSSE envelope (missing required top-level fields)")
    if env["payloadType"] != payload_type:
        raise Invalid(f"payloadType is {env['payloadType']!r}, expected {payload_type!r}")
    sigs = env.get("signatures")
    if not isinstance(sigs, list) or len(sigs) != 1 or not isinstance(sigs[0], dict):
        raise Invalid("expected exactly one signature in envelope")
    if set(sigs[0]) != {"keyid", "sig"}:
        raise Invalid("signature entry must have exactly keyid and sig fields")
    want_keyid = fingerprint(public_key_hex)
    if sigs[0].get("keyid") != want_keyid:
        raise Invalid(f"signature keyid {sigs[0].get('keyid')!r} does not match key fingerprint {want_keyid!r}")
    if not isinstance(env["payload"], str) or not isinstance(sigs[0]["sig"], str):
        raise Invalid("envelope payload and sig must be base64 strings")
    try:
        payload = base64.b64decode(env["payload"], validate=True)
        sig = base64.b64decode(sigs[0]["sig"], validate=True)
        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        pk.verify(sig, pae(payload_type, payload))
    except (InvalidSignature, ValueError, TypeError) as e:
        raise Invalid(f"signature verification failed: {e or 'invalid signature'}") from None
    try:
        body = json.loads(payload)
    except ValueError:
        raise Invalid("payload is not valid JSON") from None
    if not isinstance(body, dict):
        raise Invalid("payload is not a JSON object")
    return body


def unverified_payload(env):
    """Decode envelope payload without checking signature (to extract key identifiers)."""
    if not isinstance(env, dict) or "payload" not in env:
        raise Invalid("envelope has no payload field")
    try:
        raw = base64.b64decode(env["payload"], validate=True)
        return json.loads(raw)
    except Exception as e:
        raise Invalid(f"cannot decode envelope payload: {e}") from None


# ---------- deterministic selection (ovllm-select-v0) ----------

def parse_randomness(signature_hex):
    """Derive 32-byte randomness from a 48-byte compressed BLS12-381 round signature."""
    if not isinstance(signature_hex, str):
        raise Invalid("drand signature must be a hex string")
    try:
        raw = bytes.fromhex(signature_hex.strip())
    except ValueError:
        raise Invalid(f"signature {signature_hex!r} is not valid hex") from None
    if len(raw) != 48:
        raise Invalid(f"signature length is {len(raw)} bytes, expected 48-byte BLS12-381 G1 signature")
    return hashlib.sha256(raw).digest()


def seed_for(randomness, chain_head_hex, key_fp_hex, j):
    """Derive the deterministic seed for check j."""
    if not isinstance(chain_head_hex, str) or not re.fullmatch(r"[0-9a-f]{64}", chain_head_hex):
        raise Invalid(f"invalid chain_head_sha256 {chain_head_hex!r}")
    if not isinstance(key_fp_hex, str) or not re.fullmatch(r"[0-9a-f]{64}", key_fp_hex):
        raise Invalid(f"invalid key_fp {key_fp_hex!r}")
    if type(j) is not int or j < 1:
        raise Invalid(f"invalid check index j={j!r}")
    return hashlib.sha256(
        b"OVLLM-check-seed-v0"
        + randomness
        + bytes.fromhex(chain_head_hex)
        + bytes.fromhex(key_fp_hex)
        + j.to_bytes(8, "big")
    ).digest()


def draw(seed, n, ctr):
    """Uniform integer in [0, n) by rejection sampling over 64-bit words; returns (value, next_ctr)."""
    limit = (1 << 64) - ((1 << 64) % n)
    while True:
        x = int.from_bytes(
            hashlib.sha256(b"OVLLM-check-select-v0" + seed + ctr.to_bytes(4, "big")).digest()[:8],
            "big"
        )
        ctr += 1
        if x < limit:
            return x % n, ctr


def select(seed, n, k):
    """Pick k distinct segments out of n by partial Fisher-Yates shuffle, returned sorted."""
    if not 1 <= k <= n:
        raise Invalid(f"cannot pick k={k} segments from n={n}")
    pool, ctr, out = list(range(n)), 0, []
    for _ in range(k):
        i, ctr = draw(seed, len(pool), ctr)
        out.append(pool.pop(i))
    return sorted(out)


# ---------- release and statement ----------

def release_of(anchor_path):
    """Load release metadata and compute anchor_sha256."""
    path = Path(anchor_path)
    try:
        data = path.read_bytes()
        a = json.loads(data)
    except OSError as e:
        raise Invalid(f"cannot read anchor {path}: {e.strerror}") from None
    except ValueError:
        raise Invalid(f"anchor {path} is not valid JSON") from None

    if not isinstance(a, dict) or not {"chain_entries", "chain_head_sha256"}.issubset(a.keys()):
        raise Invalid(f"{path} is missing required anchor fields (chain_entries, chain_head_sha256)")
    if type(a["chain_entries"]) is not int or a["chain_entries"] < 2:
        raise Invalid(f"{path} has invalid chain_entries {a['chain_entries']!r}")
    if not isinstance(a["chain_head_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", a["chain_head_sha256"]):
        raise Invalid(f"{path} has invalid chain_head_sha256 {a['chain_head_sha256']!r}")

    return {
        "dataset": a.get("dataset"),
        "run_path": a.get("path"),
        "anchor_sha256": hashlib.sha256(data).hexdigest(),
        "chain_entries": a["chain_entries"],
        "chain_head_sha256": a["chain_head_sha256"],
    }


def make_statement(release, challenge, results, checker_info, extra=None):
    """Construct an in-toto Statement v1 for an ovllm check receipt."""
    predicate = {
        "schema": RECEIPT_SCHEMA,
        "release": release,
        "challenge": challenge,
        "results": results,
        "checker": checker_info,
    }
    if extra:
        predicate.update(extra)

    return {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": [
            {"name": "anchor.json", "digest": {"sha256": release["anchor_sha256"]}},
            {"name": "chain-head", "digest": {"sha256": release["chain_head_sha256"]}}
        ],
        "predicateType": PREDICATE_TYPE,
        "predicate": predicate
    }


def results_from_report(report_line, segments, outcome_override=None):
    """Extract results and environment stack from a local-verifications.jsonl line."""
    try:
        report = json.loads(report_line)
    except ValueError:
        raise Invalid("report line is not valid JSON") from None

    got = {tuple(c["segment"]): c.get("match") for c in report.get("checks", []) if "segment" in c}
    want = {(s, s + 1) for s in segments}
    if set(got) != want:
        raise Invalid(f"report checked segments {sorted(s for s, _ in got)}, but drawn segments are {segments}")

    results = []
    for s in segments:
        m = got[(s, s + 1)]
        res_outcome = outcome_override or ("MATCH" if m is True else "MISMATCH")
        results.append({"segment": [s, s + 1], "outcome": res_outcome})

    return results, report


# ---------- sign and verify APIs ----------

def keygen(out_path, allow_in_repo=False):
    """Generate an Ed25519 keypair and save as raw 32-byte seed and public key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    out = Path(out_path).expanduser()
    if out.exists():
        raise Invalid(f"{out} already exists; refusing to overwrite")
    if not allow_in_repo:
        cur = out.resolve().parent
        while cur != cur.parent:
            if (cur / ".git").exists():
                raise Invalid(f"{out} is inside git repo at {cur}; use --allow-in-repo to override")
            cur = cur.parent

    out.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    raw_private = key.private_bytes_raw()
    raw_public = key.public_key().public_bytes_raw()

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = 0o600
    fd = os.open(str(out), flags, mode)
    with os.fdopen(fd, "wb") as f:
        f.write(raw_private)
    out.with_name(out.name + ".pub").write_bytes(raw_public)

    pk_hex = raw_public.hex()
    return {"key_fp": fingerprint(pk_hex), "public_key": pk_hex}


def sign_receipt(key, anchor_path, round_num, signature_hex, segments=None, outcome=None,
                 j=1, m=1, k=1, code_commit="", stack=None, report_path=None, previous_ref=None):
    """Construct and sign a DSSE check receipt."""
    release = release_of(anchor_path)
    n = release["chain_entries"] - 1
    if not (type(j) is int and type(m) is int and type(k) is int and type(round_num) is int):
        raise Invalid("round, j, m, k must be integers")
    if round_num < 1:
        raise Invalid(f"invalid round {round_num!r}")
    if not (1 <= k <= n and 1 <= m <= MAX_M and 1 <= j <= m):
        raise Invalid(f"invalid bounds: k={k} (1..{n}), m={m} (1..{MAX_M}), j={j} (1..{m})")
    if outcome is not None and outcome not in OUTCOMES:
        raise Invalid(f"unknown outcome {outcome!r}, expected one of {OUTCOMES}")

    pk_hex = public_hex(key)
    key_fp = fingerprint(pk_hex)
    rand_bytes = parse_randomness(signature_hex)
    derived_segments = select(seed_for(rand_bytes, release["chain_head_sha256"], key_fp, j), n, k)

    if segments is None:
        segments = derived_segments
    else:
        if isinstance(segments, int) and not isinstance(segments, bool):
            segments = [segments]
        segments = sorted(segments)
        if segments != derived_segments:
            raise Invalid(f"specified segments {segments} do not match derived segments {derived_segments}")

    extra = {
        "code": {"repo": "AOSSIE-Org/OpenVerifiableLLM", "commit": code_commit},
        "stack": stack or {},
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    if report_path:
        p = Path(report_path)
        lines = p.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise Invalid(f"report {report_path} is empty")
        results, report = results_from_report(lines[-1], segments, outcome)
        extra["stack"] = report.get("environment", extra["stack"])
        extra["local_report_sha256"] = hashlib.sha256(lines[-1].encode("utf-8")).hexdigest()
        extra["started"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(report.get("time", time.time())))
    elif outcome is not None:
        results = [{"segment": [s, s + 1], "outcome": outcome} for s in segments]
    else:
        raise Invalid("must provide --report or an explicit --outcome")

    challenge = {
        "beacon_chain_hash": BEACON_CHAIN_HASH,
        "round": int(round_num),
        "signature": signature_hex.strip(),
        "selection": SELECTION,
        "j": int(j),
        "m": int(m),
        "k": int(k),
        "previous_outcome": previous_ref
    }

    checker_info = {"key_fp": key_fp, "public_key": pk_hex}
    stmt = make_statement(release, challenge, results, checker_info, extra)
    return seal(key, STATEMENT_TYPE, stmt)


def load_rounds(rounds_path):
    """Load a verified {round: signature_hex} mapping from a JSON file."""
    p = Path(rounds_path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Invalid(f"cannot read rounds file {rounds_path}: {e}") from None
    if isinstance(data, dict) and isinstance(data.get("rounds"), dict):
        data = data["rounds"]
    if not isinstance(data, dict):
        raise Invalid(f"rounds file {rounds_path} must be a JSON object")
    out = {}
    for k, v in data.items():
        sig = v.get("signature") if isinstance(v, dict) else v
        if isinstance(sig, str):
            out[str(k)] = sig.strip()
    return out


def verify_receipt(envelope, anchor_path, rounds=None, allow_unverified_beacon=False):
    """Verify a check receipt envelope against an anchor file. Fails closed on any defect.

    When `rounds` ({str(round): signature_hex}) is supplied and matches the receipt's round
    signature, returns verdict `VALID` with `beacon_verified=True`. Otherwise raises `Invalid`
    unless `allow_unverified_beacon=True`, in which case it returns `WELL_FORMED` with
    `beacon_verified=False` (proving internal consistency and signer identity only).
    """
    claimed = unverified_payload(envelope)
    p = claimed.get("predicate", {}) if isinstance(claimed, dict) else {}
    checker = p.get("checker", {}) if isinstance(p, dict) else {}
    pk_hex = checker.get("public_key")
    if not isinstance(pk_hex, str) or not re.fullmatch(r"[0-9a-f]{64}", pk_hex):
        raise Invalid("receipt checker has no valid 32-byte public_key hex")

    st = open_envelope(envelope, STATEMENT_TYPE, pk_hex)

    if st.get("_type") != "https://in-toto.io/Statement/v1":
        raise Invalid(f"statement _type is {st.get('_type')!r}, expected in-toto Statement/v1")
    if st.get("predicateType") != PREDICATE_TYPE:
        raise Invalid(f"predicateType is {st.get('predicateType')!r}, expected {PREDICATE_TYPE!r}")

    pred = st.get("predicate", {})
    if not isinstance(pred, dict) or pred.get("schema") != RECEIPT_SCHEMA:
        raise Invalid("predicate schema is not ovllm.check-receipt/v0")

    release = release_of(anchor_path)
    if pred.get("release") != release:
        raise Invalid("receipt release metadata does not match the anchor file")

    subjects = st.get("subject", [])
    expected_subjects = [
        {"name": "anchor.json", "digest": {"sha256": release["anchor_sha256"]}},
        {"name": "chain-head", "digest": {"sha256": release["chain_head_sha256"]}}
    ]
    if subjects != expected_subjects:
        raise Invalid(f"receipt subjects {subjects} do not match release digests {expected_subjects}")

    key_fp = fingerprint(pk_hex)
    if pred.get("checker", {}).get("key_fp") != key_fp:
        raise Invalid("checker key_fp does not match public key fingerprint")

    ch = pred.get("challenge", {})
    if not isinstance(ch, dict):
        raise Invalid("challenge is not a JSON object")

    for field in ("round", "signature", "selection", "j", "m", "k"):
        if field not in ch:
            raise Invalid(f"challenge is missing required field {field!r}")

    if ch.get("selection") != SELECTION:
        raise Invalid(f"selection algorithm is {ch.get('selection')!r}, expected {SELECTION!r}")
    if ch.get("beacon_chain_hash") != BEACON_CHAIN_HASH:
        raise Invalid(f"beacon_chain_hash is {ch.get('beacon_chain_hash')!r}, expected {BEACON_CHAIN_HASH!r}")

    n = release["chain_entries"] - 1
    j, m, k, rnd = ch.get("j"), ch.get("m"), ch.get("k"), ch.get("round")
    if not (type(j) is int and type(m) is int and type(k) is int and type(rnd) is int):
        raise Invalid("challenge round, j, m, k must be integers")
    if rnd < 1:
        raise Invalid(f"challenge round {rnd!r} must be >= 1")
    if not (1 <= k <= n and 1 <= m <= MAX_M and 1 <= j <= m):
        raise Invalid(f"challenge parameters out of range: k={k} (1..{n}), m={m} (1..{MAX_M}), j={j} (1..{m})")

    rand_bytes = parse_randomness(ch["signature"])
    beacon_verified = False
    if rounds is not None and str(rnd) in rounds:
        if rounds[str(rnd)] != ch["signature"]:
            raise Invalid(f"drand signature for round {rnd} does not match verified beacon")
        beacon_verified = True
    elif not allow_unverified_beacon:
        raise Invalid(f"beacon round {rnd} is not verified (supply --rounds or --allow-unverified-beacon)")

    expected_segments = select(seed_for(rand_bytes, release["chain_head_sha256"], key_fp, j), n, k)

    res = pred.get("results")
    if not isinstance(res, list) or len(res) != k:
        raise Invalid(f"receipt must report exactly k={k} results, got {len(res) if isinstance(res, list) else res}")

    reported_segments = []
    for r in res:
        if not isinstance(r, dict) or "segment" not in r or "outcome" not in r:
            raise Invalid("malformed check result entry")
        seg = r["segment"]
        if not (isinstance(seg, list) and len(seg) == 2 and seg[1] == seg[0] + 1):
            raise Invalid(f"segment {seg} is not a valid [s, s+1] interval")
        reported_segments.append(seg[0])
        outcome = r["outcome"]
        if outcome not in OUTCOMES:
            raise Invalid(f"unknown outcome {outcome!r}, expected one of {OUTCOMES}")

    if sorted(reported_segments) != expected_segments:
        raise Invalid(f"reported segments {reported_segments} do not match derived segments {expected_segments} for round {ch['round']}")

    return {
        "verdict": "VALID" if beacon_verified else "WELL_FORMED",
        "beacon_verified": beacon_verified,
        "envelope_id": envelope_id(envelope),
        "checker_key_fp": key_fp,
        "round": ch["round"],
        "j": j,
        "m": m,
        "segments": reported_segments,
        "results": res
    }


# ---------- CLI ----------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # keygen
    s = sub.add_parser("keygen", help="Generate an Ed25519 keypair")
    s.add_argument("--out", required=True, help="Path to write raw 32-byte private key seed")
    s.add_argument("--allow-in-repo", action="store_true", help="Allow writing private key inside a git work tree")

    # derive
    s = sub.add_parser("derive", help="Derive assigned segment(s) for a given drand round")
    s.add_argument("--anchor", required=True, help="Release anchor.json")
    s.add_argument("--key-fp", required=True, help="Checker key fingerprint (SHA-256 of raw public key)")
    s.add_argument("--round", required=True, type=int, help="drand round number")
    s.add_argument("--signature", required=True, help="drand BLS signature hex or randomness hex")
    s.add_argument("--j", type=int, default=1, help="Check index (1..m)")
    s.add_argument("--k", type=int, default=1, help="Segments per check")
    s.add_argument("--quiet", action="store_true", help="Print only space-separated segment numbers")

    # sign
    s = sub.add_parser("sign", help="Sign an outcome into a DSSE check receipt")
    s.add_argument("--key", required=True, help="Path to private key")
    s.add_argument("--anchor", required=True, help="Release anchor.json")
    s.add_argument("--round", required=True, type=int, help="drand round number")
    s.add_argument("--signature", required=True, help="drand signature hex or randomness hex")
    s.add_argument("--segment", type=int, help="Checked segment start (optional; derived if omitted)")
    s.add_argument("--outcome", choices=OUTCOMES, default=None, help="Check outcome (default: MATCH, or from --report)")
    s.add_argument("--report", help="Path to local-verifications.jsonl line to extract outcome & stack")
    s.add_argument("--j", type=int, default=1, help="Check index")
    s.add_argument("--m", type=int, default=1, help="Total planned checks")
    s.add_argument("--k", type=int, default=1, help="Segments per check")
    s.add_argument("--code-commit", default="", help="Git commit of checker code")
    s.add_argument("--out", help="Write receipt JSON to file (default: stdout)")

    # verify
    s = sub.add_parser("verify", help="Verify a signed check receipt")
    s.add_argument("receipt", help="Path to check receipt JSON")
    s.add_argument("--anchor", required=True, help="Release anchor.json")
    s.add_argument("--rounds", help="JSON {round: signature} of verified drand rounds")
    s.add_argument("--allow-unverified-beacon", action="store_true",
                   help="Allow verification without --rounds (reports WELL_FORMED instead of VALID)")

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.cmd == "keygen":
            info = keygen(args.out, args.allow_in_repo)
            print(json.dumps(info, indent=2))
            return 0

        if args.cmd == "derive":
            rel = release_of(args.anchor)
            n = rel["chain_entries"] - 1
            rand_bytes = parse_randomness(args.signature)
            segs = select(seed_for(rand_bytes, rel["chain_head_sha256"], args.key_fp, args.j), n, args.k)
            if args.quiet:
                print(" ".join(map(str, segs)))
            else:
                print(json.dumps({"j": args.j, "round": args.round, "segments": segs}, indent=2))
            return 0

        if args.cmd == "sign":
            key = load_key(args.key)
            env = sign_receipt(
                key=key,
                anchor_path=args.anchor,
                round_num=args.round,
                signature_hex=args.signature,
                segments=args.segment,
                outcome=args.outcome,
                j=args.j,
                m=args.m,
                k=args.k,
                code_commit=args.code_commit,
                report_path=args.report
            )
            data = json.dumps(env, indent=2)
            if args.out:
                Path(args.out).write_text(data, encoding="utf-8")
            else:
                print(data)
            return 0

        if args.cmd == "verify":
            p = Path(args.receipt)
            if not p.exists():
                raise Invalid(f"receipt file {args.receipt} not found")
            try:
                env = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise Invalid(f"cannot read receipt {args.receipt}: {e}") from None
            rounds = load_rounds(args.rounds) if args.rounds else None
            res = verify_receipt(env, args.anchor, rounds=rounds,
                                 allow_unverified_beacon=args.allow_unverified_beacon)
            print(json.dumps(res, indent=2))
            return 0

    except Invalid as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
