"""RFC 3161 timestamps for chain entries, taken while training runs.

Each boundary's entry hash (the value the next entry stores as `previous`, so it also commits to
every earlier entry) is sent to two independent timestamp authorities. The replies are saved as
RUN/timestamps/entry-XXXXX.<tsa>.tsr and prove the entry existed no later than the stamped time.
Stamping runs in a background thread and never touches the training state.

  python simple/stamp.py RUN            # stamp entries that have no timestamp yet
  python simple/stamp.py RUN --check    # verify every saved timestamp with openssl
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

TSA = {"freetsa": "https://freetsa.org/tsr", "digicert": "http://timestamp.digicert.com"}
FREETSA_CERTS = ("https://freetsa.org/files/cacert.pem", "https://freetsa.org/files/tsa.crt")
SHA256_OID = bytes.fromhex("0609608648016503040201")


def entry_hash(entry):
    return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()


def query(digest):
    """DER TimeStampReq: version 1, SHA-256 imprint, random nonce, certReq true."""
    imprint = b"\x30\x31\x30\x0d" + SHA256_OID + b"\x05\x00\x04\x20" + digest
    nonce = bytes([secrets.randbits(7) | 1]) + secrets.token_bytes(7)
    body = b"\x02\x01\x01" + imprint + b"\x02\x08" + nonce + b"\x01\x01\xff"
    return b"\x30" + bytes([len(body)]) + body


def granted(reply):
    """PKIStatus of a TimeStampResp is 0 (granted) or 1 (granted with modifications)."""
    i = 2 + (reply[1] & 0x7f if reply[1] & 0x80 else 0)  # skip the outer SEQUENCE header
    return reply[i] == 0x30 and reply[i + 2:i + 4] == b"\x02\x01" and reply[i + 4] in (0, 1)


def stamp_entry(run, entry):
    out = Path(run) / "timestamps"
    out.mkdir(exist_ok=True)
    digest = bytes.fromhex(entry_hash(entry))
    for name, url in TSA.items():
        path = out / f"entry-{entry['index']:05d}.{name}.tsr"
        if path.exists():
            continue
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, query(digest), {"Content-Type": "application/timestamp-query"})
                reply = urllib.request.urlopen(req, timeout=30).read()
                if not granted(reply):
                    raise ValueError("request not granted")
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(reply)
                tmp.replace(path)
                break
            except Exception as e:  # a missing stamp is filled in later by `stamp.py RUN`
                if attempt == 2:
                    print(f"timestamp {name} entry {entry['index']} failed: {e}", flush=True)
                else:
                    time.sleep(10 * (attempt + 1))


def stamp_async(run, entry):
    t = threading.Thread(target=stamp_entry, args=(run, dict(entry)), daemon=True)
    t.start()
    return t


def system_ca():
    """The system CA bundle that holds DigiCert's root: Linux, then macOS."""
    return next((p for p in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                             "/etc/ssl/cert.pem") if Path(p).exists()), None)


def freetsa_certs(d):
    ca, tsa = (Path(d) / "cacert.pem", Path(d) / "tsa.crt")
    for url, path in zip(FREETSA_CERTS, (ca, tsa), strict=True):
        path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
    return ca, tsa


def openssl():
    """The openssl to use. OVL_OPENSSL lets macOS users point at Homebrew's openssl@3,
    since /usr/bin/openssl is LibreSSL and cannot build DigiCert's chain (#364)."""
    return os.environ.get("OVL_OPENSSL", "openssl")


def is_libressl():
    """True when the openssl in use is LibreSSL, for an actionable error message."""
    try:
        v = subprocess.run([openssl(), "version"], capture_output=True, text=True)
    except OSError:
        return False
    return "libressl" in v.stdout.lower()


def ts_verify(path, name, imprint, certs):
    """The `openssl ts -verify` call, shared by the verdict and the diagnostic."""
    ca, tsa = certs
    trust = ["-CAfile", str(ca), "-untrusted", str(tsa)] if name == "freetsa" else ["-CAfile", str(system_ca())]
    return subprocess.run([openssl(), "ts", "-verify", *imprint, "-in", str(path)] + trust,
                          capture_output=True, text=True)


def verify_reply(path, name, imprint, certs):
    """Check one saved reply with openssl. imprint is ["-digest", hex] or ["-data", file]; returns (ok, time).

    A nonzero exit is a failure even if stdout happens to contain 'Verification: OK'; fail closed.
    """
    r = ts_verify(path, name, imprint, certs)
    when = subprocess.run([openssl(), "ts", "-reply", "-in", str(path), "-text"], capture_output=True, text=True).stdout
    when = next((l.split(":", 1)[1].strip() for l in when.splitlines() if l.startswith("Time stamp:")), "?")
    return r.returncode == 0 and "Verification: OK" in r.stdout, when


def verify_failure_detail(path, name, imprint, certs):
    """openssl's own reason for a failed reply check, so a tool error is not misreported as tampering."""
    try:
        r = ts_verify(path, name, imprint, certs)
    except OSError as e:
        return str(e)
    return r.stderr.strip() or r.stdout.strip() or "openssl gave no diagnostic"


def reply_imprint(path):
    """The SHA-256 message imprint stamped into a reply, as hex; None if it cannot be found.

    The imprint is a sha256 AlgorithmIdentifier (OID plus NULL) followed by a 32-byte OCTET
    STRING. Comparing it with the digest of the data under test tells a changed anchor apart
    from a CA or tool failure, without depending on version-specific openssl error text.
    """
    data = Path(path).read_bytes()
    marker = SHA256_OID + b"\x05\x00\x04\x20"
    i = data.find(marker)
    if i < 0:
        return None
    return data[i + len(marker):i + len(marker) + 32].hex()


def check(run, chain):
    with tempfile.TemporaryDirectory() as d:
        certs = freetsa_certs(d)
        ok = True
        for e in chain:
            for name in TSA:
                path = Path(run) / "timestamps" / f"entry-{e['index']:05d}.{name}.tsr"
                if not path.exists():
                    print(f"entry {e['index']} {name}: missing"); ok = False
                    continue
                good, when = verify_reply(path, name, ["-digest", entry_hash(e)], certs)
                ok &= good
                print(f"entry {e['index']} {name}: {'OK' if good else 'FAIL'} {when}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run"); p.add_argument("--check", action="store_true")
    a = p.parse_args()
    chain_file = Path(a.run) / "chain.jsonl"
    if not chain_file.exists():
        sys.exit(f"{chain_file} not found")
    chain = []
    with chain_file.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                chain.append(json.loads(line))
            except json.JSONDecodeError as err:
                sys.exit(f"{chain_file}: line {n} is not valid JSON ({err.msg}); the file is cut off or damaged")
    if not chain:
        sys.exit(f"{chain_file} has no chain entries")
    if a.check:
        sys.exit(0 if check(a.run, chain) else 1)
    for e in chain:
        stamp_entry(a.run, e)


if __name__ == "__main__":
    main()
