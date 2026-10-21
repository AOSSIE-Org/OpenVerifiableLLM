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


def check(run, chain):
    with tempfile.TemporaryDirectory() as d:
        ca, tsa = (Path(d) / "cacert.pem", Path(d) / "tsa.crt")
        for url, path in zip(FREETSA_CERTS, (ca, tsa)):
            path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
        system = "/etc/ssl/certs/ca-certificates.crt"
        ok = True
        for e in chain:
            for name in TSA:
                path = Path(run) / "timestamps" / f"entry-{e['index']:05d}.{name}.tsr"
                if not path.exists():
                    print(f"entry {e['index']} {name}: missing"); ok = False
                    continue
                trust = ["-CAfile", str(ca), "-untrusted", str(tsa)] if name == "freetsa" else ["-CAfile", system]
                r = subprocess.run(["openssl", "ts", "-verify", "-digest", entry_hash(e), "-in", str(path)] + trust,
                                   capture_output=True, text=True)
                when = subprocess.run(["openssl", "ts", "-reply", "-in", str(path), "-text"],
                                      capture_output=True, text=True).stdout
                when = next((l.split(":", 1)[1].strip() for l in when.splitlines() if l.startswith("Time stamp:")), "?")
                good = "Verification: OK" in r.stdout
                ok &= good
                print(f"entry {e['index']} {name}: {'OK' if good else 'FAIL'} {when}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run"); p.add_argument("--check", action="store_true")
    a = p.parse_args()
    chain = [json.loads(l) for l in (Path(a.run) / "chain.jsonl").read_text().splitlines()]
    if a.check:
        sys.exit(0 if check(a.run, chain) else 1)
    for e in chain:
        stamp_entry(a.run, e)


if __name__ == "__main__":
    main()
