#!/usr/bin/env python3
"""Download pinned evidence files from HF and write data/raw/* + manifest (sha256 per file)."""
import hashlib, json, os, sys, urllib.request
REPO = "AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence"
REV = "96dc282a9803616c6b30bb31bfc3a0b91a7191e6"  # brief pins 96dc282a; full sha resolved from HF API
PREP = "prepared/8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68"
FILES = [f"release/simple-run/{f}" for f in ["chain.jsonl","verifications.jsonl","hardware-map.jsonl",
         "release-hashes.txt","anchor.json","anchor.digicert.tsr","anchor.freetsa.tsr","README.md","recipe.json"]] + \
        [f"release/unlearning-3i-atlas/{f}" for f in ["chain.jsonl","verifications.jsonl","loo-results.json",
         "release-hashes.txt","README.md","recipe.json"]] + [f"{PREP}/preparation.json"]
out = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
manifest = {"repo": REPO, "revision": REV, "files": {}}
for p in FILES:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{p}"
    b = urllib.request.urlopen(url).read()
    dst = os.path.join(out, p); os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "wb").write(b)
    manifest["files"][p] = {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b), "url": url}
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print(len(FILES), "files ok")
