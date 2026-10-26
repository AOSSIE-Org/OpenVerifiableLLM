#!/usr/bin/env bash
# One-click independent check of the published run, for a fresh RTX 5090 pod
# (image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, at least 60 GB container disk).
# It downloads the release's own code from Hugging Face at a pinned revision, runs
# verify_segment.sh on random segments (K, default 1), and prints one result line to paste.
# Everything checked (record, checkpoints, data) comes from that same revision, and the
# result is PASS only if the record matches the release's timestamped anchor.
#   curl -fsSL https://raw.githubusercontent.com/AOSSIE-Org/OpenVerifiableLLM/<commit>/simple/checker_pod.sh | bash
set -uo pipefail
export WORK=${WORK:-/workspace} PIP_BREAK_SYSTEM_PACKAGES=1
REPO=AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence
REV=${REV:-96dc282a9803616c6b30bb31bfc3a0b91a7191e6}
RUN_PATH=${RUN_PATH:-release/simple-run}
GPU=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)
echo "GPU: $GPU"
case "$GPU" in *"RTX 5090"*) ;; *) echo "WARNING: the run was trained on an RTX 5090; other GPU types are expected to MISMATCH";; esac
mkdir -p "$WORK" && cd "$WORK"
pip install -q huggingface_hub 2>&1 | grep -vi warning
python3 - "$REPO" "$REV" "$RUN_PATH" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, path = sys.argv[1:]
snapshot_download(repo, repo_type="dataset", revision=rev, token=False, local_dir="pub",
                  allow_patterns=[f"{path}/code/**", "release/simple-run/code/**",
                                  f"{path}/anchor.json", f"{path}/anchor.*.tsr"])
PY
CODE=pub/$RUN_PATH; [ -d "$CODE/code" ] || CODE=pub/release/simple-run
# The release's fetch.sh downloads from the moving `main` branch. Point it at the pinned
# revision, so the record, checkpoints and data checked are the ones this line attests.
sed -i "s|/resolve/main/|/resolve/$REV/|" "$CODE/code/simple/fetch.sh"
grep -q "/resolve/$REV/" "$CODE/code/simple/fetch.sh" || { echo "OVL check FAIL: could not pin downloads to $REV"; exit 1; }
cd "$CODE"
RUN_PATH=$RUN_PATH bash code/simple/verify_segment.sh 2>&1 | tee "$WORK/verify.log"
python3 - "$WORK/hf/$RUN_PATH" "$WORK/pub/$RUN_PATH" "$GPU" "$REV" <<'PY'
import hashlib, json, os, shutil, subprocess, sys, time
run, pub, gpu, rev = sys.argv[1:]
try:
    r = json.loads(open(f"{run}/verifications.jsonl").read().splitlines()[-1])
except (OSError, IndexError):
    sys.exit("\nno result recorded; see the log above")

# The checked record must be the one the release anchored and timestamped.
chain = [json.loads(l) for l in open(f"{run}/chain.jsonl").read().splitlines()]
head = hashlib.sha256(json.dumps(chain[-1], sort_keys=True).encode()).hexdigest()
try:
    anchor = json.load(open(f"{pub}/anchor.json"))
    anchored = (len(chain), head) == (anchor["chain_entries"], anchor["chain_head_sha256"])
except (OSError, KeyError, ValueError):
    anchored = False
print(f"\nanchor: chain head {head[:16]}… {'MATCH' if anchored else 'MISMATCH or missing'} ({len(chain)} entries)")

# The anchor's DigiCert timestamp (RFC 3161), against the system CA store.
ca = next((p for p in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt") if os.path.exists(p)), None)
tsr = f"{pub}/anchor.digicert.tsr"
if shutil.which("openssl") and ca and os.path.exists(tsr):
    v = subprocess.run(["openssl", "ts", "-verify", "-in", tsr, "-data", f"{pub}/anchor.json", "-CAfile", ca],
                       capture_output=True, text=True)
    stamped = "Verification: OK" in v.stdout
    stamp = "timestamp OK" if stamped else "timestamp FAILED"
else:
    stamped, stamp = True, "timestamp not checked (no openssl or CA bundle)"
print(f"anchor: DigiCert {stamp}")

segs = [c for c in r["checks"] if "segment" in c]
result = "PASS" if r["result"] == "PASS" and segs and anchored and stamped else "FAIL"
print("\n===== paste this line =====")
print(f"OVL check {result}: " + ", ".join(f"segment {c['segment'][0]} {'MATCH' if c['match'] else 'MISMATCH'}" for c in segs)
      + f" | anchor {'MATCH' if anchored else 'MISMATCH'}, {stamp} | {gpu} | torch {r['environment']['torch']}"
      + f" | final state {r['chain_head'][:12]} | release {rev[:8]} | {time.strftime('%Y-%m-%d', time.gmtime())}")
print("============================\nDone. Terminate this pod now so it stops billing.")
PY
