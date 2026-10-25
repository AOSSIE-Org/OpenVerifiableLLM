#!/usr/bin/env bash
# One-click independent check of the published run, for a fresh RTX 5090 pod
# (image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, at least 60 GB container disk).
# It downloads the release's own code from Hugging Face at a pinned revision, runs
# verify_segment.sh on random segments (K, default 1), and prints one result line to paste.
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
                  allow_patterns=[f"{path}/code/**", "release/simple-run/code/**"])
PY
CODE=pub/$RUN_PATH; [ -d "$CODE/code" ] || CODE=pub/release/simple-run
cd "$CODE"
RUN_PATH=$RUN_PATH bash code/simple/verify_segment.sh 2>&1 | tee "$WORK/verify.log"
python3 - "$WORK/hf/$RUN_PATH" "$GPU" "$REV" <<'PY'
import json, sys, time
run, gpu, rev = sys.argv[1:]
try:
    r = json.loads(open(f"{run}/verifications.jsonl").read().splitlines()[-1])
except (OSError, IndexError):
    sys.exit("\nno result recorded; see the log above")
segs = [c for c in r["checks"] if "segment" in c]
print("\n===== paste this line =====")
print(f"OVL check {r['result']}: " + ", ".join(f"segment {c['segment'][0]} {'MATCH' if c['match'] else 'MISMATCH'}" for c in segs)
      + f" | {gpu} | torch {r['environment']['torch']} | final state {r['chain_head'][:12]} | release {rev[:8]}"
      + f" | {time.strftime('%Y-%m-%d', time.gmtime())}")
print("============================\nDone. Terminate this pod now so it stops billing.")
PY
