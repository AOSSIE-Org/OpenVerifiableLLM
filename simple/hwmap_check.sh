#!/usr/bin/env bash
# Hardware reproducibility probe: recompute one short published segment (default 32 -> 33,
# the first conversation epoch, 147 updates) from the public release alone, and report MATCH or
# MISMATCH together with the GPU, driver and software. Only the small conversation stream, the
# record, the code and one checkpoint are downloaded.
#   bash hwmap_check.sh            # SEGMENT=32
set -euo pipefail
WORK=${WORK:-/workspace/hwmap}
SEGMENT=${SEGMENT:-32}
REPO=AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence
REV=${REV:-1dc7fb2a430d24c06c970309d332c4891916548c}
PREP=prepared/8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68
mkdir -p "$WORK" && cd "$WORK"
# Hash-pinned install on top of the image (which has torch and numpy). The lock is fetched from
# GitHub and must have this exact SHA-256 (requirements/pod.lock; pod.in lists the packages).
POD_LOCK_SHA256=026ab3c377855e32bdfa81b48a561f67cb77afa81fa6bfef4211ec4ca6a01f2c
curl -fsSL -o pod.lock https://raw.githubusercontent.com/AOSSIE-Org/OpenVerifiableLLM/main/requirements/pod.lock
echo "$POD_LOCK_SHA256  pod.lock" | sha256sum -c --quiet - || { echo "pod.lock does not match the pinned hash" >&2; exit 1; }
pip install --break-system-packages -q --require-hashes -r pod.lock
python3 - "$REPO" "$REV" "$PREP" "$SEGMENT" <<'PY'
import sys
from huggingface_hub import hf_hub_download, snapshot_download
repo, rev, prep, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
snapshot_download(repo, repo_type="dataset", revision=rev, token=False, local_dir=".",
                  allow_patterns=["release/simple-run/code/**", "release/simple-run/chain.jsonl",
                                  "release/simple-run/recipe.json", f"release/simple-run/ckpt-{k:05d}.pt",
                                  f"{prep}/wikipedia/stream.json", f"{prep}/conversation/*"])
PY
R=release/simple-run
echo "probe: $(python3 -c 'import torch,platform;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name() if torch.cuda.is_available() else platform.processor() or "cpu")') | driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo none)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
python3 $R/code/simple/ovl.py verify --data "$PREP" --run "$R" --segment "$SEGMENT"
