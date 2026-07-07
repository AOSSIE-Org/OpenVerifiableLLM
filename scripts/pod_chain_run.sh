#!/bin/bash
# RUNBOOK section 8: the mid-scale k-of-N chain evidence run, unattended.
#
# Launched as the pod's start command (after the repo is cloned to
# /workspace/repo). Serves /workspace over HTTP :8000 from the very start so
# progress (out/status.txt, out/train.log) and results (out/audit_report.json,
# out/chain_manifest.json) can be polled through the RunPod proxy without SSH.
#
# Knobs (env): OVL_CHAIN_MODEL, OVL_CHAIN_DATASET, OVL_CHAIN_SEGMENTS,
#              OVL_CHAIN_SEGMENT_STEPS, OVL_CHAIN_K

OUT=/workspace/out
REPO=/workspace/repo
mkdir -p "$OUT"

python -m http.server 8000 --directory /workspace >"$OUT/http.log" 2>&1 &

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/status.txt"; }

log "SETUP starting"
nvidia-smi >"$OUT/nvidia.txt" 2>&1
pip install -r "$REPO/requirements.txt" >"$OUT/pip.log" 2>&1
log "SETUP done rc=$?"

MODEL="${OVL_CHAIN_MODEL:-gpt120m}"
DATASET="${OVL_CHAIN_DATASET:-enwik8}"
SEGMENTS="${OVL_CHAIN_SEGMENTS:-20}"
SEGMENT_STEPS="${OVL_CHAIN_SEGMENT_STEPS:-500}"
K="${OVL_CHAIN_K:-3}"

cd "$REPO/src" || { log "FATAL repo missing"; sleep infinity; }

log "TRAIN starting: $MODEL/$DATASET ${SEGMENTS}x${SEGMENT_STEPS}"
python chain.py train --model "$MODEL" --dataset "$DATASET" \
    --segments "$SEGMENTS" --segment-steps "$SEGMENT_STEPS" --device cuda \
    >"$OUT/train.log" 2>&1
log "TRAIN done rc=$?"

log "AUDIT starting: k=$K"
python - "$K" >"$OUT/audit.log" 2>&1 <<'PY'
import json
import sys

from chain import audit_chain

report = audit_chain(k=int(sys.argv[1]), audit_seed=7, device="cuda")
with open("/workspace/out/audit_report.json", "w") as f:
    json.dump(report, f, indent=2)
sys.exit(0 if report["ok"] else 1)
PY
log "AUDIT done rc=$?"

du -sb "$REPO/runs/chain" >"$OUT/chain_size.txt" 2>&1
cp "$REPO/runs/chain/chain_manifest.json" "$OUT/" 2>/dev/null

log "ALL_DONE"
# Keep the pod (and the HTTP server) alive so results can be downloaded;
# the controller terminates the pod after retrieval.
sleep infinity
