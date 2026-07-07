#!/bin/bash
# Replay-free forgery-detector experiment (src/forgery.py) at scale, unattended.
# Same conventions as pod_chain_run.sh: repo cloned at /workspace/repo, progress
# and results served over HTTP :8000 (out/status.txt, out/forgery.log,
# out/forgery_report.json).
#
# Knobs (env): OVL_FORGE_MODEL, OVL_FORGE_DATASET, OVL_FORGE_SEGMENTS,
#              OVL_FORGE_SEGMENT_STEPS

OUT=/workspace/out
REPO=/workspace/repo
mkdir -p "$OUT"

python -m http.server 8000 --directory /workspace >"$OUT/http.log" 2>&1 &

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/status.txt"; }

log "SETUP starting"
nvidia-smi >"$OUT/nvidia.txt" 2>&1
TORCH_INDEX="${OVL_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
python3 -m venv /workspace/venv >"$OUT/pip.log" 2>&1 \
    && . /workspace/venv/bin/activate \
    && pip install "torch>=2.10,<3.0" --index-url "$TORCH_INDEX" >>"$OUT/pip.log" 2>&1 \
    && pip install -r "$REPO/requirements.txt" >>"$OUT/pip.log" 2>&1
rc=$?
log "SETUP done rc=$rc"
if [ "$rc" -ne 0 ]; then log "FATAL setup failed"; sleep infinity; fi

MODEL="${OVL_FORGE_MODEL:-gpt120m}"
DATASET="${OVL_FORGE_DATASET:-enwik8}"
SEGMENTS="${OVL_FORGE_SEGMENTS:-5}"
SEGMENT_STEPS="${OVL_FORGE_SEGMENT_STEPS:-200}"

cd "$REPO/src" || { log "FATAL repo missing"; sleep infinity; }

log "FORGERY starting: $MODEL/$DATASET ${SEGMENTS}x${SEGMENT_STEPS}"
python forgery.py --model "$MODEL" --dataset "$DATASET" \
    --segments "$SEGMENTS" --segment-steps "$SEGMENT_STEPS" --device cuda \
    --out "$OUT/forgery_report.json" >"$OUT/forgery.log" 2>&1
rc=$?
log "FORGERY done rc=$rc"

log "ALL_DONE"
sleep infinity
