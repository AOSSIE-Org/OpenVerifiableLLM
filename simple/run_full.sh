#!/usr/bin/env bash
# Whole production loop on one GPU pod. Run from a checkout of this repository:
#   bash simple/run_full.sh            # bench, train, spot-check segments, chat sample
# Re-running it resumes wherever it stopped. Outputs go to $WORK (default /workspace).
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/workspace}
REPO=AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence
PREP_ID=8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68
PREP=$WORK/hf/prepared/$PREP_ID
RUN=$WORK/run
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=1

pip install -q "huggingface_hub[cli]" hf_transfer tokenizers numpy pynacl

# 1. Public prepared inputs (G01/G02), checked against the hashes in each stream.json.
if [ ! -f "$PREP/.verified" ]; then
  hf download "$REPO" --repo-type dataset --include "prepared/$PREP_ID/*" --local-dir "$WORK/hf"
  python - "$PREP" <<'EOF'
import hashlib, json, sys
from pathlib import Path
prep = Path(sys.argv[1])
for phase in ("wikipedia", "conversation"):
    for f in json.loads((prep / phase / "stream.json").read_text())["files"]:
        h = hashlib.sha256()
        with open(prep / phase / f["path"], "rb") as fh:
            while chunk := fh.read(1 << 24):
                h.update(chunk)
        assert h.hexdigest() == f["sha256"], f"hash mismatch: {phase}/{f['path']}"
        print("ok", phase, f["path"])
EOF
  touch "$PREP/.verified"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$WORK/gpu.txt"

# 2. Throughput (10 minutes): ms/update, projected hours, minutes per segment.
[ -f "$WORK/bench.json" ] || python simple/ovl.py bench --data "$PREP" --seconds 600 | tee "$WORK/bench.json"

# 3. Full training: one Wikipedia pass, then the conversation epochs. Resumes automatically.
#    Every boundary checkpoint is kept; each one starts an independently checkable segment.
[ -f "$RUN/conversation-model.pt" ] || python simple/ovl.py train --data "$PREP" --out "$RUN" 2>&1 | tee -a "$WORK/train.log"

# 4. Quick checks on this pod: seeded init, final weights, every conversation segment
#    (minutes) and one random Wikipedia segment. Others verify more segments in parallel
#    with simple/verify_segment.sh; nobody has to replay the whole run.
CONV=$(python -c "import json,sys; c=[json.loads(l) for l in open(sys.argv[1])]; b=[e['index'] for e in c if e['kind']=='base'][0]; print(*range(b, len(c)-1))" "$RUN/chain.jsonl")
python simple/ovl.py verify --data "$PREP" --run "$RUN" --weights "$RUN/conversation-model.pt" \
  --segment $CONV --random 1 2>&1 | tee -a "$WORK/verify.log"

# 5. Does it talk?
for q in "What is Wikipedia?" "Who was Albert Einstein?" "Explain photosynthesis."; do
  echo "Q: $q"; python simple/ovl.py chat --run "$RUN" --tokenizer "$PREP/tokenizer/tokenizer.json" "$q"; echo
done | tee "$WORK/samples.txt"
sha256sum "$RUN"/*-model.pt "$RUN"/chain.jsonl "$RUN"/recipe.json "$RUN"/ckpt-*.pt | tee "$WORK/release-hashes.txt"
echo "done: $RUN (publish chain.jsonl, recipe.json, models and every ckpt-*.pt)"
