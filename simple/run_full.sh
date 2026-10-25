#!/usr/bin/env bash
# Whole production loop on one GPU pod. Run from a checkout of this repository:
#   bash simple/run_full.sh            # bench, train, spot-check segments, chat sample
# Re-running it resumes wherever it stopped. Outputs go to $WORK (default /workspace).
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/workspace}
PREP_ID=8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68
PREP=$WORK/hf/prepared/$PREP_ID
RUN=$WORK/run
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
source simple/fetch.sh

pip install --break-system-packages -q numpy rfc8785==0.1.4 tokenizers==0.23.2 mwparserfromhell==0.7.2 "safetensors>=0.4.5,<1.0" "pynacl>=1.5,<2.0"

# 1. Public prepared inputs (G01/G02), checked against the hashes in each stream.json.
if [ ! -f "$PREP/.verified" ]; then
  fetch $(for f in $PREP_FILES; do echo "prepared/$PREP_ID/$f"; done)
fi
check_prep "$PREP"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$WORK/gpu.txt"

# 2. Throughput (10 minutes): ms/update, projected hours, minutes per segment.
[ -s "$WORK/bench.json" ] || { python simple/ovl.py bench --data "$PREP" --seconds 600 > "$WORK/bench.tmp" && mv "$WORK/bench.tmp" "$WORK/bench.json"; cat "$WORK/bench.json"; }

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
