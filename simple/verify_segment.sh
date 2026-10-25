#!/usr/bin/env bash
# Check part of a published OpenVerifiableLLM training run on your own GPU (same GPU type
# and software as recorded in chain.jsonl entry 0). Each run picks fresh random segments,
# so many independent checkers together cover the whole run.
#   bash simple/verify_segment.sh                 # 1 random segment (~10-20 min)
#   K=3 bash simple/verify_segment.sh             # 3 random segments
#   SEGMENTS="7 8" bash simple/verify_segment.sh  # chosen segments
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/workspace}
PREP_ID=8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68
RUN_PATH=${RUN_PATH:-release/simple-run}   # where the run is published in $REPO
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
source simple/fetch.sh
pip install --break-system-packages -q numpy rfc8785==0.1.4 tokenizers==0.23.2 mwparserfromhell==0.7.2 "safetensors>=0.4.5,<1.0" "pynacl>=1.5,<2.0"
fetch $(for f in $PREP_FILES; do echo "prepared/$PREP_ID/$f"; done) "$RUN_PATH/chain.jsonl" "$RUN_PATH/recipe.json"
check_prep "$WORK/hf/prepared/$PREP_ID"
RUN=$WORK/hf/$RUN_PATH
if [ -n "${SEGMENTS:-}" ]; then PICK=$SEGMENTS; else
  PICK=$(python -c "import json,secrets,sys; n=sum(1 for _ in open(sys.argv[1]))-1; k=min(int(sys.argv[2]),n); s=list(range(n)); print(*sorted(s.pop(secrets.randbelow(len(s))) for _ in range(k)))" "$RUN/chain.jsonl" "${K:-1}")
fi
# Only the checkpoints at the start of the chosen segments are needed.
fetch $(for k in $PICK; do printf "%s/ckpt-%05d.pt " "$RUN_PATH" "$k"; done)
python simple/ovl.py verify --data "$WORK/hf/prepared/$PREP_ID" --run "$RUN" --segment $PICK
