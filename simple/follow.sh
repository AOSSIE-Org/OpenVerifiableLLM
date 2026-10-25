#!/usr/bin/env bash
# Run on a second GPU pod while training runs on the first. It keeps an off-pod copy of the
# run (every checkpoint and the chain) and recomputes each segment as soon as its closing
# boundary exists, so by the end of training every segment has been checked on a different
# machine. Use the same GPU type and image as training.
#   TRAIN=root@HOST TRAIN_PORT=PORT bash simple/follow.sh
# Needs SSH access to the training pod with the default SSH key. Re-running resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/workspace}
PREP_ID=8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68
PREP=$WORK/hf/prepared/$PREP_ID
MIRROR=$WORK/mirror
REMOTE_RUN=${REMOTE_RUN:-/workspace/run}
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false
source simple/fetch.sh
command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync; } >/dev/null
pip install --break-system-packages -q numpy rfc8785==0.1.4 tokenizers==0.23.2 mwparserfromhell==0.7.2 "safetensors>=0.4.5,<1.0" "pynacl>=1.5,<2.0"
fetch $(for f in $PREP_FILES; do echo "prepared/$PREP_ID/$f"; done)
check_prep "$PREP"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$WORK/follower-gpu.txt"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -p ${TRAIN_PORT:-22}"
mkdir -p "$MIRROR"
touch "$WORK/segments.txt"   # one line per checked segment: k MATCH|MISMATCH

while true; do
  # Checkpoints are renamed into place before their chain entry is written, so a listed
  # ckpt-*.pt is complete. Copy the chain to a temporary file and keep only whole lines.
  rsync -a --ignore-existing -e "$SSH" --include 'ckpt-[0-9]*.pt' --include 'recipe.json' \
    --include '*-model.pt' --exclude '*' "$TRAIN:$REMOTE_RUN/" "$MIRROR/" || { sleep 60; continue; }
  if $SSH "${TRAIN}" cat "$REMOTE_RUN/chain.jsonl" > "$MIRROR/chain.tmp" 2>/dev/null; then
    python3 - "$MIRROR" <<'PY'
import json, os, sys
d = sys.argv[1]
good = []
for line in open(f"{d}/chain.tmp"):
    try:
        json.loads(line); good.append(line if line.endswith("\n") else line + "\n")
    except ValueError:
        break
open(f"{d}/chain.part", "w").writelines(good)
os.replace(f"{d}/chain.part", f"{d}/chain.jsonl")
PY
  fi
  [ -f "$MIRROR/chain.jsonl" ] || { sleep 60; continue; }
  n=$(wc -l < "$MIRROR/chain.jsonl")
  for ((k = 0; k < n - 1; k++)); do
    grep -q "^$k " "$WORK/segments.txt" && continue
    [ -f "$MIRROR/$(printf 'ckpt-%05d.pt' "$k")" ] || continue
    # Record only the verifier's verdict; a crash (missing package, OOM) is retried next round.
    out=$(python simple/ovl.py verify --data "$PREP" --run "$MIRROR" --segment "$k" 2>&1 | tee -a "$WORK/follow.log" || true)
    if grep -q "^segment $k->$((k + 1)) .*: MATCH" <<<"$out"; then
      echo "$k MATCH" >> "$WORK/segments.txt"
    elif grep -q "^segment $k->$((k + 1)) .*: MISMATCH" <<<"$out"; then
      echo "$k MISMATCH" >> "$WORK/segments.txt"
    else
      echo "segment $k: verifier error, retrying later" | tee -a "$WORK/follow.log"
    fi
  done
  if tail -n 1 "$MIRROR/chain.jsonl" | grep -q '"kind": "final"' && [ "$(wc -l < "$WORK/segments.txt")" -ge $((n - 1)) ]; then
    echo "all $((n - 1)) segments checked: $(grep -c " MATCH$" "$WORK/segments.txt") MATCH, $(grep -c " MISMATCH$" "$WORK/segments.txt") MISMATCH"
    break
  fi
  sleep 120
done
