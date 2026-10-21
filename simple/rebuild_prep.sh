#!/usr/bin/env bash
# Rebuild the prepared streams from the raw Wikipedia dump and OpenAssistant data, and compare
# the result with the published preparation.json (goal 1: the trained-on streams really come
# from the dump). CPU only, ~100 GB disk. Run from a checkout of this repository:
#   bash simple/rebuild_prep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/workspace}
PREP_REV=3f882ad9d57a0eeacb7c5db288af4242f8de5779   # all 14 pinned preparation files match here
PREP_ID=8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68
RAW_ID=3a7f5106d4b6898fd2c8b10f71d9054ae4daacc9e1cbef248cb5069779525af1
STMT_ID=ec263b5c3914fd1a8bd25e97fa377b4f8416c38a9ef216d27c203252948b9129
source simple/fetch.sh

# Inputs: the raw archive, the signed source statement and the published preparation result.
W=raw/$RAW_ID/wikipedia; C=raw/$RAW_ID/conversation; S=source-commitments/$STMT_ID
fetch $W/dumpstatus-receipt.json $W/dumpstatus.headers.txt $W/dumpstatus.json \
  $W/enwiki-20260901-index.html $W/enwiki-20260901-md5sums.txt $W/enwiki-20260901-sha1sums.txt \
  $W/enwiki-20260901-pages-articles.xml.bz2 \
  $W/enwiki-20260901-pages-articles.xml.bz2.receipt-c95ef29b0e5d4c5b98faaa4cf2f5a82d.json \
  $W/enwiki-20260901-pages-articles.xml.bz2.source.json $W/enwiki-20260901-pages-articles.xml.bz2.verified.json \
  $W/upstream-metadata-acquisition.json $W/upstream-metadata-response-receipts.json \
  $C/LICENSE $C/README.md $C/acquisition.json \
  $C/data/train-00000-of-00001-b42a775f407cee45.parquet $C/data/validation-00000-of-00001-134b8fd0c89408b6.parquet \
  raw/$RAW_ID/LICENSES.md raw/$RAW_ID/README.md \
  $S/statement.json $S/statement.sigstore.json $S/operator-selected-policy.json \
  prepared/$PREP_ID/preparation.json

# The pinned code and the exact locked environment it was prepared with.
SRC=$WORK/prep-src
[ -d "$SRC" ] || git clone -q https://github.com/AOSSIE-Org/OpenVerifiableLLM "$SRC"
git -C "$SRC" checkout -q "$PREP_REV"
pip install --break-system-packages -q -U uv   # 0.9 cannot fetch Python 3.12.14
uv python install 3.12.14
[ -d "$WORK/prep-venv" ] || uv venv -q -p 3.12.14 "$WORK/prep-venv"
VIRTUAL_ENV=$WORK/prep-venv uv pip sync -q --index-strategy unsafe-best-match "$SRC/requirements/preparation.lock"

# Fresh output, full comparison. The policy file documents the operator's choice; a reader
# should check it against the published signing workflow before trusting it.
rm -rf "$WORK/prep-out" "$WORK/prep-out-progress"   # the pinned code requires both fresh
cd "$SRC"
TOKENIZERS_PARALLELISM=false PYTHONPATH=src "$WORK/prep-venv/bin/python" -m ovl_pipeline.preparation \
  --statement "$WORK/hf/$S/statement.json" --bundle "$WORK/hf/$S/statement.sigstore.json" \
  --trust-policy "$WORK/hf/$S/operator-selected-policy.json" \
  --wikipedia-raw "$WORK/hf/$W" --conversation-raw "$WORK/hf/$C" --output "$WORK/prep-out" \
  --compare-preparation "$WORK/hf/prepared/$PREP_ID/preparation.json" | tee "$WORK/rebuild-result.json"
