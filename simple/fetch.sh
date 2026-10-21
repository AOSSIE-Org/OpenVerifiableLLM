# Sourced by run_full.sh and verify_segment.sh. fetch PATH... downloads files from the public
# dataset repo into $WORK/hf/PATH with 16 parallel ranges per file (a single HF connection
# stalls at a few MB/s on the 7-14 GB streams). Files already complete are skipped.
REPO=AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence
command -v aria2c >/dev/null || { apt-get update -qq && apt-get install -y -qq aria2; } >/dev/null
fetch() {
  local f
  for f in "$@"; do
    if [ -f "$WORK/hf/$f" ] && [ ! -f "$WORK/hf/$f.aria2" ]; then continue; fi
    aria2c -q -x16 -s16 -k64M --max-tries=0 --retry-wait=3 --file-allocation=none \
      -d "$WORK/hf/$(dirname "$f")" -o "$(basename "$f")" \
      "https://huggingface.co/datasets/$REPO/resolve/main/$f"
  done
}
PREP_FILES="preparation.json tokenizer/tokenizer.json tokenizer/tokenizer-manifest.json"
for phase in wikipedia conversation; do
  PREP_FILES="$PREP_FILES $phase/stream.json $phase/documents.jsonl $phase/tokens.u16 $phase/mask.u8"
done
