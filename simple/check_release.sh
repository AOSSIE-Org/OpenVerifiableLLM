#!/usr/bin/env bash
# The consistency check of the released model in one command, on any computer, no GPU.
#
#   git clone --depth 1 https://github.com/AOSSIE-Org/OpenVerifiableLLM
#   bash OpenVerifiableLLM/simple/check_release.sh            # works in ./ovl-check
#
# It makes a Python environment (CPU-only PyTorch), downloads the release without its
# checkpoints (about 330 MB; the Python environment takes about 1 GB), checks every downloaded file against release-hashes.txt, then runs
# `ovl.py verify`: the record's hash chain, its two timestamps, and the released weights.
# Takes about 2 minutes. Needs Python 3.12 or newer, git, openssl and network access. Set OVL_PYTHON to use an existing
# Python that already has torch, numpy and huggingface_hub, and skip the install.
# It recomputes no training: that needs an RTX 5090 (docs/verify.md).
set -euo pipefail

WORK=${1:-ovl-check}
RUN_PATH=${RUN_PATH:-release/simple-run}
REPO=AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
fail() { echo "OVL release check FAIL: $*" >&2; exit 1; }

command -v openssl >/dev/null || fail "openssl not found; it checks the record's timestamps"
if [ -n "${OVL_PYTHON:-}" ]; then
  PY=$OVL_PYTHON
else
  BASE=$(command -v python3 || command -v python) || fail "Python 3.12 or newer not found"
  "$BASE" -c 'import sys; sys.exit(sys.version_info < (3, 12))' || fail "needs Python 3.12 or newer, found $("$BASE" -V)"
  if [ ! -x "$WORK/venv/bin/python" ]; then
    echo "== Python environment in $WORK/venv"
    "$BASE" -m venv "$WORK/venv"
    "$WORK/venv/bin/python" -m pip install -q --upgrade pip
    # On Linux the default PyTorch download includes CUDA (several GB); this check needs only the CPU.
    [ "$(uname -s)" = Linux ] && TORCH_INDEX=(--index-url https://download.pytorch.org/whl/cpu) || TORCH_INDEX=()
    "$WORK/venv/bin/python" -m pip install -q "${TORCH_INDEX[@]}" torch
    "$WORK/venv/bin/python" -m pip install -q numpy huggingface_hub
  fi
  PY=$WORK/venv/bin/python
fi

echo "== Downloading $RUN_PATH (without checkpoints)"
"$PY" - "$REPO" "$RUN_PATH" "$WORK" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, path, out = sys.argv[1:]
snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{path}/*"], ignore_patterns=[f"{path}/ckpt-*"],
                  local_dir=out)
PY
RUN=$WORK/$RUN_PATH

echo "== Checking downloaded files against release-hashes.txt"
"$PY" - "$RUN" <<'PY' || fail "a downloaded file differs from release-hashes.txt"
import hashlib, sys
from pathlib import Path
run, bad, n = Path(sys.argv[1]), 0, 0
for line in (run / "release-hashes.txt").read_text(encoding="utf-8").splitlines():
    want, name = line.split(maxsplit=1)
    f = run / name.lstrip("*")
    if not f.is_file():
        continue  # not downloaded, like the checkpoints
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    n += 1
    if h.hexdigest() != want:
        print(f"{name}: FAILED"); bad += 1
print(f"{n} files: {'all match' if not bad else f'{bad} differ'}")
sys.exit(1 if bad or not n else 0)
PY

echo "== Checking the record and the released weights"
OVL_DEVICE=cpu "$PY" "$CODE/simple/ovl.py" verify --run "$RUN" --weights "$RUN/conversation-model.pt" \
  || fail "verify did not pass (see above)"
echo "OVL release check PASS: the released model matches its timestamped training record."
