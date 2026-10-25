# Images

The figures used in the [README](../../README.md).

## `openverifiablellm_logo.png`

The project logo: a speech bubble (a language model) with a check mark (verified). Green
`#00853f`, yellow `#ffcc00`, transparent background, 512 × 512.

## `check_snippet.svg`

A terminal session checking the released model. Every output line is copied from a real run:
the weights check is from `simple/ovl.py verify` on the published files, and the segment check
is from the first one-click check on a fresh RTX 5090 (21 Oct 2026). If the output format of
`simple/ovl.py` changes, update the SVG text to match.

```console
# Any machine: the published weights are the committed ones
$ sha256sum -c release-hashes.txt
conversation-model.pt: OK   … every file OK
$ python simple/ovl.py verify --run . \
      --weights conversation-model.pt
weights conversation-model.pt: MATCH chain entry 37
verify PASS: 1 checks; 0 of 37 segments

# RTX 5090: recompute one training segment, bit for bit
$ K=1 bash simple/verify_segment.sh
initialization: MATCH
segment 8->9 (progress): MATCH in 1139 s
verify PASS: 2 checks; 1 of 37 segments
```
