# Version 1 development pipeline formats

The package `src/ovl_pipeline` is under development. Its only runnable end-to-end
profile is `local-synthetic-fixture`. It refuses production admission. A fixture
PASS supplies no G01–G10 production credit, public identity, transparency inclusion,
Wikipedia coverage, GPU portability, public download or independent review claim.
Legacy `ovllm` experiments remain separate and are not this full-verification path.

## Run the implemented fixture

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[pipeline]'
TOKENIZERS_PARALLELISM=false .venv/bin/ovllm-pipeline fixture \
  --source tests/fixtures/pipeline --output runs/fixture-demo \
  --trust-policy runs/fixture-demo-trust.json
TOKENIZERS_PARALLELISM=false .venv/bin/ovllm-pipeline verify-fixture \
  --bundle runs/fixture-demo --trust-policy runs/fixture-demo-trust.json
```

Output directories and trust policies must be new. The command copies synthetic
inputs, extracts them, fits byte-level BPE, prepares document streams, regenerates
initialization, trains both phases, continuously replays all updates, exports base
and chat weights, signs an inventory, verifies a fresh local bundle copy, rebuilds
all data and repeats replay/inference. The fixture uses an ephemeral signing key;
its independently supplied policy file must be retained. This is a local test
trust root, not an authenticated AOSSIE publisher policy.

`verify-fixture` performs actual computation; it does not trust a stored PASS.
Missing or altered evidence returns a nonzero exit. The verified public-production profile and OIDC/transparency admission are not
implemented yet. The separately tested HTTPS acquisition command is described below.

## Canonical encoding

Manifest bytes are UTF-8 RFC 8785 canonical JSON, through the `rfc8785` library.
Allowed values are null, boolean, Unicode string, integers in the exact interoperable
range ±(2^53−1), arrays and string-keyed objects. Floats, duplicate keys, unpaired
surrogates and nonfinite numbers are rejected. Signed manifests must already have
canonical bytes. No Unicode normalization is applied. Structured training-state
scalars use tagged encodings; finite binary64 floats store their 8-byte big-endian
hexadecimal representation, preserving signed zero. Hyperparameters in the recipe
use explicit decimal strings. Tensor floating-point payloads preserve raw bits.

SHA-256 file hashes cover exact file bytes. JSON-object roots cover canonical JSON
bytes. They are different identities from tensor/state roots. Inventories bind
relative POSIX paths, lengths and SHA-256. Reject duplicate paths, traversal,
noncanonical path spellings and symlinks. Caller-owned verification directories
must not be concurrently writable by an attacker; this is not an OS sandbox.

Tensor-root preimage: ASCII `ovl.tensors.v1` followed by NUL, then the unsigned
64-bit big-endian tensor count. Iterate names in Python Unicode lexical order.
For each tensor, append a length-prefixed canonical header and length-prefixed
payload; each prefix is unsigned 64-bit big-endian. Header fields are `name`,
`dtype` (PyTorch spelling), `shape`, and `endian` (`little`). Payload is contiguous
logical C-order bytes on a supported little-endian host, without numeric coercion.
Sparse and quantized tensors are not supported. The root is SHA-256 of this preimage.
This commits dtype, shape, names and bit patterns, not just parameter values.

Merkle leaves are SHA-256(0x00 || record bytes); internal nodes are
SHA-256(0x01 || left || right). Split a nontrivial sequence at its largest strictly
smaller power of two; no odd-leaf duplication. Empty tree is SHA-256(empty).
Final root is SHA-256(`ovl.merkle.v1` || NUL || uint64_be(leaf count) || tree hash).
The accumulator keeps logarithmic memory. Tests compare it with an independently
written recursive construction over empty, odd and power-of-two lengths.
Inclusion-proof generation is not yet implemented for this new format.

## Data and targets

XML source order is explicit, then page order within each file. Require one current
revision per page and unique page/revision IDs. DTDs and entities are forbidden.
Every page has an inclusion or reason-coded exclusion record. Eligible text uses
main-namespace nonredirect wikitext pages; empty revisions/extractions and other
content models are recorded exclusions. `mwparserfromhell` removes ref/references/
gallery tags, then `strip_code(normalize=True, collapse=False)` without online
expansion. This is a specific extracted-text policy, not every XML/markup byte.

The tokenizer trains over an ordered whole-article prefix with a byte target;
a first article larger than that target is included and its actual size recorded.
BPE uses the complete byte alphabet and no special-token matcher. Its text IDs are
shifted by 5, leaving 0 pad, 1 BOS, 2 EOS, 3 user and 4 assistant. Thus literal
marker-like strings cannot become control tokens. Text roundtrips are checked.
The stored tokenizer is the raw BPE; all consumers must apply the declared ID shift.

Streams store uint16 little-endian tokens and uint8 binary loss masks. Each Wikipedia
article contributes all text tokens and EOS, predicted initially from BOS. Windows
never cross documents; subsequent windows prepend the preceding token as context.
Every target occurs once; there is no dropped final window or batch. Chat role
markers and user text are excluded from loss; assistant text and EOS are targets.
P0 accepts synthetic already-selected alternating conversations. The separate
`conversations.py` module now implements OASST tree selection, official split
separation and per-message exclusion ledgers, with synthetic branch/ancestry tests.
Production conversation preparation remains gated on the public source commitment.

Target IDs enumerate loss-bearing positions in document order. A separate validator
checks document counts, offsets, masks, EOS and totals. During updates it checks
actual target IDs form the next contiguous range. Batch transcripts commit inputs,
targets, masks and IDs. Continuous replay and raw reconstruction add checks beyond
these operator-produced transcripts. No list of every corpus target is constructed.

## Numerical and checkpoint scope

The P0 numerical stack is CPU, one thread, deterministic algorithms, FP32,
manual causal attention and tied input/output embeddings. Initialization samples
matrix parameters from normal(0, 0.02), zeros biases, and sets normalization scales
to one. Named unique parameters avoid initializing the tied matrix twice. AdamW
is nonfused, noncapturable and non-foreach; learning rate is constant. Each update
normalizes by its actual loss-bearing target count. No gradient accumulation,
loss scaler, stochastic dropout, custom RNG generator or compilation is used.

Checkpoints are safetensors plus canonical tagged JSON. State covers parameters,
buffers, tied aliases, named optimizer state/groups, module modes, Python/NumPy/
torch RNGs and phase/cursor/step/transcript/control. Only completed updates with
cleared gradients are checkpointed. The completion manifest is published after
file flushing. Incomplete directories are not valid checkpoints. Interruption tests cover scheduled boundaries, interrupted initial/checkpoint
writes, and a completed checkpoint whose chain entry was not yet committed. A
durable orphan is reused only after regenerated state matches it exactly; an
incomplete orphan is preserved in a sibling recovery directory and recomputed.
Publishing discarded-work evidence and persistent run-key storage remain production
orchestration requirements. A conservative 4096-boundary admission cap prevents
creating a chain too large for the bounded JSON reader.

Each boundary is Ed25519-signed over its canonical body and binds registration,
previous boundary, schedule position, control state and checkpoint inventory.
Full replay regenerates initialization, carries state forward, resets AdamW at
the base-to-chat transition and compares every scheduled boundary. It reads prover
checkpoints solely for comparison, never as a new replay starting state.
Resuming training restores a verified checkpoint; resuming full replay from its own
verified checkpoint is a separate future feature.

Exports omit the duplicate `lm_head.weight` under the explicit alias rule and verify
reconstructed canonical state, exact tensor dtypes/shapes and canonical safetensors
serialization. Each base/chat directory contains its tokenizer and manifest; config
binds the ID shift, controls, inference template and trusted loader source root.
Greedy inference receipts include input/output IDs. Verification rejects extra files
and symlinks and compares published data and inference config to the registration.
A signed release inventory is separate from its children. A release signature
alone never establishes computation or factual accuracy.

## External references

- [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785)
- [Safetensors format and usage](https://huggingface.co/docs/safetensors/index)
- [Tokenizers BPE trainer](https://huggingface.co/docs/tokenizers/api/trainers)
- [PyTorch reproducibility limits](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

The tests and recorded environment define observed fixture support. Pin and measure
the actual GPU/software recipe before any production precommitment.


## Complete-source acquisition and OASST selection

The following implemented command selects only the completed monolithic articles
job from a retained official status document. It does not combine split and
monolithic variants:

```sh
.venv/bin/ovllm-pipeline acquire-wikipedia \
  --status project/evidence/source-survey/dumpstatus.json \
  --date 20260901 --output .ovllm-cache/wikipedia/20260901
```

It validates HTTPS/redirect hosts, length, upstream MD5/SHA-1, local SHA-256 and all
concatenated bzip2 members before atomic promotion. Per-file locking prevents two
writers. Interrupted transfers retain `.partial` bytes under the original pinned
inventory. Range responses are checked; a server ignoring Range restarts the file.
Protocol/checksum failures retain evidence and do not silently become a download
receipt. The command is an acquisition check, not upstream signature verification
or a production preparation commitment. A complete file is reverified before reuse.
Tests cover corrupted/incorrect-range responses, resume, truncated compression,
trailing junk, decompression limits and source swaps.

OASST policy `oasst-en-preferred-path-v1` reads only declared Parquet columns,
requires approved undeleted English ancestry, chooses one branch per root, orders
assistant candidates by known rank then message ID (missing rank last), and orders
prompter branches by ID. It stops before an unanswered terminal prompt. Every
record has a reason-coded outcome; missing parents and unusable ancestry are
excluded explicitly, and otherwise eligible cycles abort. No train/validation tree
identity may overlap. Source record ordinal and projected-field digest preserve
lineage. `prepare_oasst` verifies an external source inventory before reading files.
The real source's schema was inspected after acquisition; no production selection
or training has run yet.
