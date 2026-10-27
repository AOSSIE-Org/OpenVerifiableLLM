# Frequently asked questions

- [What does "verified" mean here?](#what-does-verified-mean-here)
- [Why do I need an RTX 5090 to check a segment?](#why-do-i-need-an-rtx-5090-to-check-a-segment)
- [Isn't checking as expensive as training?](#isnt-checking-as-expensive-as-training)
- [Couldn't the publisher fake the whole run?](#couldnt-the-publisher-fake-the-whole-run)
- [Can a model "forget" a document?](#can-a-model-forget-a-document)
- [Is the model any good?](#is-the-model-any-good)
- [How much did it cost?](#how-much-did-it-cost)
- [What happened to the older pipeline?](#what-happened-to-the-older-pipeline)

## What does "verified" mean here?

The published weights are exactly what this training, on this data, with this code, produces. Every
one of the 37 training segments was recomputed from its published starting checkpoint, and each
result matched the published record bit for bit. It does not mean the model's answers are
correct.

## Why do I need an RTX 5090 to check a segment?

Bit-exact results hold within one GPU architecture. Two different NVIDIA drivers gave identical
bits on RTX 5090s, but an RTX 4090, an H100 and a CPU did not reproduce them, because they run
different low-level kernels. The run was trained on an RTX 5090, so checkers need one too. Some
checks need no GPU at all: the hash links, the published weights and the seeded starting weights.

## Isn't checking as expensive as training?

Checking everything costs about as much as training once. But nobody has to check everything:
each segment is independent and takes about 20 minutes on one GPU. Many people checking one random
segment each is cheap, and catches a forged segment with high probability. See
[What a check proves](../guides/check_a_segment.md#what-a-check-proves).

## Couldn't the publisher fake the whole run?

Any segment that wasn't really computed from its published checkpoint and the published data fails
the first check that picks it. So a convincing fake would have to be a real training run on the
published data, which is the thing being claimed. What a check can't rule out is a segment nobody
has checked yet, which is why independent checks matter. The timestamps add that the record
can't be changed after it was published.

## Can a model "forget" a document?

Not in the usual sense of removing something it already learned. What the method can show is that a
model never trained on a document. For documents the run had not reached yet, that's provable. The project forked the run at
checkpoint 30, removed one Wikipedia article, and retrained from there, with the removal recorded
in the chain. The fork shares everything before the fork byte for byte, and every fork segment
checks out. See [Verifiable exclusion](../design/simple_run.md#results). A document the run had already seen can
only be excluded by retraining from a checkpoint before its first appearance.

## Is the model any good?

It's small (23M parameters). It writes fluent, encyclopedia-style text, gets facts wrong a lot,
and holds a basic conversation. The project's point is the method, which scales to bigger models
unchanged.

## How much did it cost?

Training took 15.7 hours on one rented RTX 5090, about US$19. A clean repeat of training plus
checking every segment costs about US$21. See [Cost](../design/simple_run.md#cost).

## What happened to the older pipeline?

The project first built a large production pipeline with many safeguards around renting machines
and moving checkpoints. It never trained on real data. The simple run replaced it with one script
and segmented verification, and finished in under a day. The older code (`src/ovl_pipeline/`) and
its docs are frozen and kept for reference. See [Things to avoid](../design/simple_run.md#things-to-avoid).
