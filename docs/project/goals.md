# Goals

## Mission

Make language model training checkable by people who don't trust the publisher. A published model
should come with everything needed to confirm, independently, that it is exactly what the stated
training on the stated data produces.

## Goals

In priority order:

1. **Every step is checkable.** Each part of training can be recomputed from public files and
   compared bit for bit with a published commitment. Checking must not depend on trusting the
   publisher, the hosting platform or the original machine.
2. **The data is traceable to its source.** The training data can be rebuilt from the raw public
   source (for example, a Wikipedia dump) and gives byte-identical files.
3. **Coverage is exact.** The record shows which data was trained on, in which order, and that
   every item was seen exactly as claimed: each item once in a one-pass run, with any exclusion
   recorded.
4. **Checking is cheap and parallel.** Anyone can check one piece in about an hour for about a
   dollar, many people can check different pieces at the same time, and the chance of catching a
   forgery grows with every independent check.
5. **The record cannot be quietly rewritten.** Commitments are hash-linked and timestamped by
   independent authorities.
6. **Simple enough to read.** The training and verification loop fits in one file that a reviewer
   can read in an afternoon.

## Non-goals

- **Proving the model is correct.** Verification proves the computation, not the answers. A
  verified model can still be wrong about facts.
- **Bit-exact results across GPU architectures.** Different architectures run different
  low-level kernels. Checkers use the same GPU type as training, and the project says so up front.
- **Large models first.** The method is shown on a small model that a volunteer can afford to
  check. Larger models follow the same method.
- **Proof-of-learning against a dishonest publisher with unlimited compute.** A publisher who
  fakes a whole internally consistent run is out of scope. What is in scope is making any single
  forged or altered step detectable by a cheap independent check.

## Principles

- **Name the property, not the procedure.** Goals say what must be checkable, not how. The
  verification method can change if a cheaper one checks the same thing.
- **Run something small end to end before building safeguards around it.**
- **Measure, then claim.** Every number in the docs comes from a recorded run.
- **Fail closed.** A check that ran nothing, or could not compare, reports failure.
