# Contributing to OpenVerifiableLLM

Thanks for considering a contribution. Whether you're fixing a typo, filing a bug, or building a new feature, you're welcome here. This guide explains how to get started and what a contribution needs to look like.

This is a verification project, so it has a couple of project-specific standards (around determinism and falsifiability) that are explained below. Everything else is ordinary open-source practice.

## Ways to contribute

You don't have to write code to be useful:

- **Report a bug** or a confusing behavior by opening an issue.
- **Improve the docs.** Clarifications, fixed examples, and better explanations are genuinely valuable and a great first contribution.
- **Suggest a feature** by opening an issue to discuss it before building.
- **Fix or build something.** Pick up an open issue, or open one for what you have in mind.

If you're new to the project, issues labeled `good first issue` are a good place to start.

## Where the work is

Active development is in `simple/` (the verifiable training loop behind the released model) and in
the `ovllm verify` artifact verifier under `src/`. The earlier production pipeline
(`src/ovl_pipeline/` and the legacy pages listed in [docs/README.md](docs/README.md)) is frozen, so
pull requests that only change it are on hold. Open pull requests against `main`. The
[roadmap](docs/project/roadmap.md) lists what's most useful right now. Checking a training segment
on an RTX 5090 ([guide](docs/guides/check_a_segment.md)) is a valuable contribution too.

## Before writing code

For a small fix (a typo, a doc clarification, an obvious bug), feel free to open a pull request directly.

For anything larger, **open an issue first** (or comment on an existing one) so the approach can be discussed before you invest time. This is partly courtesy and partly practical: changes that touch determinism, the manifest format, or the verifier have correctness consequences that are much cheaper to discuss up front than to unwind at review. A short conversation saves a large PR that has to be reworked.

When you start working on an issue, leave a comment so others know it's taken.

## Project philosophy (worth knowing)

Two ideas shape how this codebase is built, and contributions are expected to respect them:

**Validate assumptions before building on them.** This project exists because a widely-held assumption (that identical training runs produce identical results) turned out to need checking. The same mindset applies throughout: before adding a feature, it's worth confirming the thing it rests on actually holds.

**Falsifiability is first-class.** This is verification infrastructure, so a feature that touches training, serialization, or the manifest needs a way to show it *fails when it should*, not only that it passes when everything is correct. In practice that means new verification behavior should extend the falsifiability test suite.

## Development setup

Dependencies are pinned and hash-locked. This is deliberate: it's both a reproducibility measure (the project's whole point) and a defense against malicious dependency updates. Use the locked environment rather than installing packages by hand.

```bash
# clone your fork
git clone https://github.com/<your-username>/OpenVerifiableLLM
cd OpenVerifiableLLM

# the same environment CI uses (CPU PyTorch, hash-locked)
pip install --require-hashes -r requirements/preparation.lock
pip install -r requirements.txt
pip install -e ".[pipeline]" pytest

# confirm things work
python -m pytest -q tests
```

Which install you need depends on what you work on:

| Working on | Install | Quick test |
|---|---|---|
| `simple/` (the training loop) | the CI environment above, or just PyTorch plus the pinned packages listed in [`simple/README.md`](simple/README.md#requirements) | `OVL_DEVICE=cpu OVL_TIMESTAMP=0 python -m pytest tests/simple` |
| `ovllm verify` (`src/ovllm.py`, `src/verifier.py`) | `pip install -r requirements.txt && pip install -e .` | `python -m pytest tests/test_verifier.py` |
| Anything else | the CI environment above | `python -m pytest -q tests` |

Training and segment checks need an RTX 5090. Everything in `tests/` runs on a CPU; GPU-only tests skip themselves.

If you need to add a dependency, add it through the lockfile so the hash-locked set stays authoritative. An unpinned dependency undermines both reproducibility and the project's supply-chain posture.

## Making changes

**Branches.** Work on a branch off `main`, named for the change, for example `fix/manifest-rng-ordering` or `feat/segment-verifier-cli`.

**Commits.** Write clear, present-tense messages that explain why, not just what. Keep commits focused.

**Tests.**

- Run the test suite (`python -m pytest -q tests`) locally before opening a PR. The determinism checks in CI must pass; a PR that breaks them won't merge.
- Code that changes verification behavior, serialization, RNG/optimizer state handling, or the manifest should include tests for both the success case and the failure case. For verification-affecting changes, extend the test suite so a clean run passes and the relevant tampered run fails.
- A change must not silently weaken the bit-exact reproducibility guarantee on the supported single-GPU stack. If a change trades determinism for performance, make that tradeoff explicit and document it.

**Docs.** If a change alters behavior, inputs, the manifest format, or the verification contract, update the relevant docs in the same PR. The manifest schema is a versioned contract that other parts of the project depend on, so changes there need maintainer sign-off and a migration note.

## Opening a pull request

- Reference the issue it addresses (`Closes #123`) if there is one.
- Describe what changed and why, and flag anything you'd like reviewers to look at closely.
- Confirm the test suite passes.
- Keep each PR to one logical change. Small, focused PRs get reviewed and merged faster than large, multi-purpose ones.

Maintainers review on a volunteer basis, so a clear, well-tested, well-scoped PR is the best way to get a quick response. Expect some back-and-forth; review comments are about the work, not about you.

## Reporting bugs

A good bug report includes enough to reproduce the problem: the exact command, your environment (OS, and for reproducibility issues the GPU, CUDA, and PyTorch versions, since the stack is usually the cause), what you expected, and what actually happened.

For security-relevant issues (for example, a way to make the verifier report a false pass, or a supply-chain concern), please contact the maintainers privately rather than opening a public issue, so it can be addressed before it's widely visible.

## Where discussion happens

Project discussion and questions happen in the [AOSSIE community](https://aossie.org) and in GitHub issues. Technical decisions are made in the open so the reasoning stays available to everyone; if something is discussed privately, it helps to summarize it back into the relevant issue.

## Code of conduct

Be respectful and constructive. Technical disagreement is welcome and expected; keep it about the work, and assume good faith from others.

## License

By contributing, you agree that your contributions are licensed under the project's [LICENSE](LICENSE).
