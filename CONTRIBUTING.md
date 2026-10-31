# Contributing to OpenVerifiableLLM

Thanks for considering a contribution. You don't have to write code to help:

- **Check a training segment** on an RTX 5090 and post the result ([verify.md](docs/verify.md#2-one-segment-one-click)).
  Independent checks are the most useful contribution right now.
- **Report a bug** or something confusing, in an issue.
- **Improve the docs.** Clearer wording and fixed examples are a good first contribution.
- **Fix or build something.** Pick an open issue, or open one first. Issues labeled
  `good first issue` are a good start. The [roadmap](docs/roadmap.md) lists what is most useful.

## Where the work is

All the code is in [`simple/`](simple/README.md). `simple/ovl.py` is the training loop behind the
released model. Open pull requests against `main`.

For a small fix (a typo, a doc fix, an obvious bug), open a pull request directly. For anything
larger, open an issue first, or comment on an existing one. A change to determinism, the chain
format or `verify` is much cheaper to discuss before you write it. When you start on an issue, say
so in a comment.

## Development setup

Use Python 3.12: the lock file was compiled for it, and `--require-hashes` installs can fail on
other versions.

```bash
git clone https://github.com/<your-username>/OpenVerifiableLLM
cd OpenVerifiableLLM
pip install --require-hashes -r requirements/preparation.lock   # the same CPU environment as CI
pip install pytest gguf ruff
OVL_DEVICE=cpu OVL_TIMESTAMP=0 python -m pytest -q tests        # runs on a CPU, about 15 s
ruff check simple tests
```

Training and segment checks need an RTX 5090. Everything in `tests/` runs on a CPU. To add a
dependency, add it to the lock file, so that every install stays pinned by hash.

## Rules for changes

This is verification software, so three rules apply:

1. **Don't change the bits.** A change to seeds, data order, dtypes, kernels, determinism settings,
   RNG or optimizer state, or how `chain.jsonl` and checkpoints are written changes the hashes and
   breaks verification of the published run. If a change must do this, say so in the PR.
2. **Test the failure, not only the success.** A change to `verify`, the chain or checkpoints needs
   a test where honest input passes and a test where tampered input fails.
3. **Fail closed.** A missing file, hash, timestamp or field must give a failure, never a silent
   skip that ends in PASS.

If a change alters behaviour, commands or output, update the docs in the same PR.

## Pull requests

- Work on a branch off `main`, named for the change, for example `fix/verify-anchor-message`.
- Write commit messages that say why, not only what. Keep each PR to one change.
- Reference the issue (`Closes #123`), and say what you want reviewers to look at.
- Make sure the tests and `ruff` pass. CI runs both on every PR.

Maintainers review as volunteers. A small, tested, well-described PR gets reviewed fastest.

## Bugs and security

A good bug report has the exact command, your OS, Python and PyTorch versions (and for
reproducibility problems, the GPU and CUDA version), what you expected and what happened.

Report a security problem privately, not in a public issue. For example, a way to make `verify`
report a false PASS. See [MAINTAINERS.md](MAINTAINERS.md#reporting-a-vulnerability).

## Discussion

Questions and discussion happen in the project's
[Discord channel](https://discord.com/channels/1022871757289422898/1472946773235466443) on the
AOSSIE server ([join the server](https://discord.gg/hjUhu33uAn) first) and in GitHub issues. If
something is decided privately, summarize it in the issue. Be respectful, and keep disagreement
about the work.

## License

By contributing, you agree that your contributions are licensed under the project's
[LICENSE](LICENSE).
