# Security policy

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately in one of these ways:

- [GitHub private vulnerability reporting](https://github.com/AOSSIE-Org/OpenVerifiableLLM/security/advisories/new) (preferred)
- email to [rajat.roy@aossie.org](mailto:rajat.roy@aossie.org)
- Discord DM to `reoarei`

Include the affected file or command, how to reproduce it, and its impact. You'll get a reply within
14 days. We fix confirmed problems on `main`, publish an advisory, and credit you unless you ask us
not to. Please wait until a fix is released, or 90 days after your report, before you disclose it.

## What counts

The project's claim is that anyone can check the released model against its timestamped record. So
the most important problems are ways to make a check pass when it shouldn't, for example:

- `ovl.py verify` or `stamp.py --check` reporting PASS or MATCH for a changed chain, anchor,
  timestamp, checkpoint or weights file
- a checkpoint or model file that runs code when the tools in `simple/` load it
- a script in `simple/` that installs or runs something other than what it pins

Ordinary bugs, and problems in third-party services we rely on (Hugging Face, the timestamp
authorities), are welcome as public issues.

## Supported versions

Fixes go into `main` and the next release. The record of the published run doesn't change: a fix
changes the tools that check it.

[AOSSIE's organization-wide security policy](https://github.com/AOSSIE-Org/.github/blob/master/SECURITY.md)
also applies. Past reviews and their fixes are in [`security-reviews/`](security-reviews/).
