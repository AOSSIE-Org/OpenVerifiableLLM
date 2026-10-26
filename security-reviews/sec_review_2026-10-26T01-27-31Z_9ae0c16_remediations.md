# Remediations of Security Review Findings

Review date and time: 2026-10-26T01:27:31Z

## Remediations

### Remediation of Finding 1: The segment checker downloads the run from the mutable `main` branch, skips the anchor check, and labels the result with a pinned revision

- [x] This remediation implements the security review's recommendation for this finding.
- [ ] This remediation addresses the finding in a way that differs from the security review's recommendation.

Remediation commits:
- [5e480c3](https://github.com/AOSSIE-Org/OpenVerifiableLLM/commit/5e480c33544cba5c21aa1cbd1e6396f2e7c0b74d)

Description: `checker_pod.sh` now points the release's `fetch.sh` at the pinned dataset revision, so the record,
checkpoints and data it checks all come from the revision named in the result line. It also downloads
`anchor.json` and its DigiCert RFC 3161 timestamp at that revision, and prints PASS only if the chain head
matches the anchor and the timestamp verifies. It installs `openssl` if it's missing, and a timestamp that
can't be checked is a FAIL. In the repository, `fetch.sh` downloads from `$REV` when it's set, and
`verify_segment.sh` fetches `anchor.json` and passes `--anchor`, so a missing or different anchor fails
instead of being skipped. The optional third recommendation is met through the timestamped anchor rather
than a hard-coded chain head. Tested against the published record: an honest record passes, an edited chain
entry fails on the anchor, and an edited anchor fails on the timestamp. The public RunPod template must point
to this commit for checkers to get the fix.

## Non-remediated findings

None.
