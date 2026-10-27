# AOSSIE Best Practices Checklist

> Criteria adapted from the [OpenSSF Best Practices Badge](https://github.com/coreinfrastructure/best-practices-badge)
> (MIT / CC BY 3.0) by OpenSSF contributors. Modified for AOSSIE multi-repo template use.
>
> This project's own answers to the full OpenSSF criteria, with evidence for each one, are on
> [bestpractices.dev](https://www.bestpractices.dev/projects/15952).

> **Purpose:** Covers OpenSSF Best Practices criteria that are NOT auto-detected by OpenSSF Scorecard.
> Scorecard already handles: License, SAST tools, CI tests, Security Policy file, Branch Protection,
> Pinned Dependencies, Signed Releases, Maintained status, and Known Vulnerabilities.
>
> **How to use:** tick `[x]` for Met, leave `[ ]` for Unmet, use `[~]` for N/A, and add evidence after
> each item. After editing, update the score table and `checklist-status.json` to match.
>
> **Legend:**
> - 🔴 MUST — Required for passing
> - 🟡 SHOULD — Required unless documented rationale given
> - 🔵 SUGGESTED — Optional but recommended
> - ⚪ N/A — Mark `[~]` if not applicable, add justification

---

## Score Summary

<!-- Updated by hand: when you change a checkbox, update this table and checklist-status.json to match -->
| Category           | Met | Total | Status |
|--------------------|-----|-------|--------|
| Basics             | 8   | 8     | ✅     |
| Change Control     | 6   | 6     | ✅     |
| Reporting          | 8   | 8     | ✅     |
| Quality            | 9   | 11     | 🟡     |
| Security           | 9   | 9     | ✅     |
| Analysis           | 7   | 7     | ✅     |
| **Total**          | **47** | **49** | **96%** |
---

## 🏗️ Basics

### Project Website & Documentation

- [x] 🔴 **description_good** — The project README/website clearly describes what the software does and what problem it solves.
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/README.md

- [x] 🔴 **interact** — The project provides information on how to obtain the software, submit bug reports, and contribute.
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/README.md#getting-started (get it), #join-us (issues, contributing)

- [x] 🔴 **contribution** — `CONTRIBUTING.md` explains the contribution process (e.g., PRs are used, how to open one).
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/CONTRIBUTING.md#opening-a-pull-request

- [x] 🟡 **contribution_requirements** — `CONTRIBUTING.md` references acceptable contribution standards (coding style, tests required, etc.).
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/CONTRIBUTING.md#making-changes (tests, determinism, pinned dependencies)

- [x] 🔴 **documentation_basics** — Basic documentation exists for the software (README, Wiki, or docs folder).
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/docs/README.md

- [x] 🔴 **documentation_interface** — Reference documentation describes the external interface (API inputs/outputs, CLI flags, config schema, etc.).
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/simple/README.md (commands and flags), https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/docs/guides/ovllm_verify.md (`ovllm verify` CLI)

### Other Basics

- [x] 🔴 **discussion** — Project has a searchable, URL-addressable discussion mechanism (GitHub Issues, Discord with archive, mailing list, etc.) that doesn't require proprietary client software.
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/issues

- [x] 🟡 **english** — Documentation is provided in English and English bug reports/comments are accepted.
  - *Note:* All documentation is in English; issues and PRs are in English.

---

## 🔄 Change Control

### Version Control

- [x] 🔵 **repo_distributed** — Project uses a distributed VCS (e.g., git). *(SUGGESTED)*
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM (git)

### Version Numbering

- [x] 🔴 **version_unique** — Each release has a unique version identifier (e.g., v1.0.0).
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/tag/v1.0.0

- [x] 🔵 **version_semver** — Project uses [SemVer](https://semver.org) or [CalVer](https://calver.org/) format. *(SUGGESTED)*
  - *Note:* SemVer (`version` in https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/pyproject.toml).

- [x] 🔵 **version_tags** — Releases are tagged in the VCS (e.g., `git tag v1.0.0`). *(SUGGESTED)*
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/tree/v1.0.0 (git tag `v1.0.0`)

### Release Notes

- [x] 🔴 **release_notes** — Each release includes human-readable release notes summarizing major changes. Raw `git log` output is NOT acceptable.
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/tag/v1.0.0

- [~] 🔴 **release_notes_vulns** — Release notes identify every publicly known vulnerability (with CVE) fixed in that release.
  - *Evidence URL:* N/A — *Justification:* no publicly known vulnerabilities (no CVEs) have been reported.

---

## 🐛 Reporting

### Bug Reporting

- [x] 🔴 **report_process** — A bug-reporting process exists (e.g., GitHub Issues link in README).
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/README.md#join-us

- [x] 🟡 **report_tracker** — An issue tracker (e.g., GitHub Issues) is used to track individual bugs.
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/issues

- [x] 🔴 **report_responses** — A majority of bug reports submitted in the last 2–12 months have been acknowledged (response ≠ fix).
  - *Self-certification note:* 10 of the 16 issues opened since May 2026 have a response (checked 2026-10-26).

- [x] 🟡 **enhancement_responses** — More than 50% of enhancement requests in the last 2–12 months have received a response.
  - *Self-certification note:* Counted with bug reports above: 10 of the 16 issues opened since May 2026 have a response.

- [x] 🔴 **report_archive** — Reports and responses are publicly archived and searchable (GitHub Issues satisfies this).
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/issues?q=is%3Aissue

### Vulnerability Reporting

- [x] 🔴 **vulnerability_report_process** — A vulnerability reporting process is documented (e.g., `SECURITY.md`).
  - *Evidence URL:* https://github.com/AOSSIE-Org/.github/blob/master/SECURITY.md (AOSSIE's org-wide policy, linked from the README and MAINTAINERS.md)

- [x] 🟡 **vulnerability_report_private** — If private vulnerability reporting is supported, the method for private submission is documented.
  - *Evidence URL:* https://github.com/AOSSIE-Org/.github/blob/master/SECURITY.md (private report to the maintainers; GitHub private vulnerability reporting is not enabled on this repository yet)

- [~] 🔴 **vulnerability_report_response** — Initial response to any vulnerability report received in the last 6 months was within 14 days.
  - *Self-certification note:* N/A — *Justification:* no vulnerability reports received.

---

## ✅ Quality

### Build System

- [x] 🔴 **build** — If the project requires building, a working build system exists that can auto-rebuild from source.
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/pyproject.toml (`pip install -e .`); Python needs no compile step.

- [x] 🔵 **build_common_tools** — Common build tools are used (npm, pip, cargo, make, gradle, etc.). *(SUGGESTED)*
  - *Evidence URL:* pip / setuptools

- [x] 🟡 **build_floss_tools** — The project can be built using only FLOSS tools.
  - *Note:* Python, pip, PyTorch, pytest: all FLOSS.

### Automated Testing

- [x] 🔵 **test_invocation** — The test suite can be invoked in a standard way for the language (e.g., `npm test`, `pytest`, `cargo test`). *(SUGGESTED)*
  - *Evidence URL:* `python -m pytest -q tests` (https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/CONTRIBUTING.md#development-setup)

- [ ] 🔵 **test_most** — The test suite covers most code branches, input fields, and functionality. *(SUGGESTED)*
  - *Estimated coverage %:* Measured in CI (job summary): 74% of `simple/` and `src/` overall. Verifier: 78–88% per module; `simple/ovl.py`: about 30% (GPU training paths have no CPU tests yet).

### New Functionality Testing Policy

- [x] 🔴 **test_policy** — The project has a general policy that new functionality must include tests in the automated test suite.
  - *Evidence:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/CONTRIBUTING.md#making-changes (verification changes need passing and tampered-failing tests)

- [x] 🔴 **tests_are_added** — Evidence exists that the test policy has been followed in recent major changes (e.g., PRs include tests).
  - *Evidence URL (recent PR with tests):* https://github.com/AOSSIE-Org/OpenVerifiableLLM/pull/218

- [x] 🔵 **tests_documented_added** — The test policy is documented in contribution instructions. *(SUGGESTED)*
  - *Evidence URL:* https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/CONTRIBUTING.md#making-changes

### Linting / Warning Flags

- [x] 🔴 **warnings** — At least one linter or compiler warning flag is enabled (ESLint, Pylint, clippy, golangci-lint, Slither for Solidity, etc.).
  - *Tool used:* Ruff (pyflakes, pycodestyle errors, flake8-bugbear) in CI on `simple/` and the verifier; config in `pyproject.toml`.

- [x] 🔴 **warnings_fixed** — Warnings from the linter are addressed (not suppressed without reason).
  - *Note:* All findings on the linted code are fixed. Ignored rules are style-only (one-line statements) and are documented in `pyproject.toml`.

- [ ] 🔵 **warnings_strict** — Project uses maximum strictness in linter config where practical. *(SUGGESTED)*
  - *Note:* Not met yet.

---

## 🔐 Security

### Secure Development Knowledge

- [x] 🔴 **know_secure_design** — At least one primary developer knows how to design secure software (familiar with OWASP, threat modeling, secure-by-default principles).
  - *Self-certification note:* The project is verification infrastructure: fail-closed checks, hash-linked records, signed and timestamped anchors. Threat model and limitations in https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/docs/design/simple_run.md.

- [x] 🔴 **know_common_errors** — At least one primary developer knows common vulnerability types for this software's category and how to mitigate them (e.g., injection, XSS, reentrancy for Solidity, prompt injection for AI).
  - *Self-certification note:* Unsafe deserialization (weights load with `weights_only=True` or safetensors), dependency tampering (hash-locked requirements), false verification passes (fail-closed verifier, #205/#218).

### Cryptography (mark N/A if project does not handle cryptography)

- [x] 🔴 **crypto_published** — Only publicly reviewed cryptographic protocols/algorithms are used by default.
  - *Note:* SHA-256, Ed25519, RFC 3161 timestamps, Sigstore.

- [x] 🟡 **crypto_call** — Project calls an established crypto library rather than reimplementing crypto functions.
  - *Library used:* hashlib, PyNaCl (libsodium), OpenSSL (`openssl ts`), sigstore.

- [x] 🔴 **crypto_working** — No broken algorithms (MD4, MD5, single DES, RC4, Dual_EC_DRBG) used unless required for interoperability (must be documented).
  - *Note:* MD5 and SHA-1 appear only to compare against checksums published by Wikimedia dumps and Git object IDs (interoperability); all project commitments use SHA-256.

- [x] 🔴 **crypto_keylength** — Key lengths meet [NIST 2030 minimums](https://www.keylength.com/en/4/) by default.
  - *Note:* Ed25519 (128-bit security), SHA-256.

- [~] 🔴 **crypto_password_storage** — Passwords for external users are stored as iterated salted hashes (Argon2id, bcrypt, scrypt, PBKDF2).
  - *Note:* N/A — *Justification:* the project doesn't store passwords.

- [x] 🔴 **crypto_random** — Cryptographic keys and nonces are generated using a CSPRNG; insecure generators (Math.random, rand()) are NOT used for security purposes.
  - *Note:* Signing keys come from PyNaCl `SigningKey.generate()` (libsodium CSPRNG). Seeded PRNGs are used only for reproducible training and sampling, never for security.

- [x] 🟡 **delivery_unsigned** — Cryptographic hashes are NOT retrieved over plain HTTP without a signature check.
  - *Note:* Release files and hashes are fetched over HTTPS from Hugging Face at a pinned revision and checked against `release-hashes.txt` and the timestamped anchor.

---

## 🔬 Analysis

### Static Code Analysis

- [x] 🔴 **static_analysis_fixed** — All medium+ severity vulnerabilities found by static analysis are fixed in a timely manner after confirmation.
  - *Evidence:* the AI security review's Medium finding was fixed and deployed the same day ([remediation](https://github.com/AOSSIE-Org/OpenVerifiableLLM/blob/main/security-reviews/sec_review_2026-10-26T01-27-31Z_9ae0c16_remediations.md)). CodeQL's three findings were triaged within a day: the tar-extraction one was fixed with a regression test (#244), and two were dismissed with reasons. No CodeQL alerts are open.

- [x] 🔵 **static_analysis_common_vulnerabilities** — The static analysis tool includes checks for common vulnerabilities in the language/environment (e.g., eslint-plugin-security, bandit, Slither). *(SUGGESTED)*
  - *Tool + ruleset:* CodeQL for Python, `security-extended` queries (`.github/workflows/codeql.yml`).

- [x] 🔵 **static_analysis_often** — Static analysis runs on every commit or at least daily (CI integration). *(SUGGESTED)*
  - *Evidence URL:* CodeQL runs on every push to `main`, every pull request, and daily (`.github/workflows/codeql.yml`).

### Dynamic Code Analysis

- [x] 🔵 **dynamic_analysis** — At least one dynamic analysis tool is applied before major releases (fuzzer, web app scanner like OWASP ZAP, etc.). *(SUGGESTED)*
  - *Tool used:* Adversarial (tampering) tests: the suite alters weights, records, signatures and manifests and checks that verification fails (`tests/test_verifier.py`, `tests/simple`).

- [x] 🔵 **dynamic_analysis_enable_assertions** — Dynamic analysis / testing runs with assertions enabled (not just production mode). *(SUGGESTED)*
  - *Note:* Tests run under pytest with assertions enabled.

- [~] 🔴 **dynamic_analysis_fixed** — Medium+ severity vulnerabilities found by dynamic analysis are fixed in a timely manner.
  - *Note:* N/A — no medium or higher vulnerabilities found by dynamic analysis.

- [~] 🔵 **dynamic_analysis_unsafe** — If the project uses memory-unsafe languages (C/C++), memory safety tools (Valgrind, AddressSanitizer) are used. *(SUGGESTED)*
  - *Note:* N/A — *Justification:* Python only; no C or C++ in the project.

---

## 📎 Project-Specific Notes

> Add domain-specific notes here for Web3, Full-Stack, or AI projects.

### Web3 / Solidity Notes
- Scorecard does not audit Solidity-specific security. Use [Slither](https://github.com/crytic/slither) for `static_analysis` and `warnings` criteria.
- For `crypto_*` criteria, document which cryptographic primitives your contracts rely on (e.g., ECDSA in EVM is standard).
- Smart contract audit reports count as evidence for `know_secure_design`.

### Full-Stack / Next.js Notes
- For `crypto_password_storage`: document which auth library handles hashing (e.g., NextAuth + bcrypt).
- For `dynamic_analysis`: [OWASP ZAP](https://www.zaproxy.org/) can be run as a GitHub Action.

### AI / LLM Notes
- For `know_common_errors`: include awareness of prompt injection, data leakage, and model output validation.
- For `dynamic_analysis`: consider adversarial input testing as a form of dynamic analysis.

---

*This checklist complements [OpenSSF Scorecard](https://scorecard.dev/) (auto-detected checks) and is
inspired by the [OpenSSF Best Practices Badge](https://www.bestpractices.dev/en/criteria/0) passing criteria.*
