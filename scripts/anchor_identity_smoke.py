"""Generate ONLY a development identity test from checked-out Actions source.

No caller-provided statement/claim/path is signed. This test cannot authorize
data preparation, training or release. Production contracts are separate work.
"""
from dataclasses import asdict
import os
from pathlib import Path
import re
import subprocess

from ovl_pipeline.anchoring import ISSUER, REPOSITORY, REPOSITORY_ID, OWNER_ID, WORKFLOW, PublisherPolicy
from ovl_pipeline.canonical import EvidenceError, digest, inventory, write_json


def generate(root, environ):
    expected_ref = "refs/heads/feat/verifiable-wikipedia-pipeline"
    if (environ.get("GITHUB_REPOSITORY") != REPOSITORY
            or environ.get("GITHUB_REF") != expected_ref
            or environ.get("GITHUB_EVENT_NAME") != "push"):
        raise EvidenceError("identity smoke restricted to approved repository branch push")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()
    sha = git("rev-parse", "HEAD")
    if environ.get("GITHUB_SHA") != sha or git("status", "--porcelain", "--untracked-files=no"):
        raise EvidenceError("checkout revision mismatch or tracked modifications")
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if not re.fullmatch(r"[1-9][0-9]*", environ.get(name, "")):
            raise EvidenceError("missing Actions run identity")
    subjects = inventory(root, [WORKFLOW, "scripts/anchor_identity_smoke.py", "src/ovl_pipeline/anchoring.py",
                                "src/ovl_pipeline/canonical.py", "requirements/anchoring.in", "requirements/anchoring.lock"])
    statement = {
        "schema": "ovl.identity-smoke.v1", "scope": "development-identity-test-only",
        "repository": REPOSITORY, "source_revision": sha, "source_tree_git": git("rev-parse", "HEAD^{tree}"),
        "workflow": WORKFLOW, "ref": expected_ref, "subjects": subjects,
        "ci_run_id": environ["GITHUB_RUN_ID"], "ci_run_attempt": environ["GITHUB_RUN_ATTEMPT"],
        "production_admission": "NOT_RUN", "training_replay": "NOT_RUN",
    }
    policy = PublisherPolicy("ovl.publisher-policy.v2", REPOSITORY, WORKFLOW, ISSUER,
                             expected_ref, sha, digest(statement), "sigstore-production-tuf",
                             REPOSITORY_ID, OWNER_ID, "github-hosted")
    policy.validate()
    output = root / "anchor-smoke"
    output.mkdir(exist_ok=False)
    write_json(output / "statement.json", statement)
    # This is a CI self-check policy, NOT automatically a consumer trust root.
    write_json(output / "ci-self-check-policy.json", asdict(policy))


if __name__ == "__main__":
    generate(Path.cwd(), os.environ)
