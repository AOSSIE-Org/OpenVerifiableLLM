"""Exact publisher identity and transparency checks; no training acceptance implied.

The trust policy is supplied by the verifier operator, never discovered in an
artifact. Sigstore's maintained client verifies certificates, signatures, log
inclusion and signed checkpoints using its independently bootstrapped TUF roots.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib.metadata
from pathlib import Path
import re

from .canonical import EvidenceError, canonical, digest, parse_json, read_json, require_digest, sha256

REPOSITORY = "AOSSIE-Org/OpenVerifiableLLM"
REPOSITORY_ID = "1159706660"
OWNER_ID = "38881995"
ISSUER = "https://token.actions.githubusercontent.com"
WORKFLOW = ".github/workflows/anchor-pipeline.yml"


@dataclass(frozen=True)
class PublisherPolicy:
    schema: str
    repository: str
    workflow: str
    issuer: str
    ref: str
    source_revision: str
    statement_sha256: str
    trust_root: str
    repository_id: str
    owner_id: str
    runner_environment: str

    def validate(self):
        if (self.schema != "ovl.publisher-policy.v2" or self.repository != REPOSITORY
                or self.workflow != WORKFLOW or self.issuer != ISSUER
                or self.trust_root != "sigstore-production-tuf"
                or self.repository_id != REPOSITORY_ID or self.owner_id != OWNER_ID
                or self.runner_environment != "github-hosted"):
            raise EvidenceError("unsupported external publisher policy")
        if self.ref != "refs/heads/feat/verifiable-wikipedia-pipeline":
            raise EvidenceError("unapproved publisher source ref")
        if not isinstance(self.source_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.source_revision):
            raise EvidenceError("invalid publisher source revision")
        require_digest(self.statement_sha256)

    @property
    def identity(self):
        return f"https://github.com/{self.repository}/{self.workflow}@{self.ref}"

    def certificate_policy(self):
        from sigstore.verify import policy
        self.validate()
        return policy.AllOf([
            policy.Identity(identity=self.identity, issuer=self.issuer),
            policy.OIDCSourceRepositoryURI(f"https://github.com/{self.repository}"),
            policy.OIDCSourceRepositoryDigest(self.source_revision),
            policy.OIDCSourceRepositoryRef(self.ref),
            policy.OIDCSourceRepositoryIdentifier(self.repository_id),
            policy.OIDCSourceRepositoryOwnerIdentifier(self.owner_id),
            policy.OIDCRunnerEnvironment(self.runner_environment),
        ])


def bounded_bytes(path: Path, limit: int):
    if path.is_symlink() or not path.is_file():
        raise EvidenceError("expected regular anchoring input")
    with path.open("rb") as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise EvidenceError("anchoring input exceeds size limit")
    return data


def verify_anchor(statement_path: Path, bundle_path: Path, policy: PublisherPolicy, *, policy_origin="caller-supplied"):
    """Recompute identity/log verification. A saved receipt cannot bypass this.

    The statement digest is selected outside the untrusted artifact. This function
    verifies endorsement of those exact bytes, not the truth of their assertions.
    Production admission must additionally validate the stage-specific contract.
    """
    from sigstore.models import Bundle, ClientTrustConfig
    from sigstore.verify import Verifier
    if importlib.metadata.version("sigstore") != "4.5.0":
        raise EvidenceError("unsupported Sigstore version; install the hash-locked anchoring environment")
    if policy_origin not in ("caller-supplied", "ci-self-generated", "operator-reconstructed-from-source"):
        raise EvidenceError("unsupported policy origin label")
    policy.validate()
    raw = bounded_bytes(statement_path, 16 * 1024 * 1024)
    parse_json(raw, canonical_required=True)
    if sha256(raw) != policy.statement_sha256:
        raise EvidenceError("statement differs from independently selected digest")
    raw_bundle = bounded_bytes(bundle_path, 2 * 1024 * 1024)
    obj = parse_json(raw_bundle)  # Reject duplicate keys before protobuf parsing.
    if obj.get("mediaType") != "application/vnd.dev.sigstore.bundle.v0.3+json":
        raise EvidenceError("only Sigstore bundle v0.3 supported")
    try:
        bundle = Bundle.from_json(raw_bundle)
        # Always refresh production TUF. No artifact-selected endpoint, root, or
        # offline switch; unavailability is a failure, never a cached PASS receipt.
        config = ClientTrustConfig.production(offline=False)
        Verifier(trusted_root=config.trusted_root).verify_artifact(raw, bundle, policy.certificate_policy())
    except Exception as e:
        raise EvidenceError(f"Sigstore verification failed: {type(e).__name__}: {e}") from e
    # Report verified, normalized bundle fields. Only the proof's tree-relative
    # index is an inclusion coordinate. Rekor's global/sharded locator can differ;
    # do not label it as a proof coordinate or require those indices to be equal.
    material = parse_json(bundle.to_json().encode())["verificationMaterial"]
    entry = material["tlogEntries"][0]
    proof = entry["inclusionProof"]
    # A pinned-client adapter for audit export only; roots were obtained above
    # from TUF, not parsed from the incoming bundle or this diagnostic export.
    root_bytes = config.trusted_root._inner.to_json().encode("utf-8")
    return {
        "schema": "ovl.anchor-verification.v2", "result": "PASS",
        "scope": "artifact-publisher-identity-and-log-inclusion-only",
        "statement_sha256": sha256(raw), "bundle_sha256": sha256(raw_bundle),
        "policy_sha256": digest(asdict(policy)), "identity": policy.identity,
        "policy_origin_operator_label": policy_origin,
        "independent_policy_selection_cryptographically_established": False,
        "issuer": policy.issuer, "source_revision": policy.source_revision,
        "trust_root_source": policy.trust_root, "trusted_root_export_sha256": sha256(root_bytes),
        "sigstore_version": importlib.metadata.version("sigstore"),
        "transparency": {"log_id": entry["logId"], "proof_tree_index": proof["logIndex"],
                         "tree_size": proof["treeSize"], "tree_root_hash_base64": proof["rootHash"],
                         "checkpoint_envelope_bytes_sha256": sha256(proof["checkpoint"]["envelope"].encode())},
        "trust_root_export_note": "diagnostic snapshot of TUF-delivered roots; may change over time, not a stable root identifier",
        "log_consistency_between_observations": "NOT_RUN", "independent_witness": "NOT_RUN",
        "locally_recomputed": ["signature", "certificate_identity", "certificate_source_revision", "log_inclusion", "signed_log_checkpoint"],
        "training_replay": "NOT_RUN", "production_admission": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statement", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--trust-policy", required=True, type=Path)
    parser.add_argument("--policy-origin", default="caller-supplied",
                        choices=["caller-supplied", "ci-self-generated", "operator-reconstructed-from-source"])
    args = parser.parse_args()
    try:
        external = read_json(args.trust_policy)
        result = verify_anchor(args.statement, args.bundle, PublisherPolicy(**external), policy_origin=args.policy_origin)
    except Exception as e:
        print(canonical({"result": "FAIL", "reason": str(e)}).decode())
        return 1
    print(canonical(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
