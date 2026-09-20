from dataclasses import replace
import pytest
from ovl_pipeline.anchoring import REPOSITORY,REPOSITORY_ID,OWNER_ID,ISSUER,PublisherPolicy
from ovl_pipeline.production_identity import ProductionPublisherPolicy,PRODUCTION_WORKFLOW
from ovl_pipeline.canonical import EvidenceError


def policy():
    return ProductionPublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,PRODUCTION_WORKFLOW,ISSUER,
        'refs/heads/feat/verifiable-wikipedia-pipeline','1'*40,'2'*64,'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')


def test_production_policy_preserves_exact_source_policy_boundary():
    p=policy();p.validate()
    assert p.identity=='https://github.com/AOSSIE-Org/OpenVerifiableLLM/.github/workflows/anchor-production.yml@refs/heads/feat/verifiable-wikipedia-pipeline'
    with pytest.raises(EvidenceError):PublisherPolicy.validate(p)

@pytest.mark.parametrize('field,value',[('workflow','.github/workflows/anchor-pipeline.yml'),('repository','attacker/OpenVerifiableLLM'),
    ('repository_id','123'),('owner_id','123'),('issuer','https://untrusted.invalid'),('ref','refs/heads/main'),
    ('source_revision','main'),('statement_sha256','unknown'),('runner_environment','self-hosted')])
def test_production_identity_cannot_be_selected_by_an_untrusted_artifact(field,value):
    with pytest.raises(EvidenceError):replace(policy(),**{field:value}).validate()


def test_certificate_policy_requires_actual_production_workflow():
    from test_anchoring import certificate
    from sigstore.errors import VerificationError
    p=policy();p.certificate_policy().verify(certificate(p))
    with pytest.raises(VerificationError):p.certificate_policy().verify(certificate(p,{'identity':p.identity.replace('anchor-production.yml','anchor-pipeline.yml')}))
