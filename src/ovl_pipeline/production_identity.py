"""Separate exact production workflow identity, selected by verifier operator.

Keeping this policy separate preserves the frozen source/preparation signer.
A valid signature still only proves publisher endorsement and log inclusion.
"""
from dataclasses import replace
from .anchoring import PublisherPolicy,WORKFLOW
from .canonical import EvidenceError

PRODUCTION_WORKFLOW='.github/workflows/anchor-production.yml'


class ProductionPublisherPolicy(PublisherPolicy):
    def validate(self):
        if self.workflow!=PRODUCTION_WORKFLOW:
            raise EvidenceError('wrong production publisher workflow')
        # Reuse all exact repository, owner, issuer, ref, source and root gates.
        # Call the base implementation directly; the actual certificate policy
        # continues to use this object's production workflow identity.
        PublisherPolicy.validate(replace(self,workflow=WORKFLOW))
