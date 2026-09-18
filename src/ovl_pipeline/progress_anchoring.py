"""Verify public endorsements for an exact production checkpoint prefix.

Source/registration authentication is a separate prerequisite. This recomputes
progress signature/log checks against separately selected policies; a local run
signature or saved PASS string cannot stand in for a public progress endorsement.
Endorsement and checkpoint availability do not prove the training computation.
"""
from dataclasses import replace
import re

from .anchoring import PublisherPolicy,WORKFLOW,verify_anchor
from .canonical import EvidenceError,canonical,confined,digest,read_json,require_digest,sha256
from .production_chain import verify_chain
from .schema import fields

PROGRESS_WORKFLOW='.github/workflows/anchor-progress.yml'

class ProgressPublisherPolicy(PublisherPolicy):
    def validate(self):
        if self.workflow!=PROGRESS_WORKFLOW:raise EvidenceError('wrong progress publisher workflow')
        PublisherPolicy.validate(replace(self,workflow=WORKFLOW))


def statement(registration,registration_root,envelopes,archive,previous_statement_root):
    """Construct a closed assertion only; no signature or availability credit."""
    verify_chain(registration,registration_root,envelopes,complete=False)
    require_digest(previous_statement_root)
    envelope=envelopes[-1];body=envelope['body'];index=body['index']
    if index==0 and previous_statement_root!=registration_root:
        raise EvidenceError('initial progress endorsement must descend from registration')
    fields(archive,'repo revision prefix inventory','checkpoint archive')
    if (type(archive['repo']) is not str or not re.fullmatch(r'AOSSIE/openverifiable-[a-z0-9-]+-evidence',archive['repo']) or
            type(archive['revision']) is not str or not re.fullmatch('[0-9a-f]{40}',archive['revision'])):
        raise EvidenceError('approved immutable checkpoint archive required')
    if archive['prefix']!='production-checkpoints/'+registration_root+'/'+body['checkpoint_path']:
        raise EvidenceError('checkpoint prefix differs from registered boundary')
    marker=canonical(body['checkpoint'])
    expected=[{'path':'checkpoint.json','bytes':len(marker),'sha256':sha256(marker)},*body['checkpoint']['files']]
    if archive['inventory']!=expected:raise EvidenceError('public checkpoint inventory differs from signed safe state')
    return {'schema':'ovl.production-progress-commitment.v1','registration_sha256':registration_root,
            'index':index,'boundary_sha256':digest(envelope),'previous_statement_sha256':previous_statement_root,
            'archive':archive}


def verify_prefix(registration,registration_root,envelopes,anchor_directory,policies,*,complete):
    """Recompute every selected progress anchor; caller policies are external inputs.

    Directory layout is progress-NNNNN/{statement.json,statement.sigstore.json}.
    No policy is discovered inside that untrusted directory. Registration identity,
    actual checkpoint downloads/state bytes and continuous replay remain separate.
    """
    chain=verify_chain(registration,registration_root,envelopes,complete=complete)
    if type(policies) is not list or len(policies)!=len(envelopes):
        raise EvidenceError('exact externally selected progress policies required')
    previous=registration_root;receipts=[]
    for i,(env,policy) in enumerate(zip(envelopes,policies)):
        if type(policy) is not ProgressPublisherPolicy:raise EvidenceError('exact progress publisher policy required')
        policy.validate()
        directory=confined(anchor_directory,f'progress-{i:05d}')
        if ({p.name for p in directory.iterdir()}!={'statement.json','statement.sigstore.json'} or
                any(p.is_symlink() or not p.is_file() for p in directory.iterdir())):
            raise EvidenceError('unexpected progress anchor files')
        value=read_json(directory/'statement.json')
        fields(value,'schema registration_sha256 index boundary_sha256 previous_statement_sha256 archive','progress commitment')
        expected=statement(registration,registration_root,envelopes[:i+1],value['archive'],previous)
        if value!=expected:raise EvidenceError('public progress statement differs from selected chain/parents')
        if digest(value)!=policy.statement_sha256:raise EvidenceError('progress root differs from external policy')
        receipt=verify_anchor(directory/'statement.json',directory/'statement.sigstore.json',policy)
        receipts.append(receipt);previous=digest(value)
    return {'schema':'ovl.progress-prefix-verification.v1','result':'PASS',
            'scope':'exact-run-prefix-and-public-progress-endorsements-only','registration_sha256':registration_root,
            'closing_boundary_sha256':chain['closing_boundary_sha256'],'closing_statement_sha256':previous,
            'boundaries_checked':len(envelopes),'complete_schedule_checked':chain['complete_schedule_checked'],
            'anchors':receipts,'registration_publisher_identity':'NOT_RUN','checkpoint_downloads':'NOT_RUN',
            'checkpoint_bytes':'NOT_RUN','training_replay':'NOT_RUN','production_admission':'NOT_RUN'}
