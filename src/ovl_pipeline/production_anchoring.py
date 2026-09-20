"""Authenticate production/source endorsements and bind their report parents.

No saved PASS receipt is consumed as signature evidence. Policies must be supplied
by the verifier operator separately from the packet. This profile does not execute
reconstruction, GPU checks, training, or admit paid/production work.
"""
import argparse
from dataclasses import asdict
from pathlib import Path

from .anchoring import PublisherPolicy,bounded_bytes,verify_anchor
from .canonical import EvidenceError,canonical,confined,digest,parse_json,sha256
from .production_identity import ProductionPublisherPolicy
from .production_parents import validate_parents

PARENT_FILES={
    'source':'source-statement.json','prepared':'preparation.json',
    'initial_record':'initial-record.json','initial_verification':'initial-verification.json',
}
PACKET_FILES={'registration.json','source-statement.sigstore.json',*PARENT_FILES.values(),
              'wikipedia-pilot-record.json','wikipedia-pilot-replay.json',
              'conversation-pilot-record.json','conversation-pilot-replay.json'}


def object_at(root,name):
    return parse_json(bounded_bytes(confined(root,name),16*1024*1024),canonical_required=True)


def packet_objects(root):
    if root.is_symlink() or not root.is_dir():raise EvidenceError('packet must be a regular directory')
    if {p.name for p in root.iterdir()}!=PACKET_FILES:raise EvidenceError('unexpected/missing production packet file')
    for name in PACKET_FILES:
        path=confined(root,name)
        if not path.is_file():raise EvidenceError('nonregular production packet object')
    parents={k:object_at(root,n) for k,n in PARENT_FILES.items()}
    for kind in ('records','replays'):
        parents['pilot_'+kind]={phase:object_at(root,f'{phase}-pilot-{kind[:-1]}.json') for phase in ('wikipedia','conversation')}
    return object_at(root,'registration.json'),parents


def check_source_parents(root,source_policy,*,policy_origin="caller-supplied"):
    """Verify source signature afresh; check consistency of all selected reports."""
    if type(source_policy) is not PublisherPolicy:raise EvidenceError('exact source publisher policy required')
    source_policy.validate()
    registration,parents=packet_objects(root)
    if sha256(bounded_bytes(confined(root,'source-statement.sigstore.json'),2*1024*1024))!=registration['source_bundle_sha256']:
        raise EvidenceError('source bundle differs from registration')
    source_check=verify_anchor(confined(root,'source-statement.json'),confined(root,'source-statement.sigstore.json'),source_policy,policy_origin=policy_origin)
    parents['source_policy']=asdict(source_policy)
    parent_check=validate_parents(registration,**parents)
    return registration,{'source_anchor':source_check,'parents':parent_check}


def verify_packet(root,registration_bundle,production_policy,source_policy,*,policy_origin="caller-supplied",source_checkout=None):
    if type(production_policy) is not ProductionPublisherPolicy:raise EvidenceError('exact production publisher policy required')
    production_policy.validate()
    # Authenticate the independently selected registration before trusting its
    # parent digests. Its signing revision need not equal the earlier pilot code
    # commit: the signing commit adds an append-only request after those pilots.
    production_check=verify_anchor(confined(root,'registration.json'),registration_bundle,production_policy,policy_origin=policy_origin)
    registration,checks=check_source_parents(root,source_policy,policy_origin=policy_origin)
    code_check='NOT_RUN'
    if source_checkout is not None:
        from .production_commitment import verify_code
        code_check=verify_code(source_checkout,registration)
    return {'schema':'ovl.production-endorsement-verification.v1','result':'PASS',
            'scope':'publisher-endorsements-and-report-parent-consistency-only',
            'registration_sha256':digest(registration),'production_anchor':production_check,**checks,
            'locally_recomputed':['publisher_signatures','certificate_identities','transparency_inclusion','report_parent_consistency'],
            'assertion_truth_established':False,'raw_reconstruction':'NOT_RUN','training_replay':'NOT_RUN',
            'code_and_dependency_lock_binding':code_check,
            'installed_runtime_identity':'NOT_RUN','container_identity':'NOT_RUN','fixed_cost_basis':'NOT_RUN',
            'conversation_split_membership':'NOT_RUN',
            'provider_guard':'NOT_RUN','production_admission':'NOT_RUN'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',type=Path,required=True);p.add_argument('--registration-bundle',type=Path,required=True)
    p.add_argument('--production-policy',type=Path,required=True);p.add_argument('--source-policy',type=Path,required=True)
    p.add_argument('--source-checkout',type=Path,help='optionally recompute code and dependency-lock binding in a trusted clone')
    p.add_argument('--policy-origin',default='caller-supplied',choices=['caller-supplied','ci-self-generated','operator-reconstructed-from-source'])
    a=p.parse_args()
    try:
        production=ProductionPublisherPolicy(**parse_json(bounded_bytes(a.production_policy,65536),canonical_required=True))
        source=PublisherPolicy(**parse_json(bounded_bytes(a.source_policy,65536),canonical_required=True))
        result=verify_packet(a.packet,a.registration_bundle,production,source,policy_origin=a.policy_origin,source_checkout=a.source_checkout)
    except Exception as error:print(canonical({'result':'FAIL','reason':str(error)}).decode());return 1
    print(canonical(result).decode());return 0


if __name__=='__main__':raise SystemExit(main())
