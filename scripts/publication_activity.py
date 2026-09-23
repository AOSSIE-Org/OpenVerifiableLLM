"""Finite local publisher progress observations; never cryptographic acceptance.

Only completed source-owned gates or identified successful Actions steps emit a
record. The workload supervisor must observe each identity once across restarts;
polling/mtime changes supply no progress. The publisher deadline is never renewed.
"""
from pathlib import Path
import re
import time
from ovl_pipeline.canonical import EvidenceError,canonical,digest,read_json,require_digest,write_json
from ovl_pipeline.schema import fields,integer

STAGES={'checkpoint-public-download-verified','request-public-commit-verified','actions-run-observed',
        'actions-anchor-signature-verified','anchor-public-download-verified',
        'checkpoint-privacy-review-verified','anchor-privacy-review-verified'}


def stage_name(name):
    if type(name) is not str or (name not in STAGES and not re.fullmatch(r'actions-step-(?:0[1-9]|[12][0-9]|3[0-2])',name)):
        raise EvidenceError('unregistered finite publication stage')
    return name


def emit(directory,registration,boundary,stage,identity,deadline,*,wall=time.time):
    require_digest(registration);require_digest(boundary);stage_name(stage)
    integer(deadline,1,2**53-1,'fixed publisher deadline')
    if wall()>=deadline:raise EvidenceError('publisher activity cannot extend expired deadline')
    if len(canonical(identity))>128*1024:raise EvidenceError('publication identity too large')
    directory=Path(directory);directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    value={'schema':'ovl.publication-activity.v1','registration_sha256':registration,'boundary_sha256':boundary,
           'stage':stage,'identity':identity,'identity_sha256':digest(identity),'deadline_epoch':deadline,
           'scope':'operator-publication-liveness-only-not-training-or-signer-verification'}
    path=directory/(stage+'.json')
    if path.exists():
        if read_json(path)!=value:raise EvidenceError('publication stage identity/deadline changed; preserve original')
        return False
    write_json(path,value);return True


def validate(value,registration,boundary,deadline):
    fields(value,'schema registration_sha256 boundary_sha256 stage identity identity_sha256 deadline_epoch scope','publication progress')
    if (value['schema']!='ovl.publication-activity.v1' or value['registration_sha256']!=registration
        or value['boundary_sha256']!=boundary or value['deadline_epoch']!=deadline
        or value['scope']!='operator-publication-liveness-only-not-training-or-signer-verification'):
        raise EvidenceError('publication observation differs from fixed selection')
    stage_name(value['stage']);require_digest(value['identity_sha256'])
    if digest(value['identity'])!=value['identity_sha256']:raise EvidenceError('publication identity content differs')
    return value
