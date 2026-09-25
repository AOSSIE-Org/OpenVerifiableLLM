"""Shared durable creation claim: one guard admits at most one uncertain POST."""
from pathlib import Path
from .canonical import EvidenceError, digest, read_json, write_json
from .lifecycle import Pending, exclusive


def update(directory, expected, *, close=False, claim=None, accepted=None, unsent=None):
    directory = Path(directory) / 'creation'
    with exclusive(directory):
        path = directory / 'claim.json'
        state = {'guard_sha256': expected, 'admission_closed': False, 'claim': None}
        if path.exists():
            state = read_json(path)
            if (set(state) != {'guard_sha256', 'admission_closed', 'claim'}
                    or state['guard_sha256'] != expected or type(state['admission_closed']) is not bool):
                raise EvidenceError('creation gate identity differs')
            old = state['claim']
            if old is not None and (set(old) != {'operation','request_sha256','status','resource_id'}
                    or old['status'] not in ('unresolved','accepted','not_submitted')):
                raise EvidenceError('invalid creation claim')
        if close:
            state['admission_closed'] = True
        if claim is not None:
            if state['admission_closed']:
                raise EvidenceError('creation admission is closed')
            if state['claim'] is not None:
                raise Pending('guard already has a once-only creation claim; reconcile it')
            operation, request = claim
            state['claim'] = {'operation': operation, 'request_sha256': digest(request),
                              'status': 'unresolved', 'resource_id': None}
        if accepted is not None or unsent is not None:
            operation, request, resource = accepted if accepted is not None else (*unsent, None)
            old = state['claim']
            if old is None or old['operation'] != operation or old['request_sha256'] != digest(request):
                raise EvidenceError('creation acknowledgement has no matching claim')
            if accepted is not None:
                if (type(resource) is not str or not resource or old['status'] == 'not_submitted'
                        or old['resource_id'] not in (None, resource)):
                    raise EvidenceError('creation acknowledgement changed resource identity')
                old.update(status='accepted', resource_id=resource)
            else:
                if old['status'] != 'unresolved':
                    raise EvidenceError('accepted creation cannot be declared unsent')
                old['status'] = 'not_submitted'
        write_json(path, state)
        return state
