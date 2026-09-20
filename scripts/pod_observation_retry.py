"""One finite retry for read-only workload supervision/metadata observations.

No launch, upload, stop or abandonment mutation is accepted here. No progress or
export credit is awarded. The external deadline never changes; heartbeat writes
only demonstrate the coordinator's bounded continued activity.
"""
from pathlib import Path
import time

from ovl_pipeline.canonical import EvidenceError,write_json,digest
from pod_transfer import TransientTransportError
from pod_job_client import job_supervision
from pod_observe import observe_many


def read(kind, transport, selection, health, health_file, output, *, sleep=time.sleep,private_diagnostics=None):
    if kind not in ('supervision','metadata'):
        raise EvidenceError('only read-only supervision and metadata may retry')
    limit=min(health.plan['external_terminate_epoch'],health.now()+45)
    for attempt in range(2):
        phase='health-before';deadline=None
        try:
            health.write(health_file)
            deadline=min(limit,health.now()+20)
            if deadline<=health.now():raise EvidenceError('original observation deadline expired')
            phase=kind
            if kind=='supervision':
                value=job_supervision(transport,selection[0],selection[1],deadline)
            else:value=observe_many(transport,selection,65536,deadline)
            phase='health-after';health.write(health_file)
        except Exception as error:
            try:
                from private_transport_diagnostics import retain
                retain(error,private_diagnostics,{'kind':kind,'phase':phase,'attempt':attempt,
                    'operation_deadline_epoch':deadline,'fixed_observation_deadline_epoch':limit,
                    'external_deadline_epoch':health.plan['external_terminate_epoch'],
                    'profile_sha256':digest(transport.profile) if transport is not None else None,
                    'selection':list(selection) if kind=='supervision' else sorted(selection)})
            except Exception:pass  # even diagnostic context failures preserve the primary error
            # A health or evidence failure is never reclassified as transport.
            if not isinstance(error,TransientTransportError) or phase!=kind or getattr(error,'transport_cleanup_diagnostic',None) is not None:raise
            write_json(Path(output)/f'{kind}-transport-failure-{attempt}.json',{
                'schema':'ovl.bounded-read-failure.v1','kind':kind,'attempt':attempt,
                'observed_epoch':health.now(),'fixed_observation_deadline_epoch':limit,
                'external_deadline_epoch':health.plan['external_terminate_epoch'],
                'category':'transient-transport','progress_credit':False})
            if attempt or health.now()+3>=limit:raise
            health.write(health_file);sleep(3)
        else:
            return value
