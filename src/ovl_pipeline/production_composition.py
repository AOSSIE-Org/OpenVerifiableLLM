"""Relationships for separately executed operator reconstruction and replay.

This is a report validator, not an execution verifier. The assembly command must
check the actual retained files, audited process exit, full numerical states and
public identities. Consumers can still run production_verify for fresh work.
"""
import re
from .canonical import EvidenceError, digest, require_digest
from .schema import fields, integer


def public_reference(value):
    fields(value, 'url sha256', 'public execution evidence')
    require_digest(value['sha256'])
    url=value['url']
    allowed=(r'https://raw\.githubusercontent\.com/AOSSIE-Org/OpenVerifiableLLM/[0-9a-f]{40}/project/evidence/[A-Za-z0-9._/-]+',
             r'https://huggingface\.co/datasets/AOSSIE/openverifiable-[a-z0-9-]+-evidence/resolve/[0-9a-f]{40}/[A-Za-z0-9._/-]+')
    if type(url) is not str or not any(re.fullmatch(pattern,url) for pattern in allowed) or any(p in ('.','..','') for p in url.split('/')[3:]):
        raise EvidenceError('execution evidence needs an immutable approved public URL')
    return value


def validate(value, registration, reconstruction, replay, process):
    fields(value, 'schema mode reconstruction replay assembly', 'composed execution evidence')
    if (value['schema'] != 'ovl.separate-computation-executions.v1'
            or value['mode'] != 'earlier-complete-clean-reconstruction-and-later-continuous-full-replay'):
        raise EvidenceError('unsupported computation execution composition')
    prior = value['reconstruction']
    fields(prior, 'report_sha256 execution_observation_sha256 public_evidence', 'earlier reconstruction execution')
    for name in ('report_sha256','execution_observation_sha256'): require_digest(prior[name])
    public_reference(prior['public_evidence'])
    if (prior['report_sha256'] != digest(reconstruction)
            or prior['execution_observation_sha256'] != reconstruction['execution_observation_sha256']):
        raise EvidenceError('composed reconstruction selects another execution')
    later = value['replay']
    fields(later, 'report_sha256 audited_process_sha256 launch_sha256 terminal_sha256 retention_sha256 public_evidence',
           'later complete replay execution')
    for name in ('report_sha256','audited_process_sha256','launch_sha256','terminal_sha256','retention_sha256'): require_digest(later[name])
    public_reference(later['public_evidence'])
    fields(process, 'schema launch_sha256 exit_code scope', 'retained audited replay process')
    if (later['report_sha256'] != digest(replay) or later['audited_process_sha256'] != digest(process)
            or later['launch_sha256'] != process['launch_sha256']
            or process['schema'] != 'ovl.audited-runtime-process.v1'
            or type(process['exit_code']) is not int or process['exit_code'] != 0
            or process['scope'] != 'external package audit and constrained target-process launch; not model verification by itself'):
        raise EvidenceError('composed replay process identity or successful exit missing')
    assembly = value['assembly']
    fields(assembly, 'raw_inventory_sha256 prepared_manifest_sha256 record_inventory_sha256 replay_inventory_sha256 '
           'safe_states_compared raw_transformations_executed numerical_updates_executed', 'assembly checks')
    for name in ('raw_inventory_sha256', 'prepared_manifest_sha256', 'record_inventory_sha256', 'replay_inventory_sha256'):
        require_digest(assembly[name])
    integer(assembly['safe_states_compared'], 1, 2**53-1, 'all replay state comparisons')
    if (assembly['prepared_manifest_sha256'] != registration['preparation_sha256']
            or assembly['safe_states_compared'] != len(replay['comparisons']) + len(replay['recovery_checkpoints'])
            or assembly['raw_transformations_executed'] is not False
            or type(assembly['numerical_updates_executed']) is not int or assembly['numerical_updates_executed'] != 0):
        raise EvidenceError('assembly must not claim new reconstruction or numerical execution')
    return value
