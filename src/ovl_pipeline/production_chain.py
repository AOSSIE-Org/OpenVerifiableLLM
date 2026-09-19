"""Exact production boundary schedule and run-signature ancestry checks.

The registration root is caller-selected after separate publisher verification.
This checks declarations and signatures, never substitutes for checkpoint bytes,
public boundary anchoring or continuous numerical replay.
"""
import re
from . import schema
from .canonical import EvidenceError,digest,require_digest,sha256
from .production_contract import validate
from .training import verify_signed


def schedule(registration):
    contract=validate(registration);interval=registration['recipe']['boundary_every']
    result=[];global_step=0
    def add(kind,phase,phase_step):
        result.append({'index':len(result),'kind':kind,'phase':phase,
                       'phase_step':phase_step,'global_step':global_step+phase_step})
    add('initial','wikipedia',0)
    for phase in ('wikipedia','conversation'):
        if phase=='conversation':add('transition',phase,0)
        total=registration['coverage'][phase]['updates']
        for step in range(interval,total,interval):add('progress',phase,step)
        add('base' if phase=='wikipedia' else 'final',phase,total)
        global_step+=total
    if len(result)!=contract['primary_boundaries']:raise EvidenceError('production schedule/count disagreement')
    return result


def checkpoint_manifest(value):
    schema.fields(value,'schema state_root files','production checkpoint manifest')
    if value['schema']!='ovl.checkpoint.v1':raise EvidenceError('unsupported safe checkpoint schema')
    require_digest(value['state_root'])
    files=value['files']
    if type(files) is not list or len(files)!=2:raise EvidenceError('complete safe checkpoint inventory required')
    names=[];size=0
    for f in files:
        schema.fields(f,'path bytes sha256','checkpoint file')
        schema.integer(f['bytes'],1,2*1024**3,'checkpoint size');require_digest(f['sha256'])
        names.append(f['path']);size+=f['bytes']
    if names!=['state.json','state.safetensors'] or size>2*1024**3:
        raise EvidenceError('unsafe, ambiguous or oversized checkpoint inventory')


def verify_chain(registration,expected_registration_sha256,envelopes,*,complete):
    require_digest(expected_registration_sha256)
    if digest(registration)!=expected_registration_sha256:raise EvidenceError('registration differs from external selection')
    expected=schedule(registration)
    if type(complete) is not bool:raise EvidenceError('explicit complete-chain requirement needed')
    if type(envelopes) is not list or not 1<=len(envelopes)<=len(expected):raise EvidenceError('empty or oversized production chain')
    if complete and len(envelopes)!=len(expected):raise EvidenceError('incomplete production boundary schedule')
    previous=expected_registration_sha256;prior_control=None
    for index,envelope in enumerate(envelopes):
        schema.fields(envelope,'body signature','signed production boundary')
        if type(envelope['signature']) is not str or not re.fullmatch('[0-9a-f]{128}',envelope['signature']):
            raise EvidenceError('invalid run-signature encoding')
        b=verify_signed(envelope,registration['run_public_key'])
        schema.fields(b,'schema index registration previous kind control checkpoint_path checkpoint','production boundary')
        if b['schema']!='ovl.production-boundary.v1' or b['registration']!=expected_registration_sha256 or b['previous']!=previous:
            raise EvidenceError('broken production boundary ancestry')
        schema.integer(b['index'],0,len(expected)-1,'boundary index')
        if b['index']!=index or b['kind']!=expected[index]['kind'] or b['checkpoint_path']!=f'boundary-{index:05d}':
            raise EvidenceError('production boundary order/kind/path differs')
        c=b['control'];schema.control(c)
        if any(c[k]!=expected[index][k] for k in ('phase','phase_step','global_step')):
            raise EvidenceError('production control differs from exact schedule')
        checkpoint_manifest(b['checkpoint'])
        total=registration['coverage'][c['phase']]['targets']
        if b['kind'] in ('initial','transition'):
            if c['cursor']!=0:raise EvidenceError('phase opening cursor must be zero')
            if b['kind']=='initial':
                if c['transcript']!=sha256(b'ovl.batch-transcript.v1') or b['checkpoint']['state_root']!=registration['initialization']['state_sha256']:
                    raise EvidenceError('initial boundary differs from registered initialization')
            elif c['transcript']!=prior_control['transcript']:raise EvidenceError('phase transition changed batch transcript')
        else:
            if not prior_control['cursor']<c['cursor']<=total:raise EvidenceError('production target progress regressed/exceeded')
            if (b['kind'] in ('base','final'))!=(c['cursor']==total):raise EvidenceError('phase tail coverage differs')
        previous=digest(b);prior_control=c
    return {'schema':'ovl.production-chain-check.v1','result':'PASS','registration_sha256':expected_registration_sha256,
            'scope':'authenticated-run-chain-and-declared-schedule-only','boundaries_checked':len(envelopes),
            'complete_schedule_checked':len(envelopes)==len(expected),'closing_boundary_sha256':previous,
            'checkpoint_bytes':'NOT_RUN','public_boundary_anchors':'NOT_RUN','training_replay':'NOT_RUN',
            'production_admission':'NOT_RUN'}


def verify_artifacts(registration,expected_registration_sha256,envelopes,checkpoint_directory,stream_directories,*,complete):
    """Check all selected safe state bytes plus entire phase inputs and cursors.

    The caller supplies an independently authenticated registration root. Reading
    tensors here only verifies artifacts; it never restores them into a model or
    establishes that the declared optimization trajectory actually happened.
    """
    from .canonical import confined
    from .production_observation import schedule_counts,boundary_cursors
    from .state import read_state,unpack
    checked=verify_chain(registration,expected_registration_sha256,envelopes,complete=complete)
    if type(stream_directories) is not dict or set(stream_directories)!={'wikipedia','conversation'}:
        raise EvidenceError('both complete phase stream directories required')
    expected=schedule(registration);maps={}
    for phase,directory in stream_directories.items():
        census=schedule_counts(directory,registration['recipe'])
        if census!=registration['coverage'][phase]:raise EvidenceError('actual full stream census differs from registration')
        steps=[s['phase_step'] for s in expected if s['phase']==phase]
        maps[phase]=boundary_cursors(directory,registration['recipe'],census['stream_sha256'],steps)
    cursor_lookup={phase:{b['step']:b['targets'] for b in m['boundaries']} for phase,m in maps.items()}
    for envelope in envelopes:
        b=envelope['body'];c=b['control']
        if c['cursor']!=cursor_lookup[c['phase']][c['phase_step']]:
            raise EvidenceError('checkpoint cursor differs from complete input prefix census')
        path=confined(checkpoint_directory,b['checkpoint_path'])
        if (any(p.is_symlink() or not p.is_file() for p in path.iterdir()) or
                {p.name for p in path.iterdir()}!={'checkpoint.json','state.json','state.safetensors'}):
            raise EvidenceError('unexpected production checkpoint artifacts')
        metadata,tensors=read_state(path,b['checkpoint'])
        obj=unpack(metadata['tree'],tensors)
        if obj.get('control')!=c:raise EvidenceError('safe checkpoint control differs from signed boundary')
    return {**checked,'schema':'ovl.production-artifact-check.v1',
            'scope':'authenticated-run-chain-safe-state-bytes-and-complete-input-cursors-only',
            'checkpoint_bytes':'PASS','full_stream_census':'PASS','cursor_maps':maps,
            'public_boundary_anchors':'NOT_RUN','training_replay':'NOT_RUN','production_admission':'NOT_RUN'}
