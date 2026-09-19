"""Select a replay parent only from completely retained development record bytes.

This checks report ancestry and actual safe states before deriving the next job.
It does not prove that the recorded arithmetic ran; continuous replay remains
required. No remote report or selected PASS field substitutes for retained bytes.
"""
from pathlib import Path

from ovl_pipeline import schema, pilot_delivery
from ovl_pipeline.canonical import EvidenceError,confined,digest,read_json,require_digest,verify_inventory
from ovl_pipeline.state import read_state,unpack


def check(directory,files,binding):
    directory=Path(directory)
    schema.fields(binding,'schema recipe_sha256 kernel_sha256 stream_sha256 code_root','pilot record parent binding')
    if binding['schema']!='ovl.pilot-record-parent-binding.v1':raise EvidenceError('wrong pilot parent binding')
    for k in ('recipe_sha256','kernel_sha256','stream_sha256','code_root'):require_digest(binding[k])
    if directory.is_symlink() or not directory.is_dir():raise EvidenceError('regular retained record directory required')
    actual=[]
    for p in directory.rglob('*'):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):raise EvidenceError('unsafe retained pilot tree')
        if p.is_file():actual.append(p.relative_to(directory).as_posix())
    if sorted(actual)!=[f['path'] for f in files]:raise EvidenceError('retained record inventory is incomplete')
    verify_inventory(directory,files)
    value=read_json(confined(directory,'record.json'))
    if (value.get('schema')!='ovl.gpu-pilot-record.v1' or value.get('scope')!='development-gpu-pilot-only'
        or value.get('result')!='RECORDED_NOT_REPLAYED' or value.get('production_admission')!='NOT_RUN'):
        raise EvidenceError('complete development record required')
    settings=value['settings']
    delivery=pilot_delivery.settings_delivery(settings)
    if settings.get('scope')!='development-gpu-pilot-only':
        raise EvidenceError('wrong pilot settings')
    for key in ('recipe','kernel','stream'):
        if digest(settings[key])!=binding[key+'_sha256']:raise EvidenceError('record differs from selected '+key)
    if settings['code_root']!=binding['code_root']:raise EvidenceError('record differs from selected code')
    schema.recipe(settings['recipe'],gpu=True);schema.stream(settings['stream'])
    schema.integer(value['updates'],1,1_000_000,'recorded pilot updates')
    interval=settings['checkpoint_every'];schema.integer(interval,1,1_000_000,'pilot checkpoint interval')
    steps=list(range(0,value['updates']+1,interval))
    if steps[-1]!=value['updates']:steps.append(value['updates'])
    boundaries=value['boundaries']
    if type(boundaries) is not list or len(boundaries)!=len(steps) or len(boundaries)>4096:
        raise EvidenceError('complete bounded pilot boundary schedule required')
    previous=digest(settings);roots=[];previous_targets=0
    for i,(boundary,step) in enumerate(zip(boundaries,steps)):
        schema.fields(boundary,'index step control path checkpoint previous','retained pilot boundary')
        if (boundary['index']!=i or boundary['step']!=step or boundary['path']!=f'boundary-{i:05d}'
            or boundary['previous']!=previous):raise EvidenceError('retained pilot boundary ancestry differs')
        c=dict(boundary['control']);cycle=c.pop('pilot_cycle',None);schema.control(c)
        schema.integer(cycle,0,value['updates'],'pilot data cycle')
        if (c['global_step']!=step or c['phase_step']!=step or c['phase']!=settings['stream']['phase']
            or c['cursor']>settings['stream']['targets']):raise EvidenceError('retained pilot control differs')
        targets=cycle*settings['stream']['targets']+c['cursor']
        if (i==0 and targets!=0) or (i and targets<=previous_targets):raise EvidenceError('retained pilot target trajectory differs')
        state=confined(directory,boundary['path'])
        if {p.name for p in state.iterdir()}!={'checkpoint.json','state.json','state.safetensors'}:
            raise EvidenceError('incomplete retained safe state')
        metadata,tensors=read_state(state,boundary['checkpoint'])
        if unpack(metadata['tree'],tensors)['control']!=boundary['control']:
            raise EvidenceError('retained safe-state control differs from record')
        roots.append(boundary['checkpoint']['state_root']);previous=digest(boundary);previous_targets=targets
    if previous_targets!=value['measured_targets'] or value['timed_checkpoints']!=len(steps)-1:
        raise EvidenceError('record work differs from retained trajectory')
    if delivery is not None:pilot_delivery.verify_tree(directory,delivery,pilot_delivery.origin(binding),boundaries)
    return {'schema':'ovl.retained-pilot-record-parent.v1','record_sha256':digest(value),'record':value,
            'files':files,'state_roots':roots,'scope':'complete retained bytes and safe-state/control ancestry; arithmetic NOT_RUN'}
