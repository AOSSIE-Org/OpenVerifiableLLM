"""Check completely retained initialization bytes against operator-selected parents.

These checks do not regenerate weights. The fresh numerical verifier must run in
another process; this checker binds its retained report without promoting an
operator process observation to hardware attestation or third-party acceptance.
"""
from pathlib import Path

from ovl_pipeline import schema
from ovl_pipeline.canonical import EvidenceError,confined,digest,read_json,require_digest,verify_inventory
from ovl_pipeline.state import read_state,unpack


def complete_tree(directory,files):
    directory=Path(directory)
    if any(p.is_symlink() for p in [directory,*directory.absolute().parents]) or not directory.is_dir():
        raise EvidenceError('regular retained initialization directory required')
    actual=[]
    for p in directory.rglob('*'):
        if p.is_symlink() or not (p.is_file() or p.is_dir()):raise EvidenceError('unsafe retained initialization tree')
        if p.is_file():actual.append(p.relative_to(directory).as_posix())
    if sorted(actual)!=[f['path'] for f in files]:raise EvidenceError('incomplete initialization inventory')
    verify_inventory(directory,files)
    return directory


def policy(binding):
    schema.fields(binding,'schema recipe_sha256 kernel_sha256 stream_sha256 code_root parameter_count warmup_updates','initialization parent binding')
    if binding['schema']!='ovl.initialization-record-parent-binding.v1':raise EvidenceError('wrong initialization parent binding')
    for k in ('recipe_sha256','kernel_sha256','stream_sha256','code_root'):require_digest(binding[k])
    schema.integer(binding['parameter_count'],1,2**40,'selected parameter count')
    schema.integer(binding['warmup_updates'],0,10000,'selected discarded warmup updates')
    return binding


def process(value):
    schema.fields(value,'pid boot_id start_ticks','initialization process observation')
    schema.integer(value['pid'],1,2**31-1,'initialization PID')
    # Operator OS observations, not trusted hardware identities. Keep test-only
    # CPU process substitutes explicit at their callers.
    if (type(value['boot_id']) is not str or not 1<=len(value['boot_id'])<=128
        or type(value['start_ticks']) is not str or not value['start_ticks'].isdigit()
        or len(value['start_ticks'])>24):raise EvidenceError('invalid initialization process observation')


def check(directory,files,binding):
    binding=policy(binding);directory=complete_tree(directory,files)
    value=read_json(confined(directory,'record.json'))
    schema.fields(value,'schema scope result recipe kernel warmup_updates warmup_weights_discarded stream_sha256 code_root environment checkpoint control parameter_count process_observation production_admission','retained initialization record')
    if (value['schema']!='ovl.initialization-record.v1' or value['scope']!='preproduction-regenerated-initial-state'
        or value['result']!='RECORDED_AWAITING_FRESH_REGENERATION' or value['warmup_weights_discarded'] is not True
        or value['production_admission']!='NOT_RUN'):
        raise EvidenceError('unsupported retained initialization record')
    for key in ('recipe','kernel'):
        if digest(value[key])!=binding[key+'_sha256']:raise EvidenceError('initialization differs from selected '+key)
    for key in ('stream_sha256','code_root','parameter_count','warmup_updates'):
        if value[key]!=binding[key] or type(value[key]) is not type(binding[key]):
            raise EvidenceError('initialization differs from selected '+key)
    schema.recipe(value['recipe'],gpu=True);schema.control(value['control']);process(value['process_observation'])
    control=value['control']
    if control['phase']!='wikipedia' or any(control[k]!=0 for k in ('global_step','phase_step','cursor')):
        raise EvidenceError('initial state retained training or warmup progress')
    state=confined(directory,'initial-state')
    if {p.name for p in state.iterdir()}!={'checkpoint.json','state.json','state.safetensors'}:
        raise EvidenceError('closed complete initial safe state required')
    if read_json(state/'checkpoint.json')!=value['checkpoint']:raise EvidenceError('initial checkpoint marker differs')
    metadata,tensors=read_state(state,value['checkpoint'])
    if unpack(metadata['tree'],tensors)['control']!=control:raise EvidenceError('initial safe-state control differs')
    return {'schema':'ovl.retained-initialization-record-parent.v1','record_sha256':digest(value),
            'record':value,'files':files,'initial_state_sha256':value['checkpoint']['state_root'],
            'scope':'complete retained initial bytes and selected ancestry; numerical regeneration NOT_RUN'}


def check_verification(directory,files,record_directory,record_files,binding):
    """Retained report consistency only; never execute or claim a fresh replay."""
    parent=check(record_directory,record_files,binding)
    directory=complete_tree(directory,files);v=read_json(confined(directory,'verification.json'));record=parent['record']
    schema.fields(v,'schema result record_sha256 initial_state_sha256 recipe_sha256 code_root stream_sha256 environment warmup_updates scope prover_tensors_loaded_as_state process_observation distinct_process_from_record process_identity_scope performed_by independent_third_party production_admission','retained regeneration report')
    if (v['schema']!='ovl.initialization-verification.v1' or v['result']!='PASS'
        or v['scope']!='complete-initial-state-regenerated-and-compared'
        or v['record_sha256']!=parent['record_sha256'] or v['initial_state_sha256']!=parent['initial_state_sha256']
        or v['prover_tensors_loaded_as_state'] is not False or v['distinct_process_from_record'] is not True
        or v['process_observation']==record['process_observation']
        or v['performed_by']!='project-operator' or v['independent_third_party'] is not False
        or v['production_admission']!='NOT_RUN'
        or v['process_identity_scope']!='operator OS observation, not hardware attestation'):
        raise EvidenceError('regeneration report identity or verified scope differs')
    process(v['process_observation'])
    for key in ('recipe_sha256','code_root','stream_sha256','warmup_updates'):
        if v[key]!=binding[key] or type(v[key]) is not type(binding[key]):raise EvidenceError('regeneration parent differs')
    if v['environment']['compatible']!=record['environment']['compatible']:
        raise EvidenceError('regeneration compatible environment differs')
    return {'schema':'ovl.retained-initialization-cycle.v1','result':'PASS','record_sha256':parent['record_sha256'],
            'verification_sha256':digest(v),'initial_state_sha256':parent['initial_state_sha256'],
            'scope':'complete retained report and safe-state consistency; numerical regeneration not executed by this checker',
            'production_admission':'NOT_RUN','independent_third_party':False}
