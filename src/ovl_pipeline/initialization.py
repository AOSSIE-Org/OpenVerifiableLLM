"""Record and freshly regenerate a safe GPU initial state before registration.

Warmup uses discarded weights. Verification compares a newly generated complete
state; it never loads the recorded tensors into the model. This is operator
initialization evidence, not production training or independent verification.
"""
import argparse
import os
from pathlib import Path

from . import gpu_pilot,schema
from .canonical import EvidenceError,canonical,confined,digest,read_json,require_digest,write_json
from .data import validate_stream
from .state import capture,read_state,save_state,state_root
from .training import code_root


def process_identity():
    return {'pid':os.getpid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}


def fresh(directory,recipe,kernel,warmup_updates):
    schema.recipe(recipe,gpu=True)
    manifest=read_json(confined(directory,'stream.json'))
    if manifest['phase']!='wikipedia':raise EvidenceError('initialization warmup uses registered Wikipedia stream only')
    validate_stream(directory,manifest)
    model,optimizer,control,flags,environment=gpu_pilot.initialize(directory,recipe,kernel,manifest,warmup_updates)
    # pilot_cycle only describes pilot traversal; production state uses the
    # original shared kernel's closed control schema after discarded warmup.
    if control.pop('pilot_cycle')!=0:raise EvidenceError('initialization retained pilot progress')
    schema.control(control)
    if control['global_step']!=0 or control['phase_step']!=0 or control['cursor']!=0:
        raise EvidenceError('initialization retained warmup updates')
    return model,optimizer,control,environment,manifest


def record(directory,recipe,kernel,output,*,warmup_updates=4):
    if output.exists():raise EvidenceError('initialization output must be fresh')
    model,opt,control,environment,stream=fresh(directory,recipe,kernel,warmup_updates)
    output.mkdir(parents=True,exist_ok=False)
    checkpoint=save_state(output/'initial-state',model,opt,control)
    value={'schema':'ovl.initialization-record.v1','scope':'preproduction-regenerated-initial-state',
           'result':'RECORDED_AWAITING_FRESH_REGENERATION','recipe':recipe,'kernel':kernel,
           'warmup_updates':warmup_updates,'warmup_weights_discarded':True,'stream_sha256':digest(stream),
           'code_root':code_root(),'environment':environment,'checkpoint':checkpoint,
           'control':control,'parameter_count':sum(p.numel() for p in model.parameters()),
           'process_observation':process_identity(),
           'production_admission':'NOT_RUN'}
    write_json(output/'record.json',value)
    return value


def verify(directory,record_directory,expected_record_sha256,output):
    require_digest(expected_record_sha256)
    if output.exists():raise EvidenceError('fresh regeneration output must be new')
    value=read_json(confined(record_directory,'record.json'))
    if digest(value)!=expected_record_sha256:raise EvidenceError('initialization record differs from selected root')
    schema.fields(value,'schema scope result recipe kernel warmup_updates warmup_weights_discarded stream_sha256 code_root environment checkpoint control parameter_count process_observation production_admission','initialization record')
    if (value['schema']!='ovl.initialization-record.v1' or value['scope']!='preproduction-regenerated-initial-state'
            or value['result']!='RECORDED_AWAITING_FRESH_REGENERATION' or value['warmup_weights_discarded'] is not True
            or value['production_admission']!='NOT_RUN' or value['code_root']!=code_root()):
        raise EvidenceError('unsupported initialization record or code drift')
    process=process_identity()
    schema.fields(value['process_observation'],'pid boot_id start_ticks','initialization process observation')
    if process==value['process_observation']:
        raise EvidenceError('initialization comparison requires a fresh process')
    model,opt,control,environment,stream=fresh(directory,value['recipe'],value['kernel'],value['warmup_updates'])
    if digest(stream)!=value['stream_sha256'] or environment['compatible']!=value['environment']['compatible']:
        raise EvidenceError('initialization stream or compatible runtime differs')
    if control!=value['control'] or sum(p.numel() for p in model.parameters())!=value['parameter_count']:
        raise EvidenceError('initialization recipe/control differs')
    initial_directory=confined(record_directory,'initial-state')
    if any(p.is_symlink() or not p.is_file() for p in initial_directory.iterdir()) or {p.name for p in initial_directory.iterdir()}!={'checkpoint.json','state.json','state.safetensors'}:
        raise EvidenceError('unexpected initial checkpoint artifact')
    md,tensors=read_state(initial_directory,value['checkpoint'])
    actual=state_root(*capture(model,opt,control))
    if actual!=state_root(md,tensors):raise EvidenceError('freshly regenerated initial state differs')
    report={'schema':'ovl.initialization-verification.v1','result':'PASS','record_sha256':expected_record_sha256,
            'initial_state_sha256':actual,'recipe_sha256':digest(value['recipe']),'code_root':value['code_root'],
            'stream_sha256':digest(stream),'environment':environment,'warmup_updates':value['warmup_updates'],
            'scope':'complete-initial-state-regenerated-and-compared','prover_tensors_loaded_as_state':False,
            'process_observation':process,'distinct_process_from_record':True,
            'process_identity_scope':'operator OS observation, not hardware attestation',
            'performed_by':'project-operator','independent_third_party':False,'production_admission':'NOT_RUN'}
    output.mkdir(parents=True,exist_ok=False);write_json(output/'verification.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('record');a.add_argument('--recipe',type=Path,required=True);a.add_argument('--kernel',type=Path,required=True);a.add_argument('--warmup-updates',type=int,default=4)
    b=sub.add_parser('verify');b.add_argument('--record-directory',type=Path,required=True);b.add_argument('--expected-record-sha256',required=True)
    for parser in (a,b):parser.add_argument('--stream',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        result=(record(a.stream,read_json(a.recipe),read_json(a.kernel),a.output,warmup_updates=a.warmup_updates)
                if a.action=='record' else verify(a.stream,a.record_directory,a.expected_record_sha256,a.output))
    except Exception as e:print(canonical({'result':'FAIL','reason':str(e)}).decode());return 1
    print(canonical({'result':result['result'],'report_sha256':digest(result),'production_admission':'NOT_RUN'}).decode());return 0


if __name__=='__main__':raise SystemExit(main())
