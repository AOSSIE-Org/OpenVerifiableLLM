#!/usr/bin/env python3
"""Check all retained pilot states against an independently selected record.

This reads actual safe checkpoint bytes and checks report consistency. It does
not execute numerical replay, attest a GPU, or establish production acceptance.
"""
from pathlib import Path

from ovl_pipeline.canonical import EvidenceError,confined,digest,read_json,require_digest,verify_inventory
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.state import read_state,unpack
from pilot_record_parent import check as check_record
from ovl_pipeline import pilot_delivery


def complete_inventory(directory,files):
    directory=Path(directory)
    if any(p.is_symlink() for p in [directory,*directory.absolute().parents]):
        raise EvidenceError('regular retained pilot directory required')
    actual=[]
    for p in directory.rglob('*'):
        if p.is_symlink() or not(p.is_file() or p.is_dir()):raise EvidenceError('unsafe retained pilot file')
        if p.is_file():actual.append(p.relative_to(directory).as_posix())
    if sorted(actual)!=[f['path'] for f in files]:raise EvidenceError('complete retained pilot inventory required')
    verify_inventory(directory,files)


def replay_states(directory,files,record,expected,*,resume_from):
    directory=Path(directory);complete_inventory(directory,files)
    report=read_json(confined(directory,'verification.json'))
    delivered=report.get('schema')=='ovl.gpu-pilot-replay.v2'
    fields(report,('delivery ' if delivered else '')+'schema result record_sha256 scope updates_recomputed initial_state_regenerated resume_from compared environment measured_targets measured_full_batch_updates timed_checkpoints measured_ms setup_including_warmup_ms warmup_excluded overhead_included eligible_for_forecast_comparison verifier_checkpoint_overhead_included verifier_checkpoints_saved performed_by independent_third_party production_training_coverage production_admission','retained pilot replay report')
    boundaries=record['boundaries']
    if resume_from is not None:integer(resume_from,1,len(boundaries)-2,'selected resume boundary')
    if report['resume_from'] is not None:integer(report['resume_from'],1,len(boundaries)-2,'reported resume boundary')
    for key in ('updates_recomputed','verifier_checkpoints_saved','timed_checkpoints','measured_ms',
                'setup_including_warmup_ms','measured_targets','measured_full_batch_updates'):
        integer(report[key],0,2**53-1,'retained replay '+key)
    if type(report['compared']) is not list:raise EvidenceError('retained comparison list required')
    for item in report['compared']:
        fields(item,'index state_root','retained state comparison')
        integer(item['index'],0,len(boundaries)-1,'retained comparison index');require_digest(item['state_root'])
    opening=0 if resume_from is None else boundaries[resume_from]['step']
    indices=list(range(len(boundaries))) if resume_from is None else [0,*range(resume_from,len(boundaries))]
    scope='fresh-initialization-continuous-pilot-replay' if resume_from is None else 'training-resume-continuation-probe'
    if (report['schema'] not in ('ovl.gpu-pilot-replay.v1','ovl.gpu-pilot-replay.v2') or report['result']!='PASS'
        or report['record_sha256']!=expected or report['scope']!=scope
        or report['resume_from']!=resume_from or report['updates_recomputed']!=record['updates']-opening
        or report['initial_state_regenerated'] is not True or report['performed_by']!='project-operator'
        or report['independent_third_party'] is not False
        or report['production_training_coverage']!='NOT_RUN' or report['production_admission']!='NOT_RUN'):
        raise EvidenceError('retained pilot replay scope or parent differs')
    if report['environment']['compatible']!=record['settings']['environment']['compatible']:
        raise EvidenceError('retained replay compatible environment differs')
    expected_compared=[{'index':i,'state_root':boundaries[i]['checkpoint']['state_root']} for i in indices]
    if report['compared']!=expected_compared:raise EvidenceError('missing or changed retained boundary comparisons')
    initial_comparisons=1 if resume_from is None else 2
    if (report['verifier_checkpoints_saved']!=len(indices) or report['timed_checkpoints']!=len(indices)-initial_comparisons
        or any(report[k] is not True for k in ('warmup_excluded','overhead_included','verifier_checkpoint_overhead_included'))
        or report['eligible_for_forecast_comparison'] is not (resume_from is None and record['eligible_duration_for_forecast'] is True)):
        raise EvidenceError('retained replay checkpoint accounting differs')
    opening_control=boundaries[0 if resume_from is None else resume_from]['control']
    opening_targets=opening_control['pilot_cycle']*record['settings']['stream']['targets']+opening_control['cursor']
    if report['measured_targets']!=record['measured_targets']-opening_targets:
        raise EvidenceError('retained replay target accounting differs')
    if resume_from is None and report['measured_full_batch_updates']!=record['measured_full_batch_updates']:
        raise EvidenceError('retained full replay update accounting differs')
    if report['measured_full_batch_updates']>report['updates_recomputed']:
        raise EvidenceError('retained replay full batch count exceeds updates')
    expected_paths={'verification.json'}
    if (directory/'timing.json').exists():
        from ovl_pipeline.phase_timing import validate
        validate(read_json(directory/'timing.json'),report,scope='operator-pilot-phase-timing')
        expected_paths.add('timing.json')
    for i in indices:
        name=f'verifier-boundary-{i:05d}';state=confined(directory,name)
        expected_paths.update(name+'/'+p for p in ('checkpoint.json','state.json','state.safetensors'))
        checkpoint=read_json(confined(state,'checkpoint.json'))
        metadata,tensors=read_state(state,checkpoint)
        if (checkpoint['state_root']!=boundaries[i]['checkpoint']['state_root']
            or unpack(metadata['tree'],tensors)['control']!=boundaries[i]['control']):
            raise EvidenceError('retained replay safe state differs from selected record')
    if pilot_delivery.settings_delivery(record['settings']) is not None and not delivered and resume_from is None:
        raise EvidenceError('retained replay omitted selected delivery')
    if delivered:
        selected=pilot_delivery.policy(report['delivery'])
        if selected['mode']!='replay' or selected['phase']!=record['settings']['stream']['phase'] or resume_from is not None:
            raise EvidenceError('retained replay delivery selection differs')
        expected_paths.update(pilot_delivery.verify_tree(directory,selected,pilot_delivery.origin(pilot_delivery.binding(record['settings']),expected),boundaries))
    if {f['path'] for f in files}!=expected_paths:raise EvidenceError('retained verifier tree has unselected or missing files')
    return {'report_sha256':digest(report),'checked_boundaries':indices,'actual_safe_states':len(indices),
            'scope':scope,'updates_reported_recomputed':report['updates_recomputed']}


def verify(record_directory,record_files,binding,expected_record,replay_directory,replay_files,resume_directory,resume_files,*,resume_from):
    require_digest(expected_record);complete_inventory(record_directory,record_files)
    parent=check_record(record_directory,record_files,binding)
    if parent['record_sha256']!=expected_record:raise EvidenceError('record differs from independently selected digest')
    record=parent['record'];record_directory=Path(record_directory)
    if read_json(confined(record_directory,'settings.json'))!=record['settings']:
        raise EvidenceError('retained record settings differ')
    progress=read_json(confined(record_directory,'progress.json'))
    if progress!={'settings_sha256':digest(record['settings']),'boundaries':record['boundaries'],'complete':False}:
        raise EvidenceError('retained record progress differs')
    full=replay_states(replay_directory,replay_files,record,expected_record,resume_from=None)
    resume=replay_states(resume_directory,resume_files,record,expected_record,resume_from=resume_from)
    return {'schema':'ovl.retained-pilot-cycle-verification.v1','result':'PASS','record_sha256':expected_record,
            'binding_sha256':digest(binding),'record_safe_states':len(record['boundaries']),
            'full_replay':full,'resume_probe':resume,
            'scope':'complete retained safe-state bytes and report consistency only; this check executes no numerical replay',
            'locally_recomputed':'safe-state hashes and metadata only','performed_by':'project-operator',
            'independent_third_party':False,'production_acceptance':'NOT_RUN'}


def main():
    import argparse
    from ovl_pipeline.canonical import file_hash,write_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--selection-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        require_digest(a.selection_sha256)
        if file_hash(a.selection)!=a.selection_sha256:raise EvidenceError('pilot cycle selection changed')
        s=read_json(a.selection)
        fields(s,'schema record_directory record_files binding expected_record replay_directory replay_files resume_directory resume_files resume_from','pilot cycle selection')
        if s['schema']!='ovl.retained-pilot-cycle-selection.v1':raise EvidenceError('unsupported cycle selection')
        if a.output.exists() or any(q.is_symlink() for q in [a.output,*a.output.absolute().parents]):
            raise EvidenceError('fresh regular verification output required')
        result=verify(**{k:v for k,v in s.items() if k!='schema'})
        write_json(a.output,result);print(digest(result))
    except Exception as error:p.exit(1,'retained pilot cycle refused: '+type(error).__name__+'\n')


if __name__=='__main__':main()
