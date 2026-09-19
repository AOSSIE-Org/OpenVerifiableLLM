import copy
from pathlib import Path

import pytest
from sustained_pilot_selection import derive,DEADLINE,RECORD
from pilot_record_parent import check
from test_gpu_pilot import cpu_runtime,prepared
from ovl_pipeline import gpu_pilot
from ovl_pipeline.fixture import recipe
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json


def descriptor():
    return {'schema':'ovl.pod-job.v1','kind':'pilot','argv':['/selected/python','--deadline',DEADLINE],
            'cwd':'/selected','environment':{'PATH':'/usr/bin:/bin'},'deadline_epoch':DEADLINE,
            'stop_grace_seconds':15,'minimum_free_bytes':1000,'required_files':[{'path':'/selected/python','bytes':1,'sha256':'a'*64}],
            'export_roots':['/selected/output']}


def stage(template):
    return {'name':'record','template_sha256':digest(template),'work_seconds':1200,'export_reserve_seconds':650,'parent_record_root':None}


PLAN={'input':{'now_epoch':1000,'checkpoint_grace_seconds':900},'request_checkpoint_epoch':10000}


def test_measured_phase_is_frozen_once_and_adopted_after_its_deadline(tmp_path):
    t=descriptor();s=stage(t)
    path,root,value=derive(s,t,tmp_path/'job','b'*64,PLAN,2000)
    assert value['deadline_epoch']==3200 and value['argv'][-1]=='3200'
    assert derive(s,t,tmp_path/'job','b'*64,PLAN,5000)==(path,root,value)
    path.unlink()  # interrupted publication repairs exact descriptor, not a new deadline
    assert derive(s,t,tmp_path/'job','b'*64,PLAN,6000)==(path,root,value)
    (tmp_path/'job/selection.json').unlink()
    with pytest.raises(EvidenceError,match='lost its original'):derive(s,t,tmp_path/'job','b'*64,PLAN,7000)


def test_longer_measured_pilot_window_still_uses_original_rental_and_deadline(tmp_path):
    t=descriptor();s=stage(t);s['work_seconds']=2100
    selected=derive(s,t,tmp_path/'job','b'*64,PLAN,2000)
    assert selected[2]['deadline_epoch']==4100
    assert derive(s,t,tmp_path/'job','b'*64,PLAN,5000)==selected
    for name,seconds,kind,now in [('too-long',2101,'pilot',2000),
                                  ('setup',1501,'setup',2000),
                                  ('late',2100,'pilot',8000)]:
        changed=copy.deepcopy(t);changed['kind']=kind
        selection={**s,'work_seconds':seconds,'template_sha256':digest(changed)}
        with pytest.raises(EvidenceError):derive(selection,changed,tmp_path/name,'b'*64,PLAN,now)


@pytest.mark.parametrize('damage',['remaining','grace','template','production','changed-adoption','changed-deadline'])
def test_no_shortened_phase_renewal_or_foreign_job(tmp_path,damage):
    t=descriptor();s=stage(t);p=copy.deepcopy(PLAN);now=2000
    if damage=='remaining':now=9000
    elif damage=='grace':p['input']['checkpoint_grace_seconds']=300
    elif damage=='template':t['minimum_free_bytes']=1
    elif damage=='production':t['kind']='production-record';s['template_sha256']=digest(t)
    else:
        derive(s,t,tmp_path/'job','b'*64,p,now)
        if damage=='changed-adoption':s['work_seconds']+=1
        else:
            selected=read_json(tmp_path/'job/selection.json');selected['deadline_epoch']+=1
            write_json(tmp_path/'job/selection.json',selected)
    with pytest.raises(EvidenceError):derive(s,t,tmp_path/'job','b'*64,p,now)


def recorded(tmp_path,prepared):
    directory,manifest=prepared;r=recipe(manifest['tokenizer']['vocab_size']);kernel={'schema':'ovl.gpu-kernel.v1','precision':'fp32'}
    out=tmp_path/'record';value=gpu_pilot.record(directory/'wikipedia',r,kernel,out,updates=6,checkpoint_every=3)
    files=inventory(out,[p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()])
    settings=value['settings'];binding={'schema':'ovl.pilot-record-parent-binding.v1',
        **{k+'_sha256':digest(settings[k]) for k in ('recipe','kernel','stream')},'code_root':settings['code_root']}
    return out,value,files,binding


def test_replay_parent_derived_from_actual_full_retention_and_rehashed_by_worker(cpu_runtime,prepared,tmp_path):
    out,value,files,binding=recorded(tmp_path,prepared);parent=check(out,files,binding)
    assert len(parent['state_roots'])==3 and parent['record_sha256']==digest(value)
    t=descriptor();t['argv']+=['--expected-record-sha256',RECORD];s=stage(t);s['parent_record_root']='/selected/record'
    path,root,job=derive(s,t,tmp_path/'derived','b'*64,PLAN,2000,parent=parent)
    assert job['argv'][-1]==digest(value)
    assert job['required_files'][1:]==[{**f,'path':'/selected/record/'+f['path']} for f in files]
    assert derive(s,t,tmp_path/'derived','b'*64,PLAN,9000,parent=parent)[1]==root


@pytest.mark.parametrize('damage',['bytes','omitted','settings','ancestry','control','missing-marker','report-not-retained'])
def test_incomplete_or_altered_record_cannot_become_replay_parent(cpu_runtime,prepared,tmp_path,damage):
    out,value,files,binding=recorded(tmp_path,prepared)
    if damage=='bytes':(out/'boundary-00001/state.safetensors').write_bytes(b'corrupt')
    elif damage=='omitted':files=[f for f in files if not f['path'].startswith('boundary-00001/')]
    elif damage=='settings':binding['stream_sha256']='f'*64
    elif damage=='missing-marker':(out/'boundary-00001/checkpoint.json').unlink()
    elif damage=='report-not-retained':files=[f for f in files if f['path']!='record.json']
    else:
        if damage=='ancestry':value['boundaries'][1]['previous']='f'*64
        else:value['boundaries'][1]['control']['transcript']='f'*64
        write_json(out/'record.json',value);files=inventory(out,[p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()])
    with pytest.raises(EvidenceError):check(out,files,binding)
