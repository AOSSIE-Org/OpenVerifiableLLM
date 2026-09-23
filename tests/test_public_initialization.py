"""Synthetic process observations; no machine records or execution credit."""
from copy import deepcopy
import pytest
from ovl_pipeline.canonical import EvidenceError,canonical,digest
from ovl_pipeline.production_parents import public_initialization,validate_parents
from test_production_parents import parents


def selected():
    registration,values=parents()
    private={'record':values['initial_record'],'verification':values['initial_verification']}
    projected=public_initialization(private)
    public={**values,'initial_record':projected['record'],'initial_verification':projected['verification']}
    registration['initialization']['regeneration_report_sha256']=digest(projected['verification'])
    return registration,private,public


def rehash(registration,public):
    public['initial_verification']['record_sha256']=digest(public['initial_record'])
    registration['initialization']['regeneration_report_sha256']=digest(public['initial_verification'])


def test_public_derivative_preserves_numerical_parents_and_private_originals():
    registration,private,public=selected();original=deepcopy(private)
    first=public_initialization(private);assert first==public_initialization(private)
    assert private==original
    assert validate_parents(registration,**public)['assertion_truth_established'] is False
    for kind in ('record','verification'):
        p=first[kind];raw=private[kind]
        assert 'process_observation' not in p
        assert b'synthetic-boot' not in canonical(p) and b'boot_id' not in canonical(p)
        for key in ('recipe','kernel','checkpoint','control','code_root','environment','stream_sha256'):
            if key in raw:assert p[key]==raw[key]
    assert first['verification']['record_sha256']==digest(first['record'])
    assert first['verification']['record_sha256']!=digest(private['record'])


@pytest.mark.parametrize('damage',['unknown-field','bad-process','same-process','wrong-parent','scope'])
def test_projection_does_not_launder_invalid_local_reports(damage):
    _,private,_=selected()
    if damage=='unknown-field':private['record']['private-note']='synthetic extra'
    elif damage=='bad-process':private['record']['process_observation']['private-note']='synthetic extra'
    elif damage=='same-process':private['verification']['process_observation']=private['record']['process_observation']
    elif damage=='scope':private['verification']['process_identity_scope']='different assertion'
    if damage!='wrong-parent':private['verification']['record_sha256']=digest(private['record'])
    else:private['verification']['record_sha256']='0'*64
    with pytest.raises(EvidenceError):public_initialization(private)


@pytest.mark.parametrize('damage',['raw-process','unknown-field','same-process','malformed-commitment',
                                  'wrong-schema','loaded-state','state','recipe','environment','third-party'])
def test_rehashed_public_derivatives_keep_strict_scope_and_parent_checks(damage):
    r,private,p=selected();a=p['initial_record'];b=p['initial_verification']
    if damage=='raw-process':a['process_observation']=private['record']['process_observation']
    elif damage=='unknown-field':a['private-note']='synthetic extra'
    elif damage=='same-process':b['process_observation_commitment']=a['process_observation_commitment']
    elif damage=='malformed-commitment':a['process_observation_commitment']='invalid'
    elif damage=='wrong-schema':b['schema']='ovl.initialization-verification.v1'
    elif damage=='loaded-state':b['prover_tensors_loaded_as_state']=True
    elif damage=='state':a['checkpoint']['state_root']='0'*64
    elif damage=='recipe':a['recipe']['seed']+=1
    elif damage=='environment':a['environment']={'compatible':{'changed':True}}
    else:b['independent_third_party']=True
    rehash(r,p)
    with pytest.raises(EvidenceError):validate_parents(r,**p)


def test_changed_private_process_changes_public_commitment_without_state_changes():
    _,private,_=selected();before=public_initialization(private)
    private['record']['process_observation']['start_ticks']='11'
    private['verification']['record_sha256']=digest(private['record'])
    after=public_initialization(private)
    assert before['record']['process_observation_commitment']!=after['record']['process_observation_commitment']
    assert before['record']['checkpoint']==after['record']['checkpoint']
    assert before['verification']['record_sha256']!=after['verification']['record_sha256']


@pytest.mark.parametrize('kind',['record','verification'])
@pytest.mark.parametrize('field',['process_observation','private-note'])
def test_projection_rejects_private_environment_envelope_fields(kind,field):
    _,private,_=selected()
    private[kind]['environment']={**private[kind]['environment'],field:{'synthetic':'private metadata'}}
    private['verification']['record_sha256']=digest(private['record'])
    with pytest.raises(EvidenceError,match='environment envelope'):public_initialization(private)


@pytest.mark.parametrize('ticks',['010','١٠','-1','1.0',' 10','1'*25])
def test_noncanonical_process_ticks_cannot_manufacture_distinct_identity(ticks):
    _,private,_=selected()
    private['verification']['process_observation']={**private['record']['process_observation'],'start_ticks':ticks}
    with pytest.raises(EvidenceError,match='noncanonical'):public_initialization(private)


@pytest.mark.parametrize('damage',['control-extra','control-progress','parameter-count','warmup-type','recipe-type',
                                   'checkpoint-extra','checkpoint-files','checkpoint-type','environment-extra'])
def test_rehashed_public_metadata_is_structurally_validated(damage):
    r,_,p=selected();a=p['initial_record']
    if damage=='control-extra':a['control']['private-note']='synthetic'
    elif damage=='control-progress':a['control']['global_step']=500
    elif damage=='parameter-count':a['parameter_count']=-1
    elif damage=='warmup-type':a['warmup_updates']=float(a['warmup_updates'])
    elif damage=='recipe-type':a['recipe']={**a['recipe'],'seed':float(a['recipe']['seed'])}
    elif damage=='checkpoint-extra':a['checkpoint']['private-note']='synthetic'
    elif damage=='checkpoint-files':a['checkpoint']['files'].append(a['checkpoint']['files'][0])
    elif damage=='checkpoint-type':a['checkpoint']['files'][0]['bytes']=True
    else:a['environment']={**a['environment'],'process_observation':{'synthetic':'private'}}
    with pytest.raises(EvidenceError):
        rehash(r,p)
        validate_parents(r,**p)


@pytest.mark.parametrize('parent',['source','prepared','pilot_records','pilot_replays','initial_record','initial_verification'])
def test_new_public_packet_forbids_the_observed_process_fields_in_every_parent(parent):
    from ovl_pipeline.production_parents import reject_private_process_fields
    r,private,p=selected()
    p[parent]['nested-extra']={'process_observation':private['record']['process_observation']}
    with pytest.raises(EvidenceError,match='private process'):
        reject_private_process_fields([r,*p.values()])


def test_rehashed_pilot_with_private_process_cannot_enter_public_packet():
    r,private,p=selected();v=p['pilot_replays']['wikipedia']
    v['process_observation']=private['record']['process_observation']
    r['pilots']['wikipedia']['replay_sha256']=digest(v)
    r['forecast_input']['phases']['wikipedia']['replay_sha256']=digest(v)
    with pytest.raises(EvidenceError,match='private process'):validate_parents(r,**p)
