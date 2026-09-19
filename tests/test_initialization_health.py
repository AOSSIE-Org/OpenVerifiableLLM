"""Actual tiny scans plus explicit CPU warmup; no CUDA or regeneration credit."""
import pytest

from initialization_health import InitializationHealth,InitializationCycleHealth
from test_workload_health import Clock,JOB,SELECTION,activity
from test_external_watchdog import intent,NOW
from test_pilot_health import validation
from test_pipeline import prepared
from test_gpu_pilot import cpu_runtime
from test_production_observation import observations
from ovl_pipeline.canonical import EvidenceError,digest,read_json
from ovl_pipeline.supervision import Journal


BINDING={'schema':'ovl.initialization-validation-binding.v1','stream_sha256':'c'*64,
         'documents':100,'action':'record'}


def health(j,c,binding=None):
    return InitializationHealth(j,intent(),'owned-pod',{JOB:binding or dict(BINDING)},
        wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})


def scan(sequence=1,count=0,complete=False):
    return {**validation(sequence,count,complete),'schema':'ovl.runtime-production-scan.v1',
            'pass_index':1,'operation':'stream-validation'}


def test_raw_observations_survive_restart_without_empty_duplicate_or_export_credit(tmp_path):
    c=Clock();path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=health(j,c);h.start_job(SELECTION);c.advance(30)
        assert not h.activity(JOB,scan());assert h.progress==h.exported==NOW
        c.advance(30);assert h.activity(JOB,scan(2,20))
        assert j.events[-1]['body']['detail']['observation']==scan(2,20)
        assert j.events[-1]['body']['kind']=='initialization-scan'
    with Journal(path).lease() as j:
        h=health(j,c);assert not h.start_job(SELECTION)
        c.advance(30);assert not h.activity(JOB,scan(2,20))
        assert not h.activity(JOB,scan(3,20));assert h.progress==NOW+60
        assert h.activity(JOB,scan(4,100,True));assert h.activity(JOB,activity(5))
        assert h.exported==NOW and not h.complete
        with pytest.raises(EvidenceError):h.activity(JOB,scan(6,100,True))
    with Journal(path).lease() as j:
        h=health(j,c);assert h.activities[JOB]['sequence']==5


@pytest.mark.parametrize('damage',['pass','bool-pass','operation','old-schema','root','total','prefix',
    'bool-prefix','premature','pid','instance','sequence','changed-repeat','extra'])
def test_changed_or_foreign_scan_cannot_poison_journal(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job(SELECTION);h.activity(JOB,scan(3,40));before=len(j.events)
        v=scan(4,50)
        if damage=='pass':v['pass_index']=2
        elif damage=='bool-pass':v['pass_index']=True
        elif damage=='operation':v['operation']='coverage-census'
        elif damage=='old-schema':v=validation(4,50)
        elif damage=='root':v['stream_sha256']='d'*64
        elif damage=='total':v['documents']=101
        elif damage=='prefix':v['completed_documents']=39
        elif damage=='bool-prefix':v['completed_documents']=True
        elif damage=='premature':v['complete']=True
        elif damage=='pid':v['pid']+=1
        elif damage=='instance':v['process_instance']='d'*32
        elif damage=='sequence':v['sequence']=2
        elif damage=='changed-repeat':v['sequence']=3
        else:v['heartbeat']=True
        with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert len(j.events)==before and h.exported==NOW


def test_initializer_action_binding_cannot_change_live_or_on_adoption(tmp_path):
    c=Clock();binding=dict(BINDING);path=tmp_path/'j'
    with Journal(path).lease() as j:
        h=health(j,c,binding);h.start_job(SELECTION);binding['action']='verify'
        with pytest.raises(EvidenceError):h.activity(JOB,scan())
        with pytest.raises(EvidenceError):h.activity(JOB,activity())
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):health(j,c,binding)


@pytest.mark.parametrize('damage',['old-event','export-credit','completion-credit','empty-credit','changed-pass'])
def test_adoption_rechecks_raw_scan_and_cost_decision(tmp_path,damage):
    c=Clock();path=tmp_path/'j'
    with Journal(path).lease() as j:
        h=health(j,c);h.start_job(SELECTION)
        body={'schema':'ovl.cost-activity-event.v2','kind':'initialization-scan','observed_epoch':NOW,
              'detail':{'job_sha256':JOB,'observation':scan()},'advances_progress':False,
              'advances_export':False,'completes':False}
        if damage=='old-event':
            body['kind']='stream-validation';body['detail']['observation']=validation()
        elif damage=='export-credit':body['advances_export']=True
        elif damage=='completion-credit':body['completes']=True
        elif damage=='empty-credit':body['advances_progress']=True
        else:body['detail']['observation']['pass_index']=2
        j.append('decision',body)
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):health(j,c)


@pytest.mark.parametrize('damage',['pid','sequence','instance'])
def test_warmup_cannot_change_scan_process_or_reuse_sequence(tmp_path,damage):
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c);h.start_job(SELECTION);h.activity(JOB,scan(3,40));before=len(j.events)
        value=activity(4)
        if damage=='pid':value['pid']+=1
        elif damage=='sequence':value['sequence']=3
        else:value['process_instance']='d'*32
        with pytest.raises(EvidenceError):h.activity(JOB,value)
        assert len(j.events)==before


def test_actual_initializer_scan_then_discarded_cpu_warmup(prepared,cpu_runtime,observations,tmp_path):
    from ovl_pipeline import initialization
    from ovl_pipeline.fixture import recipe
    directory,manifest=prepared;wiki=directory/'wikipedia';stream=read_json(wiki/'stream.json')
    result=initialization.record(wiki,recipe(manifest['tokenizer']['vocab_size']),
        {'schema':'ovl.gpu-kernel.v1','precision':'bf16'},tmp_path/'record',warmup_updates=2)
    assert result['control']['global_step']==0 and result['warmup_weights_discarded'] is True
    assert observations and observations[-1]['complete'] is True
    c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c,{**BINDING,'stream_sha256':digest(stream),'documents':stream['documents']})
        h.start_job(SELECTION)
        for observation in observations:h.activity(JOB,observation)
        # The CPU substitute does not emit GPU activity. Explicitly exercise the
        # same-process sequence transition; this is not an observed CUDA update.
        last=observations[-1]
        numeric={**activity(last['sequence']+1),'process_instance':last['process_instance'],'pid':last['pid']}
        assert h.activity(JOB,numeric)
        assert h.exported==NOW and not h.complete


def test_cycle_download_contract_and_restart_keep_initializer_protocol_separate(tmp_path):
    from test_sustained_health import BINDING as download_binding,transfer
    c=Clock();path=tmp_path/'cycle';download='f'*64
    def make(j,changed=False):
        b=dict(download_binding)
        if changed:b['bytes']+=1
        return InitializationCycleHealth(j,intent(),'owned-pod',{JOB:dict(BINDING)},{download:b},
            wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})
    with Journal(path).lease() as j:
        h=make(j);h.start_job({**SELECTION,'job_sha256':download,'kind':'setup'})
        c.advance(10);assert h.activity(download,transfer(1024**2))
        with pytest.raises(EvidenceError):h.activity(download,scan())
        assert h.exported==NOW and not h.complete
    with Journal(path).lease() as j:
        h=make(j);c.advance(20);assert not h.activity(download,transfer(1024**2))
        assert h.progress==NOW+10 and h.exported==NOW
    with Journal(path).lease() as j:
        with pytest.raises(EvidenceError):make(j,changed=True)
