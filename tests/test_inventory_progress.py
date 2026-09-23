"""Real file hashes with explicit slow-I/O clocks; no GPU or production credit."""
import copy
from types import SimpleNamespace
import pytest
from ovl_pipeline import canonical as can,data,inventory_activity as observed,runtime_activity
from ovl_pipeline import observed_validation,production_observation
from ovl_pipeline.canonical import EvidenceError,Merkle,canonical,digest,inventory,read_json,write_json
from ovl_pipeline.supervision import Journal
from pilot_health import PilotHealth
from initialization_health import InitializationHealth
from test_external_watchdog import intent,NOW
from test_workload_health import Clock,JOB,SELECTION,activity
from test_pipeline import prepared

@pytest.fixture
def large_stream(tmp_path):
    path=tmp_path/'stream';path.mkdir();n=5*1024**2
    (path/'tokens.u16').write_bytes(b'\x02\x00'*n);(path/'mask.u8').write_bytes(b'\x01'*n)
    row={'identity':{'fixture':'large-synthetic-stream'},'offset':0,'tokens':n,'target_start':0,'targets':n}
    (path/'documents.jsonl').write_bytes(canonical(row)+b'\n');tree=Merkle();tree.add(canonical(row))
    manifest={'schema':'ovl.stream.v1','phase':'wikipedia','token_dtype':'uint16-le','tokenizer_sha256':'a'*64,
        'documents':1,'tokens':n,'targets':n,'index_root':tree.root(),'window_policy':'per-document-overlap-one-v1',
        'files':inventory(path,['documents.jsonl','mask.u8','tokens.u16'])}
    write_json(path/'stream.json',manifest);return path,manifest

def message(manifest,sequence=1,count=0,**kw):
    return {'schema':'ovl.runtime-inventory-read.v1','process_instance':'b'*32,'pid':123,'sequence':sequence,
        'pass_index':1,'manifest':copy.deepcopy(manifest),'stream_sha256':digest(manifest),'read_bytes':count,
        'scope':'operator-supervision-only-not-training-verification',**kw}

def health(j,c,manifest,kind=PilotHealth):
    binding={'schema':'ovl.pilot-validation-binding.v1','stream_sha256':digest(manifest),'documents':manifest['documents']}
    if kind is InitializationHealth:binding.update(schema='ovl.initialization-validation-binding.v1',action='record')
    return kind(j,intent(),'owned-pod',{JOB:binding},wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})

@pytest.mark.parametrize('driver',[observed_validation,production_observation])
def test_full_cold_hash_reports_actual_chunks_before_rows_and_preserves_checks(large_stream,tmp_path,monkeypatch,driver):
    path,manifest=large_stream;c=Clock();events=[];oldhash=can.file_hash
    monkeypatch.setattr(observed,'time',SimpleNamespace(monotonic=lambda:c.now))
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'))
    monkeypatch.setattr(observed,'_pass_index',0);monkeypatch.setattr(production_observation,'_pass_index',0)
    for k,v in (('_sequence',0),('_last',None),('_process',None)):monkeypatch.setattr(runtime_activity,k,v)
    def slow_hash(path,*,progress=None):
        def chunk(count):
            c.advance(61)
            if progress:progress(count)
        return oldhash(path,progress=chunk)
    monkeypatch.setattr(can,'file_hash',slow_hash);original_emit=runtime_activity._emit
    def emit(v,**kw):
        original_emit(v,clock=lambda:c.now,force=True);events.append((c.now,read_json(tmp_path/'activity.json')))
    monkeypatch.setattr(runtime_activity,'_emit',emit)
    assert driver.validate_stream(path,manifest)==manifest['targets']
    hashes=[(t,e) for t,e in events if e['schema']=='ovl.runtime-inventory-read.v1']
    rows=[(t,e) for t,e in events if e['schema']!='ovl.runtime-inventory-read.v1']
    assert hashes and rows and rows[0][0]-NOW>300
    assert max(b-a for a,b in zip([NOW]+[t for t,e in hashes],[t for t,e in hashes]))<=61
    assert hashes[-1][1]['read_bytes']==sum(e['bytes'] for e in manifest['files'])
    assert rows[-1][1]['complete'] is True
    monkeypatch.delenv('OVL_ACTIVITY_FILE');assert data.validate_stream(path,manifest)==manifest['targets']
    assert [e['sequence'] for t,e in events]==list(range(1,len(events)+1))
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'));events.clear()
    with (path/'tokens.u16').open('r+b') as f:f.write(b'\x03\x00')
    with pytest.raises(EvidenceError,match='hash mismatch'):driver.validate_stream(path,manifest)
    assert events and not any(e.get('complete') for t,e in events)

@pytest.mark.parametrize('kind',[PilotHealth,InitializationHealth])
def test_slow_hash_adoption_trickle_and_stall_do_not_renew_export(large_stream,tmp_path,kind):
    _,manifest=large_stream;c=Clock();p=tmp_path/'journal';total=sum(e['bytes'] for e in manifest['files'])
    with Journal(p).lease() as j:
        h=health(j,c,manifest,kind);h.start_job(SELECTION)
        for i in range(1,7):
            c.advance(61);assert h.activity(JOB,message(manifest,i,i*1024**2))
        assert h.progress==NOW+366 and h.exported==NOW and not h.complete
    with Journal(p).lease() as j:
        h=health(j,c,manifest,kind);assert not h.start_job(SELECTION)
        for i in range(7,18):
            c.advance(31);assert not h.activity(JOB,message(manifest,i,6*1024**2+i))
        assert c.now-h.progress>300 and h.exported==NOW and not h.complete
        assert h.activity(JOB,message(manifest,18,total));assert not h.activity(JOB,message(manifest,19,total))
        assert h.activity(JOB,activity(20))
        with pytest.raises(EvidenceError):h.activity(JOB,message(manifest,21,total))

@pytest.mark.parametrize('damage',['root','manifest','over','negative','bool','pid','sequence','instance','pass','extra','scope'])
def test_foreign_or_changed_inventory_fails_before_journal_write(large_stream,tmp_path,damage):
    _,manifest=large_stream;c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c,manifest);h.start_job(SELECTION);h.activity(JOB,message(manifest,3,1024**2));n=len(j.events)
        v=message(manifest,4,2*1024**2)
        if damage=='root':v['stream_sha256']='f'*64
        elif damage=='manifest':v['manifest']['files'][0]['bytes']+=1
        elif damage=='over':v['read_bytes']=2**40
        elif damage=='negative':v['read_bytes']=-1
        elif damage=='bool':v['read_bytes']=True
        elif damage=='pid':v['pid']+=1
        elif damage=='sequence':v['sequence']=3
        elif damage=='instance':v['process_instance']='c'*32
        elif damage=='pass':v['pass_index']=2
        elif damage=='extra':v['complete']=True
        else:v['scope']='verification'
        with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert len(j.events)==n and h.exported==NOW

@pytest.mark.parametrize('damage',['export','completion','unearned-progress','wrong-binding'])
def test_adoption_rejects_altered_credit_or_identity(large_stream,tmp_path,damage):
    _,manifest=large_stream;c=Clock();p=tmp_path/'j'
    with Journal(p).lease() as j:
        h=health(j,c,manifest);h.start_job(SELECTION)
        body={'schema':'ovl.cost-activity-event.v2','kind':'inventory-read','observed_epoch':NOW,
              'detail':{'job_sha256':JOB,'observation':message(manifest)},'advances_progress':False,
              'advances_export':False,'completes':False}
        if damage=='export':body['advances_export']=True
        elif damage=='completion':body['completes']=True
        elif damage=='unearned-progress':body['advances_progress']=True
        else:body['detail']['observation']['stream_sha256']='f'*64
        j.append('decision',body)
    with Journal(p).lease() as j:
        with pytest.raises(EvidenceError):health(j,c,manifest)

def test_cache_reuse_still_hashes_every_file_and_fails_on_damage(large_stream,tmp_path,monkeypatch):
    path,manifest=large_stream;events=[]
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'));monkeypatch.setattr(observed,'_pass_index',0)
    monkeypatch.setattr(production_observation,'_pass_index',0)
    monkeypatch.setattr(runtime_activity,'_emit',lambda v,**kw:events.append(copy.deepcopy(v)))
    @production_observation.checked_stream_scope
    def run():
        assert production_observation.validate_stream(path,manifest)==manifest['targets']
        events.clear();assert production_observation.validate_stream(path,manifest)==manifest['targets']
        assert events and all(e['schema']=='ovl.runtime-inventory-read.v1' for e in events)
        with (path/'tokens.u16').open('r+b') as f:f.write(b'\x03\x00')
        with pytest.raises(EvidenceError,match='hash mismatch'):production_observation.validate_stream(path,manifest)
    run()

def test_generic_unbound_health_refuses_hash_protocol(large_stream,tmp_path):
    _,manifest=large_stream;c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=c.health(j);h.start_job(SELECTION)
        with pytest.raises(EvidenceError):h.activity(JOB,message(manifest,1,1024**2))

@pytest.mark.parametrize('damage',['pid','sequence','instance'])
@pytest.mark.parametrize('kind',[PilotHealth,InitializationHealth])
def test_hash_to_numerical_transition_rejects_changed_process_even_when_rows_missed(large_stream,tmp_path,kind,damage):
    _,manifest=large_stream;c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c,manifest,kind);h.start_job(SELECTION);h.activity(JOB,message(manifest,3,1024**2));n=len(j.events)
        v=activity(4)
        if damage=='pid':v['pid']+=1
        elif damage=='sequence':v['sequence']=3
        else:v['process_instance']='e'*32
        with pytest.raises(EvidenceError):h.activity(JOB,v)
        assert len(j.events)==n


def test_original_guard_still_stops_a_stalled_hash(large_stream,tmp_path):
    _,manifest=large_stream;c=Clock()
    with Journal(tmp_path/'j').lease() as j:
        h=health(j,c,manifest);h.start_job(SELECTION);h.activity(JOB,message(manifest,1,1024**2));c.advance(301)
        assert not h.activity(JOB,message(manifest,2,1024**2))
        value=h.write(tmp_path/'health.json')
        assert value['observed_epoch']-value['progress_epoch']==301
        assert value['exported_checkpoint_epoch']==NOW


def test_production_rehash_after_rows_requires_new_pass_and_survives_adoption(prepared,tmp_path):
    from test_production_health import bound
    from test_production_scan_health import observation,start
    from test_workload_stage import intent as production_intent
    from production_health import ProductionHealth
    control,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=production_intent()
    manifest=read_json(prepared[0]/'wikipedia/stream.json');total=sum(f['bytes'] for f in manifest['files']);p=tmp_path/'j'
    def make(j):return ProductionHealth(j,w,control.profile['pod_id'],r,bindings)
    with Journal(p).lease() as j:
        h=make(j);start(h,root);first=message(manifest,1,total,pid=42)
        assert h.activity(root,first)
        row=observation(r,sequence=2,completed_documents=1);assert h.activity(root,row)
        with pytest.raises(EvidenceError):h.activity(root,message(manifest,3,total,pid=42))
        second=message(manifest,3,0,pid=42,pass_index=2);assert not h.activity(root,second)
        second=message(manifest,4,total,pid=42,pass_index=2);assert h.activity(root,second)
        assert not h.complete
    with Journal(p).lease() as j:
        h=make(j);assert not h.activity(root,second)
        for changes in ({'pass_index':1},{'pass_index':33},{'pid':43},{'sequence':3},{'read_bytes':total+1}):
            with pytest.raises(EvidenceError):h.activity(root,{**second,'sequence':5,**changes})
        numeric=activity(5);numeric['pid']=42;numeric['control'].pop('pilot_cycle')
        assert h.activity(root,numeric)
        with pytest.raises(EvidenceError):h.activity(root,message(manifest,6,total,pid=42,pass_index=3))


def test_observer_pass_bound_and_no_progress_without_completed_read(large_stream,tmp_path,monkeypatch):
    _,manifest=large_stream;events=[]
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'));monkeypatch.setattr(observed,'_pass_index',31)
    monkeypatch.setattr(runtime_activity,'_emit',lambda v,**kw:events.append(v))
    callback=observed.observe(manifest);assert not events
    with pytest.raises(EvidenceError,match='pass limit'):observed.observe(manifest)
    with pytest.raises(EvidenceError):callback(1,0,False)
    assert not events


def test_hash_throttle_precedes_network_activity_path_checks(large_stream,monkeypatch,tmp_path):
    _,manifest=large_stream;events=[];clock=[0]
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'));monkeypatch.setattr(observed,'_pass_index',0)
    monkeypatch.setattr(observed,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    monkeypatch.setattr(runtime_activity,'_emit',lambda v,**kw:events.append(v))
    callback=observed.observe(manifest);first=manifest['files'][0]['bytes']
    callback(0,first,False);callback(0,first,True)
    for count in range(1,1001):callback(1,count,False)
    assert len(events)==1
    clock[0]=30;callback(1,1001,False)
    assert len(events)==2 and events[-1]['read_bytes']==first+1001

@pytest.mark.parametrize('kind',['pilot','initialization'])
def test_whole_run_dispatcher_routes_bound_hashes_and_adopts_them(large_stream,tmp_path,kind):
    from production_run_health import ProductionRunHealth
    _,manifest=large_stream;c=Clock();p=tmp_path/'j'
    b={'schema':'ovl.pilot-validation-binding.v1','stream_sha256':digest(manifest),'documents':manifest['documents']}
    if kind=='initialization':b.update(schema='ovl.initialization-validation-binding.v1',action='record')
    def make(j):return ProductionRunHealth(j,intent(),'owned-pod',None,{JOB:b},{},wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})
    with Journal(p).lease() as j:
        h=make(j);h.start_job(SELECTION);assert h.activity(JOB,message(manifest,1,1024**2))
    with Journal(p).lease() as j:
        h=make(j);assert not h.activity(JOB,message(manifest,1,1024**2))
        c.advance(61);assert h.activity(JOB,message(manifest,2,2*1024**2))
        assert h.exported==NOW and not h.complete
        with pytest.raises(EvidenceError):h.activity(JOB,message(manifest,3,1024**2,pass_index=2))

@pytest.mark.parametrize('kind',[PilotHealth,InitializationHealth])
@pytest.mark.parametrize('with_hash',[False,True])
def test_empty_row_start_gives_no_progress_credit_live_or_adopted(large_stream,tmp_path,kind,with_hash):
    from test_pilot_health import validation
    _,manifest=large_stream;c=Clock();p=tmp_path/'j'
    with Journal(p).lease() as j:
        h=health(j,c,manifest,kind);h.start_job(SELECTION)
        if with_hash:assert h.activity(JOB,message(manifest,1,1024**2))
        c.advance(100);v=validation(2,0);v.update(stream_sha256=digest(manifest),documents=manifest['documents'])
        if kind is InitializationHealth:v.update(schema='ovl.runtime-production-scan.v1',pass_index=1,operation='stream-validation')
        assert not h.activity(JOB,v) and h.progress==NOW
    with Journal(p).lease() as j:
        h=health(j,c,manifest,kind);assert not h.activity(JOB,v) and h.progress==NOW

@pytest.mark.parametrize('kind',['pilot','initialization','production'])
@pytest.mark.parametrize('damage',['regression','repeat-credit','same-sequence-changed','export','completion'])
def test_numerical_adoption_rechecks_latest_sequence_and_credit(prepared,tmp_path,kind,damage):
    from production_health import ProductionHealth
    from test_production_health import bound
    from test_workload_stage import intent as production_intent
    manifest=read_json(prepared[0]/'wikipedia/stream.json');c=Clock();p=tmp_path/'j'
    if kind=='production':
        control,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=production_intent();selected={**SELECTION,'job_sha256':root,'pod_id':control.profile['pod_id'],'kind':'production-record'}
        def make(j):return ProductionHealth(j,w,control.profile['pod_id'],r,bindings)
    else:
        root=JOB;selected=SELECTION
        def make(j):return health(j,c,manifest,PilotHealth if kind=='pilot' else InitializationHealth)
    with Journal(p).lease() as j:
        h=make(j);h.start_job(selected);h.activity(root,message(manifest,3,sum(e['bytes'] for e in manifest['files'])))
        v=activity(10)
        if kind=='production':v['control'].pop('pilot_cycle')
        assert h.activity(root,v);body=copy.deepcopy(j.events[-1]['body'])
        if damage=='regression':body['detail']['observation']['sequence']=9
        elif damage=='same-sequence-changed':body['detail']['observation']['control']['global_step']+=1
        elif damage=='export':body['detail']['observation']['sequence']=11;body['advances_progress']=False;body['advances_export']=True
        elif damage=='completion':body['detail']['observation']['sequence']=11;body['advances_progress']=False;body['completes']=True
        j.append('decision',body)
    with Journal(p).lease() as j:
        with pytest.raises(EvidenceError):make(j)


def test_row_hash_row_same_pass_rejected_live_and_on_adoption(prepared,tmp_path):
    from test_production_health import bound
    from test_production_scan_health import observation,start
    from test_workload_stage import intent as production_intent
    from production_health import ProductionHealth
    control,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=production_intent();p=tmp_path/'j'
    manifest=read_json(prepared[0]/'wikipedia/stream.json');total=sum(e['bytes'] for e in manifest['files'])
    def make(j):return ProductionHealth(j,w,control.profile['pod_id'],r,bindings)
    with Journal(p).lease() as j:
        h=make(j);start(h,root);assert h.activity(root,observation(r,sequence=10,completed_documents=1))
        assert h.activity(root,message(manifest,11,total,pid=42,pass_index=2))
        bad=observation(r,sequence=12,completed_documents=2)
        with pytest.raises(EvidenceError,match='row pass reopened'):h.activity(root,bad)
        good={**bad,'pass_index':2};assert h.activity(root,good)
        # A fresh rehash then a forged replay reopening this same row pass.
        assert h.activity(root,message(manifest,13,total,pid=42,pass_index=3))
        body={'schema':'ovl.cost-activity-event.v2','kind':'production-scan','observed_epoch':h.now(),
            'detail':{'job_sha256':root,'observation':{**good,'sequence':14}},'advances_progress':False,
            'advances_export':False,'completes':False};j.append('decision',body)
    with Journal(p).lease() as j:
        with pytest.raises(EvidenceError,match='row pass reopened'):make(j)

@pytest.mark.parametrize('kind',['pilot','initialization','production'])
def test_actual_sender_normal_throttle_missed_snapshots_and_adopted_receiver(large_stream,prepared,tmp_path,monkeypatch,kind):
    from production_health import ProductionHealth
    from test_production_health import bound
    from test_workload_stage import intent as production_intent
    path,manifest=large_stream;c=Clock();journal=tmp_path/'integrated-journal';counts={'snapshots':0,'accepted':0}
    if kind=='production':
        ctl,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=production_intent()
        # Explicit health-only fixture: controller selects the actual large
        # synthetic manifest. This is not a signed training registration test.
        r['coverage']['wikipedia'].update(stream_sha256=digest(manifest),documents=manifest['documents'])
        c.now=w['plan']['input']['now_epoch']
        selected={**SELECTION,'job_sha256':root,'pod_id':ctl.profile['pod_id'],'kind':'production-record'}
        def make(j):return ProductionHealth(j,w,ctl.profile['pod_id'],r,bindings,wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})
    else:
        root=JOB;selected=SELECTION
        def make(j):return health(j,c,manifest,PilotHealth if kind=='pilot' else InitializationHealth)
    start_epoch=c.now
    with Journal(journal).lease() as j:make(j).start_job(selected)
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'))
    monkeypatch.setattr(observed,'_pass_index',0);monkeypatch.setattr(production_observation,'_pass_index',0)
    monkeypatch.setattr(observed,'time',SimpleNamespace(monotonic=lambda:c.now))
    for k,v in (('_sequence',0),('_last',None),('_process',None)):monkeypatch.setattr(runtime_activity,k,v)
    original_hash=can.file_hash;original_emit=runtime_activity._emit;last_sequence=[0]
    def slow_hash(p,*,progress=None):
        def completed(n):
            c.advance(61)
            if progress:progress(n)
        return original_hash(p,progress=completed)
    monkeypatch.setattr(can,'file_hash',slow_hash)
    def emit(value,**kw):
        original_emit(value,clock=lambda:c.now,force=kw.get('force',False))
        snapshot=read_json(tmp_path/'activity.json')
        if snapshot['sequence']==last_sequence[0]:return
        last_sequence[0]=snapshot['sequence'];counts['snapshots']+=1
        if snapshot['schema']=='ovl.runtime-inventory-read.v1' and counts['snapshots']%2:return
        with Journal(journal).lease() as j:
            h=make(j);counts['accepted']+=int(h.activity(root,snapshot))
            assert c.now-h.progress<300 and h.exported==start_epoch and not h.complete
    monkeypatch.setattr(runtime_activity,'_emit',emit)
    driver=observed_validation if kind=='pilot' else production_observation
    assert driver.validate_stream(path,manifest)==manifest['targets']
    c.advance(31);v=activity()['control']
    if kind=='production':v.pop('pilot_cycle')
    runtime_activity.update(v)
    with Journal(journal).lease() as j:
        h=make(j);assert h.activities[root]['control']==v and h.exported==start_epoch and not h.complete
    assert counts['snapshots']>counts['accepted']>=3 and c.now-start_epoch>300

@pytest.mark.parametrize('kind',['pilot','initialization','production','whole-run'])
def test_adoption_cannot_switch_to_forged_wrapper_phases_after_numerical_work(prepared,tmp_path,kind):
    from production_health import ProductionHealth
    from production_run_health import ProductionRunHealth
    from test_production_health import bound
    from test_workload_stage import intent as production_intent
    manifest=read_json(prepared[0]/'wikipedia/stream.json');c=Clock();p=tmp_path/'j'
    if kind in ('production','whole-run'):
        ctl,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=production_intent();selected={**SELECTION,'job_sha256':root,'pod_id':ctl.profile['pod_id'],'kind':'production-record'}
        def make(j):return (ProductionHealth(j,w,ctl.profile['pod_id'],r,bindings) if kind=='production' else ProductionRunHealth(j,w,ctl.profile['pod_id'],r,bindings,{}))
    else:
        root=JOB;selected=SELECTION
        def make(j):return health(j,c,manifest,PilotHealth if kind=='pilot' else InitializationHealth)
    with Journal(p).lease() as j:
        h=make(j);h.start_job(selected);h.activity(root,message(manifest,3,sum(e['bytes'] for e in manifest['files'])))
        v=activity(10)
        if kind in ('production','whole-run'):v['control'].pop('pilot_cycle')
        assert h.activity(root,v)
        phase={'schema':'ovl.audited-pilot-phases.v1','process_instance':'b'*32,'pid':123,'sequence':11,
            'completed':[{'phase':'record','report_sha256':'e'*64}], 'scope':'operator-supervision-only-not-training-verification'}
        body={'schema':'ovl.cost-activity-event.v2','kind':'pilot-phases','observed_epoch':h.now(),
            'detail':{'job_sha256':root,'observation':phase},'advances_progress':True,'advances_export':True,'completes':True}
        j.append('decision',body)
    with Journal(p).lease() as j:
        with pytest.raises(EvidenceError):make(j)

@pytest.mark.parametrize('damage',['none','duplicate','export','completion','extra'])
def test_generic_wrapper_phase_replay_preserves_valid_protocol_only(tmp_path,damage):
    c=Clock();p=tmp_path/'j'
    phase={'schema':'ovl.audited-pilot-phases.v1','process_instance':'b'*32,'pid':123,
        'completed':[{'phase':'record','report_sha256':'e'*64}], 'scope':'operator-supervision-only-not-training-verification'}
    with Journal(p).lease() as j:
        h=c.health(j);h.start_job(SELECTION);assert h.activity(JOB,phase)
        if damage!='none':
            body=copy.deepcopy(j.events[-1]['body'])
            if damage=='export':body['advances_progress']=False;body['advances_export']=True
            elif damage=='completion':body['advances_progress']=False;body['completes']=True
            elif damage=='extra':body['detail']['observation']['sequence']=3
            j.append('decision',body)
    with Journal(p).lease() as j:
        if damage=='none':
            h=c.health(j);assert not h.activity(JOB,phase) and not h.complete and h.exported==NOW
        else:
            with pytest.raises(EvidenceError):c.health(j)

@pytest.mark.parametrize('whole',[False,True])
@pytest.mark.parametrize('rows_seen',[False,True])
@pytest.mark.parametrize('damage',['registration','worker'])
def test_production_numeric_contract_drift_rejected_before_append(prepared,tmp_path,whole,rows_seen,damage):
    from production_health import ProductionHealth
    from production_run_health import ProductionRunHealth
    from test_production_health import bound
    from test_production_scan_health import observation,start
    from test_workload_stage import intent as production_intent
    ctl,record,job,root,worker,r,bindings=bound(prepared,tmp_path);w=production_intent()
    manifest=read_json(prepared[0]/'wikipedia/stream.json')
    with Journal(tmp_path/'j').lease() as j:
        h=(ProductionRunHealth(j,w,ctl.profile['pod_id'],r,bindings,{}) if whole else ProductionHealth(j,w,ctl.profile['pod_id'],r,bindings));start(h,root)
        h.activity(root,message(manifest,1,sum(e['bytes'] for e in manifest['files']),pid=42))
        if rows_seen:h.activity(root,observation(r,sequence=2,completed_documents=1))
        if damage=='registration':r['coverage']['wikipedia']['documents']+=1
        else:bindings[root]['worker_sha256']='f'*64
        before=len(j.events);v=activity(3);v['pid']=42;v['control'].pop('pilot_cycle')
        with pytest.raises(EvidenceError):h.activity(root,v)
        assert len(j.events)==before


def test_mutating_caller_snapshot_cannot_change_retained_or_adopted_progress(large_stream,tmp_path):
    _,manifest=large_stream;c=Clock();p=tmp_path/'j';v=message(manifest,1,1024**2)
    with Journal(p).lease() as j:
        h=health(j,c,manifest);h.start_job(SELECTION);assert h.activity(JOB,v)
        v.update(sequence=2,read_bytes=2*1024**2)
        assert h.inventory.reads[JOB]['sequence']==1 and j.events[-1]['body']['detail']['observation']['sequence']==1
        c.advance(31);assert h.activity(JOB,v)
        expected=copy.deepcopy(h.inventory.reads[JOB]);v['manifest']['files'][0]['bytes']+=1
        assert h.inventory.reads[JOB]==expected
    with Journal(p).lease() as j:
        h=health(j,c,manifest);assert h.inventory.reads[JOB]==expected and h.inventory.credited[JOB]==2*1024**2
        assert h.progress==NOW+31 and h.exported==NOW
