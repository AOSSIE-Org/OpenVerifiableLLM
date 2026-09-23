"""Synthetic retained-storage refusals; no provider or service mutations."""
import copy
from decimal import Decimal
import pytest
import reconcile_capacity_rejection as rejection
import recover_capacity_refusal as recovery
import test_terminal_capacity_refusal as legacy
from test_reconcile_capacity_rejection import fixture
from test_retained_network_volume import volume_intent, observation
from ovl_pipeline.canonical import EvidenceError, digest, canonical


def selected_observation(value, watch):
    value=observation(value,watch['retained_volume'])
    value.update(schema='ovl.retained-volume-account-identity-observation.v1',
                 retained_volume_sha256=digest(watch['retained_volume']))
    return value


def volume_fixture():
    args=fixture();r=volume_intent();w=r['watchdog_intent'];args[0]=r;args[1]=digest(r)
    args[3]['variables_sha256']=digest({'input':r['payload']})
    args[2]['variables_sha256']=args[3]['variables_sha256'];args[5][0]['body']=r
    args[6]=[selected_observation(o,w) for o in args[6]]
    args[7].update(intent_sha256=digest(w),plan_sha256=digest(w['plan']),external_terminate_epoch=w['plan']['external_terminate_epoch'])
    return args


def ready():
    args=volume_fixture();r=args[0];w=r['watchdog_intent'];now=args[-1]
    fence={'schema':'ovl.one-shot-creation-fence.v1','intent_sha256':digest(r),'attempt_id':r['payload']['name']}
    args[5][1]['body']['fence_sha256']=digest(fence)
    # Complete histories contain the selected billed volume, with no compute.
    args[5].append({'kind':'provider-observation','body':copy.deepcopy(w['baseline'])})
    args[5]=legacy.chain(args[5],w['plan']['input']['now_epoch'])
    receipt=rejection.verify(*args)
    watch=legacy.chain([{'kind':'creation-intent','body':w},{'kind':'provider-observation','body':copy.deepcopy(w['baseline'])}],w['plan']['input']['now_epoch'])
    current=now+30
    def reads(times):return [selected_observation(x,w) for x in legacy.observations(receipt['account_identity_sha256'],times)]
    before={'controller':args[5],'watchdog':watch,'fence':fence,'observations':reads([now+10,current]),'heartbeat':{**args[7],'observed_epoch':current},'now':current}
    prep=recovery.prepare(args,receipt,**before)
    return {'original_args':args,'receipt':receipt,'before':before,'preparation':prep,'controller':args[5],'watchdog':watch,'fence':fence,'observations':reads([current+2,current+22]),'service_states':{'controller':'inactive','watchdog':'inactive'},'stopped_epoch':current+1,'now':current+22}


def test_exact_retained_baseline_preserves_storage_and_releases_only_unused_compute():
    data=ready();v=recovery.finalize(**data);s=data['original_args'][0]['watchdog_intent']['retained_volume']
    assert v['schema']=='ovl.terminal-capacity-refusal-closure.v2'
    assert v['retained_volume_sha256']==digest(s) and v['retained_storage_reserved_usd']==s['reserved_usd']
    assert v['storage_action']=='RETAIN_UNCHANGED_NO_STORAGE_RESERVATION_RELEASE'
    b={'current_rental_intent_sha256':v['rental_intent_sha256'],'current_rental_projected_maximum_usd':v['eligible_unused_creation_allowance_usd'],'actual_project_spend':'7','prior_unsettled_reservation_usd':'5','retained_volume_reserved_usd':s['reserved_usd'],'remaining_mandatory_reservation_usd':s['reserved_usd'],'protected_reserve_usd':'10'}
    changed,releases=recovery.release_budget(b,{},v,canonical(v),closure_inputs=data)
    assert changed=={**b,'current_rental_projected_maximum_usd':'0'}
    assert recovery.release_budget(changed,releases,v,canonical(v),closure_inputs=data)==(changed,releases)


def damage(o,kind):
    if kind=='pod':o['pods']=[{'id':'synthetic-late-pod'}]
    elif kind=='missing':o['volume_ids']=[];o['network_volumes']=[]
    elif kind=='extra':o['volume_ids'].append('synthetic-other')
    elif kind=='duplicate':o['volume_ids']*=2;o['network_volumes']*=2
    elif kind=='name':o['network_volumes'][0]['name']='synthetic-other'
    elif kind=='size':o['network_volumes'][0]['size']+=1
    elif kind=='location':o['network_volumes'][0]['dataCenterId']='EUR-NO-1'
    elif kind=='metadata':o.pop('network_volumes')
    elif kind=='rate':o['account_hourly_usd']='0.023335'
    elif kind=='nan':o['account_hourly_usd']='NaN'
    elif kind=='autopay':o['autopay']=True
    elif kind=='identity':o['account_identity_sha256']='0'*64
    else:o['retained_volume_sha256']='0'*64


@pytest.mark.parametrize('kind',['pod','missing','extra','duplicate','name','size','location','metadata','rate','nan','autopay','identity','selection'])
@pytest.mark.parametrize('stage',['rejection','preparation','final'])
def test_changed_baseline_never_closes(kind,stage):
    d=ready()
    if stage=='rejection':
        args=d['original_args'];damage(args[6][1],kind)
        with pytest.raises(EvidenceError):rejection.verify(*args)
    elif stage=='preparation':
        damage(d['before']['observations'][1],kind)
        with pytest.raises(EvidenceError):recovery.prepare(d['original_args'],d['receipt'],**d['before'])
    else:
        damage(d['observations'][1],kind)
        with pytest.raises(EvidenceError):recovery.finalize(**d)


@pytest.mark.parametrize('role',['controller','watchdog'])
@pytest.mark.parametrize('kind',['pod','missing','extra','duplicate','name','size','location','metadata','rate','nan','autopay'])
def test_changed_historical_baseline_never_closes(role,kind):
    d=ready();events=copy.deepcopy(d[role]);o=copy.deepcopy(events[-1]['body']);damage(o,kind)
    events.append({'kind':'provider-observation','body':o})
    d[role]=legacy.chain([{'kind':e['kind'],'body':e['body']} for e in events],d['now'])
    # The altered later observation is correctly chained; original prefix remains.
    for i,e in enumerate(d[role][:len(events)-1]):d[role][i]=d['before'][role][i]
    last=d[role][-1];last['previous']=digest(d[role][-2])
    with pytest.raises(EvidenceError):recovery.finalize(**d)


@pytest.mark.parametrize('test_name',[
 'test_execute_and_resume_preserve_journals_fence_and_no_double_actions',
 'test_failure_after_stop_restores_watchdog_and_never_produces_closure',
 'test_competing_account_owner_cannot_be_displaced',
 'test_supervisor_restores_watchdog_after_unsealed_interruption',
 'test_durable_teardown_recovers_missing_final_file_and_supervisor_checks_evidence'])
def test_existing_recovery_lifecycle_with_retained_volume(tmp_path,monkeypatch,test_name):
    monkeypatch.setattr(legacy,'ready',ready)
    getattr(legacy,test_name)(tmp_path,monkeypatch)


def test_default_collector_receives_original_retained_selection(tmp_path,monkeypatch):
    monkeypatch.setattr(legacy,'ready',ready)
    d,attempt,fences=legacy.setup_disk(tmp_path,monkeypatch);w=d['original_args'][0]['watchdog_intent'];reads=iter(d['before']['observations']+d['observations']);seen=[]
    def capture(selected):assert selected==w;seen.append(selected);return next(reads)
    monkeypatch.setattr(recovery,'capture',capture);monkeypatch.setattr(recovery.time,'sleep',lambda _:None)
    times=iter([d['before']['now'],d['before']['now'],d['stopped_epoch'],d['now']])
    got=recovery.recover(attempt,digest(d['original_args'][0]),tmp_path/'out',legacy.FakeGuards(),execute=True,wall=lambda:next(times),fences=fences)
    assert got==recovery.finalize(**d) and len(seen)==4


@pytest.mark.parametrize('bad',[None,'extra','fraction','nan','bool','rate','autopay','pod'])
def test_capture_checks_actual_selected_storage_without_erasing_it(monkeypatch,bad):
    import probe_provider_deadline as provider
    w=volume_intent()['watchdog_intent'];s=w['retained_volume']
    raw={'id':'synthetic-account','pods':[],'networkVolumes':[{'id':s['id'],'name':s['name'],'size':Decimal(224),'dataCenterId':s['data_center_id']}],'isAutoPayEnabled':False,'currentSpendPerHr':Decimal('.022')}
    if bad=='extra':raw['networkVolumes'].append({**raw['networkVolumes'][0],'id':'other'})
    elif bad in ('fraction','nan','bool'):raw['networkVolumes'][0]['size']={'fraction':Decimal('224.1'),'nan':Decimal('NaN'),'bool':True}[bad]
    elif bad=='rate':raw['currentSpendPerHr']=Decimal('.024')
    elif bad=='autopay':raw['isAutoPayEnabled']=True
    elif bad=='pod':raw['pods']=[{}]
    original=provider.OPERATIONS['account'];clock={'server_epoch':100,'request_started_epoch':100,'request_completed_epoch':100}
    monkeypatch.setattr(provider,'request',lambda name:({'myself':raw},'b'*64,clock))
    if bad:
        with pytest.raises(EvidenceError):rejection.capture(w)
    else:
        got=rejection.capture(w);assert got['network_volumes'][0]['size']==224 and got['volume_ids']==[s['id']] and got['account_hourly_usd']=='0.022'
        with pytest.raises(EvidenceError):rejection.capture()
    assert provider.OPERATIONS['account']==original


@pytest.mark.parametrize('boundary',['retention_deadline_epoch','billing_ceiling_epoch'])
def test_expired_storage_cannot_be_reconciled_or_closed_from_saved_preparation(boundary):
    d=ready();w=d['original_args'][0]['watchdog_intent'];future=w['retained_volume'][boundary]+3600
    a=copy.deepcopy(d['original_args']);a[-1]=future
    with pytest.raises(EvidenceError,match='storage recovery window'):rejection.verify(*a)
    d['stopped_epoch']=future;d['now']=future+22
    for o,t in zip(d['observations'],[future+1,future+22]):
        o['observed_epoch']=t;o['http_clock']={k:t for k in o['http_clock']}
    with pytest.raises(EvidenceError,match='storage recovery window'):recovery.finalize(**d)


def test_expired_saved_preparation_keeps_watchdog_protection(tmp_path,monkeypatch):
    from ovl_pipeline.canonical import write_json
    monkeypatch.setattr(legacy,'ready',ready)
    d,attempt,fences=legacy.setup_disk(tmp_path,monkeypatch);output=tmp_path/'recovery';output.mkdir(mode=0o700)
    write_json(output/'before.json',d['before']);write_json(output/'preparation.json',d['preparation'])
    guards=legacy.FakeGuards();future=d['original_args'][0]['watchdog_intent']['retained_volume']['retention_deadline_epoch']
    with pytest.raises(EvidenceError,match='storage recovery window'):
        recovery.recover(attempt,digest(d['original_args'][0]),output,guards,execute=True,
                         collect=lambda:pytest.fail('expired preparation must not collect'),wall=lambda:future,fences=fences)
    assert guards.actions==[('stop','controller'),('restore','watchdog')]
    assert guards.states['watchdog']=='active' and not(output/'closure.json').exists()


@pytest.mark.parametrize('field',['retained_volume_reserved_usd','remaining_mandatory_reservation_usd'])
@pytest.mark.parametrize('value',[None,'0','4.480127'])
def test_current_storage_reserve_must_cover_closed_selection(field,value):
    d=ready();v=recovery.finalize(**d)
    b={'current_rental_intent_sha256':v['rental_intent_sha256'],'current_rental_projected_maximum_usd':v['eligible_unused_creation_allowance_usd'],'retained_volume_reserved_usd':'4.480128','remaining_mandatory_reservation_usd':'4.480128'}
    if value is None:b.pop(field)
    else:b[field]=value
    with pytest.raises(EvidenceError):recovery.release_budget(b,{},v,canonical(v),closure_inputs=d)


@pytest.mark.parametrize('with_volume',[False,True])
def test_transient_heartbeat_requires_a_new_ordinary_observation(with_volume):
    args=volume_fixture() if with_volume else fixture();h=args[7];h['provider_observed_epoch']=args[-1]-5
    with pytest.raises(EvidenceError,match='heartbeat fields'):rejection.verify(*args)
    # An actual subsequent ordinary heartbeat is accepted; do not strip the
    # retained provider evidence in a caller to manufacture this observation.
    args[7]={k:v for k,v in h.items() if k!='provider_observed_epoch'}
    assert rejection.verify(*args)['billing']=='NOT_SETTLED_NO_CEILING_RELEASE'
    d=ready() if with_volume else legacy.ready();before=copy.deepcopy(d['before']);before['heartbeat']['provider_observed_epoch']=before['now']-5
    with pytest.raises(EvidenceError,match='heartbeat fields'):recovery.prepare(d['original_args'],d['receipt'],**before)
    assert recovery.prepare(d['original_args'],d['receipt'],**d['before'])==d['preparation']


def test_predecessor_empty_account_receipt_bytes_remain_compatible():
    # Complete synthetic receipt digest measured using predecessor8ece870's
    # verifier and pytest synthetic authorization fixture; no provider data or keys.
    assert digest(rejection.verify(*fixture()))=='8295560cf3d21ab41c2e6e65e9af8d668e8bc421060128018ccb978d3d1ca484'
