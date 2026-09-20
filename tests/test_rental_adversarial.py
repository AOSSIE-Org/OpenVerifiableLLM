"""Executed counterexamples from ordinary Opus advice; deterministic provider only."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from test_rental_controller import RentalFake
from test_external_watchdog import intent,Fake,NOW
import run_rental_controller as controller
import run_external_watchdog as watchdog
from rental_safety import Lifetime,account_lease
from probe_provider_deadline import ProviderFailure
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.supervision import Journal


@pytest.mark.parametrize('debit,early',[('0.150000',False),('0.235212',False),('0.235213',True)])
def test_prior_settlement_and_rental_ceiling_microdollar_boundary(tmp_path,debit,early):
    from decimal import Decimal
    f=RentalFake(tmp_path);original=f.account
    def settled():
        if f.alive:f.balance=str(Decimal(100)-Decimal(debit))
        return original()
    f.account=settled;f.run();first=next(c for c in f.calls if c[0]=='terminate')
    assert (first[2]==NOW)==early
    assert f.writes==1 and not f.alive


def test_another_journal_cannot_repeat_same_creation_even_when_listing_lags(tmp_path):
    f=RentalFake(tmp_path);f.create_crash=True
    with pytest.raises(KeyboardInterrupt):f.run()
    assert f.writes==1 and f.alive
    old=f.directory;f.directory=tmp_path/'second-journal';f.create_crash=False
    f.run();assert f.writes==1 and not f.alive
    assert (old/'event-00000000.json').exists()


def test_account_lease_serializes_different_journals(tmp_path):
    f=RentalFake(tmp_path)
    with account_lease(tmp_path/'fences'):
        with pytest.raises(EvidenceError,match='account lease'):f.run()
    assert f.calls==[]


@pytest.mark.parametrize('who',['controller','watchdog'])
def test_duplicate_uuid_names_all_terminated_but_unrelated_name_untouched(tmp_path,who):
    f=RentalFake(tmp_path);f.alive=True;ids={'owned-pod','owned-duplicate','unrelated'};calls=[]
    def provider(operation,variables=None):
        assert operation!='create'
        if operation=='identities':return {'myself':{'pods':[{'id':n,'name':f.i['payload']['name'] if n!='unrelated' else 'other-project'} for n in sorted(ids)]}},'a'*64,{}
        assert operation=='terminate';target=variables['input']['podId'];calls.append(target);ids.discard(target);return {},'b'*64,{}
    original=f.account
    def observation():
        obs=original();obs['pods']=[{**f.pod(),'id':n,'name':f.i['payload']['name'] if n!='unrelated' else 'other-project'} for n in sorted(ids)]
        return obs
    path=tmp_path/'existing'
    with Journal(path).lease() as j:
        j.append('creation-intent',f.value if who=='controller' else f.i)
        Lifetime(j,f.i['plan'],wall=lambda:NOW,clock=lambda:{'boot_id':'fake-boot','boottime_ms':0},initialize=True)
    kw=dict(get_account=observation,provider_request=provider,wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep)
    if who=='controller':controller.run(path,f.value,digest(f.value),f.heartbeat,f.health,**kw,boot=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)},fence_root=tmp_path/'fences')
    else:watchdog.run(path,f.i,digest(f.i),**kw,clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
    assert set(calls)=={'owned-pod','owned-duplicate'} and ids=={'unrelated'}
    assert read_json(path/'result.json')['complete']


@pytest.mark.parametrize('who',['controller','watchdog'])
@pytest.mark.parametrize('error',[ProviderFailure('http',status=503,transient=True),ProviderFailure('http',status=401,transient=False)])
def test_transient_read_survives_but_auth_failure_aborts(tmp_path,who,error):
    f=RentalFake(tmp_path);f.alive=who=='watchdog';original=f.account;injected=[False]
    def faulty():
        if f.now>=NOW+100 and not injected[0]:injected[0]=True;raise error
        return original()
    f.account=faulty
    if who=='controller':f.run();calls=f.calls
    else:
        f.calls=[]
        watchdog.run(tmp_path/'watchdog',f.i,digest(f.i),get_account=f.account,provider_request=f.provider,
                     wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,
                     clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
        calls=f.calls
    first=next(c for c in calls if c[0]=='terminate')
    assert first[2]>NOW+100 if error.transient else first[2]==NOW+100
    assert not f.alive


def test_clock_deadline_not_regranted_across_restart_and_backward_wall_step(tmp_path):
    f=RentalFake(tmp_path);f.create_crash=True
    with pytest.raises(KeyboardInterrupt):f.run()
    f.create_crash=False;f.now=NOW+100;f.elapsed=700
    f.run()
    first=next(c for c in f.calls if c[0]=='terminate')
    # Only20s remains on the original720s BOOTTIME lease; old code grants620s.
    assert first[2]<=NOW+120 and f.elapsed<=740


def test_boot_change_aborts_instead_of_renewing_persisted_deadline(tmp_path):
    f=RentalFake(tmp_path);f.create_crash=True
    with pytest.raises(KeyboardInterrupt):f.run()
    controller.run(f.directory,f.value,digest(f.value),f.heartbeat,f.health,get_account=f.account,
        provider_request=f.provider,wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,
        boot=lambda:{'boot_id':'different-boot','boottime_ms':0},fence_root=tmp_path/'fences')
    assert next(c for c in f.calls if c[0]=='terminate')[2]==NOW


def test_corrupt_rental_journal_attempts_cleanup_and_preserves_evidence(tmp_path):
    f=RentalFake(tmp_path);f.alive=True;f.directory.mkdir();event=f.directory/'event-00000000.json';event.write_bytes(b'damaged')
    with pytest.raises(EvidenceError):
        controller.run_guarded(f.directory,f.value,digest(f.value),f.heartbeat,f.health,get_account=f.account,
            provider_request=f.provider,wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,fence_root=tmp_path/'fences')
    assert not f.alive and event.read_bytes()==b'damaged'


@pytest.mark.parametrize('mode',['pretty-json','future-timestamp','disappearing-file'])
def test_bad_health_requests_graceful_stop_without_immediate_kill(tmp_path,mode,monkeypatch):
    f=RentalFake(tmp_path);original=f.refresh
    def invalid():
        original()
        if f.alive:
            h=read_json(f.health)
            if mode=='pretty-json':
                import json
                f.health.write_text(json.dumps(h,indent=2))
            elif mode=='future-timestamp':h['observed_epoch']+=1;write_json(f.health,h)
    f.refresh=invalid
    if mode=='disappearing-file':
        actual=controller.read_json
        def gone(path):
            if path==f.health:raise FileNotFoundError('simulated replacement gap')
            return actual(path)
        monkeypatch.setattr(controller,'read_json',gone)
    f.run();stop=read_json(f.directory/'stop-request.json')
    assert stop['observed_epoch']==NOW and 'invalid-workload-export-health' in stop['reasons']
    assert next(c for c in f.calls if c[0]=='terminate')[2]==NOW+300


@pytest.mark.parametrize('change',['quote-root','authorization-root','underpriced','raw-catalog-bytes','selected-price','stale-catalog','storage-price'])
def test_quote_and_authorization_links_fail_before_provider_mutation(tmp_path,change):
    from ovl_pipeline.supervision import rental_plan
    f=RentalFake(tmp_path);v=dict(f.i['plan']['input']);q=f.value['quote']
    if change=='quote-root':v['quote_sha256']='a'*64
    if change=='authorization-root':v['authorization_sha256']='b'*64
    if change=='underpriced':v['hourly_upper_usd']='0.01'
    if change=='raw-catalog-bytes':q['catalog_response']+=' '
    if change=='selected-price':q['selected_gpu']['secure_hourly_usd']='0.01'
    if change=='stale-catalog':q['observed_epoch']-=601
    if change=='storage-price':q['container_gb_month_usd']='0'
    if change!='quote-root':v['quote_sha256']=digest(q)
    f.i['plan']=rental_plan(v);f.refresh()
    with pytest.raises(EvidenceError):f.run()
    assert f.calls==[]


def test_residual_volume_is_observed_and_reported_not_deleted(tmp_path):
    f=RentalFake(tmp_path);original=f.account
    def volume():
        o=original()
        if f.writes:o['volume_ids']=['unrelated-volume']
        return o
    f.account=volume;f.run()
    assert read_json(f.directory/'result.json')['residual_network_volumes']==['unrelated-volume']
    assert not f.alive and all(c[0] in ('create','terminate','identities') for c in f.calls)


def test_transient_grace_cannot_cross_absolute_deadline(tmp_path):
    f=RentalFake(tmp_path);f.alive=True;f.now=f.i['plan']['external_terminate_epoch']-20;f.elapsed=700
    def down():raise ProviderFailure('http',status=503,transient=True)
    path=tmp_path/'watchdog'
    with Journal(path).lease() as j:
        j.append('creation-intent',f.i);j.append('creation-observed',{'id':'owned-pod'})
        Lifetime(j,f.i['plan'],wall=lambda:NOW,clock=lambda:{'boot_id':'fake-boot','boottime_ms':0},initialize=True)
    original=f.provider
    def recovered(operation,variables=None):
        result=original(operation,variables)
        if operation=='terminate':f.account=RentalFake.account.__get__(f,RentalFake)
        return result
    f.account=down
    watchdog.run(path,f.i,digest(f.i),get_account=lambda:f.account(),provider_request=recovered,
        wall=lambda:f.now,monotonic=lambda:f.elapsed,sleep=f.sleep,
        clock=lambda:{'boot_id':'fake-boot','boottime_ms':int(f.elapsed*1000)})
    assert next(c for c in f.calls if c[0]=='terminate')[2]==f.i['plan']['external_terminate_epoch']
