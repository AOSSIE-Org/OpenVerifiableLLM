"""Operational guard test controller checks; no provider mutations."""
import importlib.util
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from probe_provider_deadline import make_intent,match_pod,verdict,GPU
from ovl_pipeline.canonical import EvidenceError

NOW=1789688266
IMAGE='ubuntu@sha256:'+'1'*64

def intent():
    return make_intent(NOW,{'pods':[],'volume_ids':[],'autopay':False,'account_hourly_usd':'0','balance_usd':'100.50','http_clock':{'server_epoch':NOW,'request_started_epoch':NOW,'request_completed_epoch':NOW}},
        {'observed_epoch':NOW,'gpus':[{'id':GPU,'secure':True,'availability':'LOW','price':{'secure':'0.24'}}]},IMAGE)

def pod(i):
    return {'id':'test-pod','name':i['payload']['name'],'imageName':IMAGE,'gpuCount':1,'createdAt':'2026-09-17T23:37:46Z','containerDiskInGb':4,'volumeInGb':0}

def test_probe_funds_reserve_full_remaining_work():
    i=intent();p=i['plan']
    assert p['maximum_charge_micro_usd']==75000
    assert p['input']['reserved_remaining_usd']=='89.75' and p['protected_reserve_micro_usd']==10000000
    assert i['fallback_terminate_epoch']<p['billing_ceiling_epoch']
    assert 'stopAfter' not in i['payload'] # Don't confuse an earlier stop with termination.

@pytest.mark.parametrize('change',[{'pods':[{}]},{'volume_ids':['foreign']},{'autopay':True},{'balance_usd':'99'},{'account_hourly_usd':'0.1'}])
def test_account_gate(change):
    i=intent()
    with pytest.raises(EvidenceError):make_intent(NOW,{**i['baseline'],**change},i['quote'],IMAGE)

def test_stale_and_changed_quote_rejected():
    i=intent()
    with pytest.raises(EvidenceError,match='stale'):make_intent(NOW+601,i['baseline'],i['quote'],IMAGE)
    i['quote']['gpus'][0]['price']['secure']='0.25'
    with pytest.raises(EvidenceError,match='quote changed'):make_intent(NOW,i['baseline'],i['quote'],IMAGE)

@pytest.mark.parametrize('field,value',[('imageName','ubuntu:latest'),('gpuCount',2),('containerDiskInGb',500),('volumeInGb',5),('createdAt','2020-01-01T00:00:00Z'),('name','foreign')])
def test_wrong_identity_never_authorizes_teardown(field,value):
    i=intent();p=pod(i);p[field]=value
    if field=='name':
        with pytest.raises(EvidenceError):match_pod(i,{'pods':[p]},known_id='test-pod')
    else:
        from probe_provider_deadline import provision_errors
        assert match_pod(i,{'pods':[p]},known_id='test-pod')==p
        assert field in provision_errors(i,p)

def test_uncertain_creation_adopts_exact_intent_and_refuses_ambiguity():
    i=intent();p=pod(i)
    assert match_pod(i,{'pods':[p]})==p
    with pytest.raises(EvidenceError,match='ambiguous'):match_pod(i,{'pods':[p,{**p,'id':'other'}]})

def test_guard_timing_is_scoped_not_production_admission():
    i=intent();d=i['plan']['provider_terminate_epoch'];p=pod(i)
    observations=[{'observed_epoch':d-10,'pod':p},{'observed_epoch':d+10,'pod':None}]
    assert verdict(i,observations,False)=='OBSERVED_TERMINATION_IN_DEADLINE_WINDOW'
    assert verdict(i,observations,True)=='FAIL_REQUIRED_CALLER_TEARDOWN'
    assert verdict(i,[{'observed_epoch':d-100,'pod':p},{'observed_epoch':d-80,'pod':None}],False)=='INCONCLUSIVE_DISAPPEARED_OUTSIDE_WINDOW'


@pytest.mark.parametrize('mode',['native','ignored-deadline','lost-create-response','crash-after-create','misprovision-lost-response','disk-full-observation','transient-read','persistent-read','auth-failure'])
def test_controller_reconciliation_no_duplicate_and_teardown(tmp_path,monkeypatch,mode):
    import types
    import probe_provider_deadline as probe
    now=[NOW];live=[None];calls=[];initial=intent()
    monkeypatch.setattr(probe,'time',types.SimpleNamespace(time=lambda:now[0],monotonic=lambda:now[0]-NOW,sleep=lambda seconds:now.__setitem__(0,now[0]+seconds)))
    def observe():
        if live[0] and now[0]>=NOW+100:
            if mode=='transient-read' and now[0]<NOW+105:raise probe.ProviderFailure('http',status=503,transient=True)
            if mode=='persistent-read':raise probe.ProviderFailure('transport',transient=True)
            if mode=='auth-failure':raise probe.ProviderFailure('http',status=401)
        if mode!='ignored-deadline' and now[0]>=NOW+600:live[0]=None
        return {'observed_epoch':now[0],'response_sha256':'0'*64,'balance_usd':'100.4',
                'account_hourly_usd':'0.24' if live[0] else '0','autopay':False,
                'pods':[live[0]] if live[0] else [],'volume_ids':[],
                'http_clock':{'server_epoch':now[0],'request_started_epoch':now[0],'request_completed_epoch':now[0]}}
    def request(op,variables):
        calls.append(op)
        if op=='create':
            live[0]={**pod({'payload':variables['input']}),'costPerHr':'0.24','adjustedCostPerHr':'0.24','desiredStatus':'RUNNING'}
            if mode=='misprovision-lost-response':live[0]['imageName']='docker.io/library/'+IMAGE
            if mode in ('lost-create-response','misprovision-lost-response'):raise probe.Refused('lost response')
            return {'podFindAndDeployOnDemand':live[0]},'0'*64,{}
        if op=='identities':return {'myself':{'pods':[live[0]] if live[0] else []}},'0'*64,{}
        assert op=='terminate' and variables['input']['podId']=='test-pod'
        live[0]=None;return {'podTerminate':None},'0'*64,{}
    monkeypatch.setattr(probe,'account',observe);monkeypatch.setattr(probe,'request',request)
    directory=tmp_path/'journal'
    if mode=='crash-after-create':
        original=probe.Journal.append
        def die(self,kind,body):
            if kind=='creation-observed':raise KeyboardInterrupt('simulated process death')
            return original(self,kind,body)
        with monkeypatch.context() as m:
            m.setattr(probe.Journal,'append',die)
            with pytest.raises(KeyboardInterrupt):probe.run(directory,initial['quote'],IMAGE)
    if mode=='disk-full-observation':
        original=probe.Journal.append;failed=[False]
        def full(self,kind,body):
            if kind=='provider-observation' and not failed[0]:
                failed[0]=True;raise OSError('ENOSPC')
            return original(self,kind,body)
        monkeypatch.setattr(probe.Journal,'append',full)
    probe.run(directory,initial['quote'],IMAGE)
    probe.run(directory,initial['quote'],IMAGE) # terminal adoption must not create a new resource
    assert calls.count('create')==1 and live[0] is None
    result=probe.read_json(directory/'result.json')
    assert result['production_guard_admission']=='NOT_RUN'
    if mode in ('ignored-deadline','misprovision-lost-response','disk-full-observation','persistent-read','auth-failure'):
        assert calls.count('terminate')==1 and result['result']=='FAIL_REQUIRED_CALLER_TEARDOWN'
    else:assert 'terminate' not in calls and result['result']=='OBSERVED_TERMINATION_IN_DEADLINE_WINDOW'


def test_disk_failure_never_blocks_attributed_emergency_teardown(monkeypatch):
    import probe_provider_deadline as p
    class BrokenJournal:
        def append(self,*a):raise OSError('ENOSPC')
    calls=[]
    monkeypatch.setattr(p,'request',lambda operation,variables:(calls.append((operation,variables)) or ({},'0'*64,{})))
    assert p.emergency_terminate(intent(),'test-pod',BrokenJournal())=='test-pod'
    assert calls==[('terminate',{'input':{'podId':'test-pod'}})]


def test_clock_skew_refuses_before_creation():
    i=intent();i['baseline']['http_clock']['server_epoch']-=61
    with pytest.raises(EvidenceError,match='clock skew'):make_intent(NOW,i['baseline'],i['quote'],IMAGE)


def test_linked_probe_reserves_entire_predecessor_charge():
    i=intent()
    second=make_intent(NOW,i['baseline'],i['quote'],IMAGE,{'reserved_unsettled_usd':'0.075','baseline_balance_usd':'100.50'})
    assert second['plan']['input']['outstanding_usd']=='0.075'
    assert second['plan']['input']['allowance_usd']=='0.175'
    assert second['plan']['maximum_charge_micro_usd']==75000
    with pytest.raises(EvidenceError,match='debit exceeds'):
        make_intent(NOW,{**i['baseline'],'balance_usd':'100.40'},i['quote'],IMAGE,{'baseline_balance_usd':'100.50'})


@pytest.mark.parametrize('status,transient',[(401,False),(403,False),(400,False),(429,True),(503,True)])
def test_http_diagnostics_select_only_status(monkeypatch,status,transient):
    import probe_provider_deadline as p
    from urllib.error import HTTPError
    class Opener:
        def open(self,*a,**kw):raise HTTPError('https://private.invalid/reflected-secret',status,'reflected-secret',{},None)
    monkeypatch.setattr(p,'credential',lambda:'local-secret')
    monkeypatch.setattr(p,'build_opener',lambda *a:Opener())
    with pytest.raises(p.ProviderFailure) as e:p.request('account')
    assert p.diagnostic(e.value)=={'error_type':'ProviderFailure','category':'http','http_status':status,'transient':transient}
    assert 'secret' not in str(e.value)


def test_transient_grace_refuses_stale_backward_clock_or_near_deadline():
    from probe_provider_deadline import ProviderFailure,transient_read_grace
    error=ProviderFailure('transport',transient=True)
    assert transient_read_grace(error,20,0,20,0,100,False)
    assert not transient_read_grace(error,30,0,20,0,100,False)
    assert not transient_read_grace(error,20,0,30,0,100,False)
    assert not transient_read_grace(error,-1,0,20,0,100,False)
    assert not transient_read_grace(error,20,0,20,0,45,False)
    assert not transient_read_grace(error,20,0,20,0,100,True)
