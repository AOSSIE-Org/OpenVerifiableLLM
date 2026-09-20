"""Explicit fake provider packets: no resource mutations or real closure credit."""
import copy,json,hashlib
import pytest
import reconcile_capacity_rejection as m
from provider_request_receipts import shape
from test_rental_controller import intent
from ovl_pipeline.canonical import EvidenceError,digest


def fixture():
    r=intent();w=r['watchdog_intent'];start=w['plan']['input']['now_epoch'];now=w['creation_latest_epoch']+240
    raw=json.dumps({'data':{'podFindAndDeployOnDemand':None},'errors':[{'message':m.MESSAGE,'extensions':{'code':'SUPPLY_CONSTRAINT','userId':'explicit-fake-account'}}]}).encode()
    request={'operation':'create','variables_sha256':digest({'input':r['payload']}),'observed_epoch':start}
    failure={'error_type':'ProviderFailure','category':'invalid-response-Refused','http_status':None,'transient':False}
    d={'schema':'ovl.provider-creation-response-diagnostic.v1','variables_sha256':request['variables_sha256'],'http_status':200,'same_endpoint':True,'content_encoding':'identity','request_result':'RAISED','retained_read_complete':True,'failure':failure,'retained_bytes':len(raw),'observed_read_bytes':len(raw),'retained_bytes_sha256':hashlib.sha256(raw).hexdigest(),'shape':shape(raw),'observed_epoch':start+1}
    events=[{'kind':'creation-intent','body':r},{'kind':'decision','body':{'action':'CREATION_FENCED'}},{'kind':'failure','body':{'stage':'creation','reissue':'FORBIDDEN',**failure}}]
    observations=[{'schema':'ovl.empty-account-identity-observation.v1','account_identity_sha256':m.principal('explicit-fake-account'),'response_sha256':'a'*64,'http_clock':{'server_epoch':t,'request_started_epoch':t,'request_completed_epoch':t},'observed_epoch':t,'pods':[],'volume_ids':[],'autopay':False,'account_hourly_usd':'0'} for t in (now-20,now)]
    heartbeat={'schema':'ovl.external-watchdog-heartbeat.v1','intent_sha256':digest(w),'plan_sha256':digest(w['plan']),'external_terminate_epoch':w['plan']['external_terminate_epoch'],'observed_epoch':now,'state':'ARMED','pod_id':None,'pid':123,'automatic_provider_termination':'UNVERIFIED'}
    return [r,digest(r),d,request,raw,events,observations,heartbeat,now]


def test_exact_rejection_does_not_release_funds_or_watchdog_or_reissue_authority():
    args=fixture();result=m.verify(*args)
    assert result['result']=='EXPLICIT_REJECTION_AND_LATER_EMPTY_ACCOUNT_VERIFIED'
    assert result['billing']=='NOT_SETTLED_NO_CEILING_RELEASE'
    assert result['original_watchdog']=='MUST_REMAIN_ARMED_UNCHANGED'
    assert result['original_external_deadline_epoch']==args[0]['watchdog_intent']['plan']['external_terminate_epoch']


@pytest.mark.parametrize('code',[None,'UNAUTHENTICATED','FORBIDDEN','INTERNAL_SERVER_ERROR',
                               'BAD_USER_INPUT','UNKNOWN',False,[],{}])
def test_matching_message_cannot_override_missing_or_contradictory_error_code(code):
    args=fixture();value=json.loads(args[4]);ext=value['errors'][0]['extensions']
    if code is None:ext.pop('code')
    else:ext['code']=code
    args[4]=json.dumps(value).encode()
    args[2].update(retained_bytes=len(args[4]),observed_read_bytes=len(args[4]),
                   retained_bytes_sha256=hashlib.sha256(args[4]).hexdigest(),shape=shape(args[4]))
    with pytest.raises(EvidenceError,match='exact explicit capacity rejection'):m.verify(*args)


@pytest.mark.parametrize('damage',['null-error','list-error','null-extensions','list-extensions','null-code'])
def test_malformed_error_identity_fails_closed(damage):
    args=fixture();value=json.loads(args[4])
    if damage=='null-error':value['errors'][0]=None
    elif damage=='list-error':value['errors'][0]=[]
    elif damage=='null-extensions':value['errors'][0]['extensions']=None
    elif damage=='list-extensions':value['errors'][0]['extensions']=[]
    else:value['errors'][0]['extensions']['code']=None
    args[4]=json.dumps(value).encode()
    args[2].update(retained_bytes=len(args[4]),observed_read_bytes=len(args[4]),
                   retained_bytes_sha256=hashlib.sha256(args[4]).hexdigest(),shape=shape(args[4]))
    with pytest.raises(EvidenceError,match='exact explicit capacity rejection'):m.verify(*args)


@pytest.mark.parametrize('damage',['timeout','partial','bytes','request','returned-id','different-error','extra-error','nonfinite','wrong-account','pods','volumes','autopay','rate','too-early','stale','unseparated','clock','old-pod','duplicate-fence','wrong-journal','dead-watchdog','changed-deadline','watchdog-pod'])
def test_ambiguous_creation_or_changed_identity_never_reconciles(damage):
    a=fixture();r,_,d,request,raw,events,obs,h,now=a
    if damage=='timeout':d['failure']['category']='transport'
    elif damage=='partial':d['retained_read_complete']=False
    elif damage=='bytes':a[4]=raw+b' '
    elif damage=='request':request['variables_sha256']='b'*64
    elif damage in ('returned-id','different-error','extra-error','nonfinite'):
        v=json.loads(raw)
        if damage=='returned-id':v['data']['podFindAndDeployOnDemand']={'id':'unexpected'}
        elif damage=='different-error':v['errors'][0]['message']='ambiguous server failure'
        elif damage=='extra-error':v['errors'].append(v['errors'][0])
        else:v['unexpected']=float('nan')
        a[4]=json.dumps(v).encode();d.update(retained_bytes=len(a[4]),observed_read_bytes=len(a[4]),retained_bytes_sha256=hashlib.sha256(a[4]).hexdigest(),shape=shape(a[4]))
    elif damage=='wrong-account':obs[1]['account_identity_sha256']='b'*64
    elif damage=='pods':obs[1]['pods']=[{'id':'present'}]
    elif damage=='volumes':obs[1]['volume_ids']=['present']
    elif damage=='autopay':obs[1]['autopay']=True
    elif damage=='rate':obs[1]['account_hourly_usd']='0.01'
    elif damage=='too-early':obs[0]['http_clock']['request_started_epoch']=r['watchdog_intent']['creation_latest_epoch']
    elif damage=='stale':a[-1]+=31
    elif damage=='unseparated':obs[0]=copy.deepcopy(obs[1])
    elif damage=='clock':obs[1]['http_clock']['server_epoch']+=100
    elif damage=='old-pod':events.append({'kind':'creation-observed','body':{'id':'observed-once'}})
    elif damage=='duplicate-fence':events.append(copy.deepcopy(events[1]))
    elif damage=='wrong-journal':events[0]['body']={}
    elif damage=='dead-watchdog':h['state']='STOPPED'
    elif damage=='changed-deadline':h['external_terminate_epoch']+=1
    else:h['pod_id']='late-pod'
    with pytest.raises(EvidenceError):m.verify(*a)
