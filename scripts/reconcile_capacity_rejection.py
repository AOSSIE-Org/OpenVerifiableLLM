"""Read-only reconciliation of one fully retained, explicit capacity rejection.

This does not create, stop services, clear a fence, release money or alter any
watchdog deadline. Only the exact provider rejection plus two later empty reads
of the same account can produce a receipt. The original watchdog must remain
armed. Timeouts, partial responses, arbitrary errors and any observed pod fail.
Provider account/control-plane observations are trusted here, not cryptographic
proofs of noncreation or final billing settlement.
"""
from decimal import Decimal
import hashlib
import json
import time
from ovl_pipeline.canonical import EvidenceError,digest,require_digest
from ovl_pipeline.schema import fields,integer
from provider_preflight import unique_pairs
from provider_request_receipts import shape
from run_rental_controller import validate

MESSAGE='There are no longer any instances available with the requested specifications. Please refresh and try again.'


def principal(value):
    if type(value) is not str or not 1<=len(value)<=128:raise EvidenceError('missing provider account identity')
    return digest({'runpod_account_id':value})


def capture():
    """One bounded read in this dedicated, single-threaded observation process."""
    import probe_provider_deadline as provider
    original=provider.OPERATIONS['account']
    try:
        provider.OPERATIONS['account']=original.replace('myself {','myself { id',1)
        data,root,clock=provider.request('account')
    finally:provider.OPERATIONS['account']=original
    v=data['myself']
    if v['pods']!=[] or v['networkVolumes']!=[] or v['isAutoPayEnabled'] is not False:
        raise EvidenceError('capacity reconciliation requires an empty account')
    if type(v['currentSpendPerHr']) not in (int,Decimal) or Decimal(v['currentSpendPerHr'])!=0:
        raise EvidenceError('capacity reconciliation requires zero observed rate')
    return {'schema':'ovl.empty-account-identity-observation.v1','account_identity_sha256':principal(v['id']),
            'response_sha256':root,'http_clock':clock,'observed_epoch':int(time.time()),
            'pods':[],'volume_ids':[],'autopay':False,'account_hourly_usd':'0'}


def verify(rental,expected,diagnostic,request,raw,events,observations,heartbeat,now):
    w,p=validate(rental,expected);integer(now,1,2**53-1,'reconciliation time')
    expected_variables=digest({'input':rental['payload']})
    fields(request,'operation variables_sha256 observed_epoch','private creation request')
    if request['operation']!='create' or request['variables_sha256']!=expected_variables:
        raise EvidenceError('creation request belongs to another payload')
    integer(request['observed_epoch'],p['input']['now_epoch'],w['creation_latest_epoch'],'creation request clock')
    if (diagnostic['schema']!='ovl.provider-creation-response-diagnostic.v1'
        or diagnostic['variables_sha256']!=expected_variables or diagnostic['http_status']!=200
        or diagnostic['same_endpoint'] is not True or diagnostic['content_encoding']!='identity'
        or diagnostic['request_result']!='RAISED' or diagnostic['retained_read_complete'] is not True
        or diagnostic['failure']!={'error_type':'ProviderFailure','category':'invalid-response-Refused','http_status':None,'transient':False}
        or diagnostic['retained_bytes']!=len(raw) or diagnostic['observed_read_bytes']!=len(raw)
        or not 1<=len(raw)<=1024**2 or diagnostic['retained_bytes_sha256']!=hashlib.sha256(raw).hexdigest()
        or diagnostic['shape']!=shape(raw)):
        raise EvidenceError('missing or inconsistent complete creation response')
    integer(diagnostic['observed_epoch'],request['observed_epoch'],now,'creation response clock')
    try:value=json.loads(raw,object_pairs_hook=unique_pairs,parse_float=Decimal,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite response')))
    except Exception:raise EvidenceError('invalid complete creation response') from None
    if (type(value) is not dict or value.get('data')!={'podFindAndDeployOnDemand':None}
        or type(value.get('errors')) is not list or len(value['errors'])!=1
        or type(value['errors'][0]) is not dict
        or value['errors'][0].get('message')!=MESSAGE
        or type(value['errors'][0].get('extensions')) is not dict
        or value['errors'][0]['extensions'].get('code')!='SUPPLY_CONSTRAINT'):
        raise EvidenceError('only the exact explicit capacity rejection can reconcile early')
    identity=principal(value['errors'][0]['extensions'].get('userId'))
    if [e['body'] for e in events if e['kind']=='creation-intent']!=[rental]:
        raise EvidenceError('controller journal selects another creation')
    if any(e['kind']=='creation-observed' for e in events):raise EvidenceError('a pod was observed; use normal resource teardown')
    failures=[e['body'] for e in events if e['kind']=='failure' and e['body'].get('stage')=='creation']
    if failures!=[{'stage':'creation','reissue':'FORBIDDEN',**diagnostic['failure']}]:
        raise EvidenceError('creation failure journal differs')
    fences=[e for e in events if e['kind']=='decision' and e['body'].get('action')=='CREATION_FENCED']
    if len(fences)!=1:raise EvidenceError('one original creation fence required')
    for e in events:
        if e['kind']=='provider-observation':
            account=e['body'].get('account',e['body'])
            if account.get('pods') or account.get('volume_ids'):raise EvidenceError('resource appeared during attempted creation')
    if type(observations) is not list or len(observations)!=2:raise EvidenceError('two complete later observations required')
    for o in observations:
        fields(o,'schema account_identity_sha256 response_sha256 http_clock observed_epoch pods volume_ids autopay account_hourly_usd','identified empty account observation')
        require_digest(o['response_sha256'])
        if (o['schema']!='ovl.empty-account-identity-observation.v1' or o['account_identity_sha256']!=identity
            or o['pods']!=[] or o['volume_ids']!=[] or o['autopay'] is not False or o['account_hourly_usd']!='0'):
            raise EvidenceError('account identity or empty resource state differs')
        integer(o['observed_epoch'],max(w['creation_latest_epoch']+180,diagnostic['observed_epoch']),now,'post-creation observation')
        c=o['http_clock'];fields(c,'server_epoch request_started_epoch request_completed_epoch','provider observation clock')
        for k in c:integer(c[k],1,2**53-1,k)
        if not w['creation_latest_epoch']+180<=c['request_started_epoch']<=c['request_completed_epoch']<=o['observed_epoch']:
            raise EvidenceError('account request precedes reconciliation window')
        if not c['request_started_epoch']-5<=c['server_epoch']<=c['request_completed_epoch']+5:
            raise EvidenceError('account/provider clock mismatch')
    if observations[1]['observed_epoch']-observations[0]['observed_epoch']<15 or now-observations[1]['observed_epoch']>30:
        raise EvidenceError('separated, fresh absence observations required')
    fields(heartbeat,'schema intent_sha256 plan_sha256 external_terminate_epoch observed_epoch state pod_id pid automatic_provider_termination','original watchdog heartbeat')
    integer(heartbeat['pid'],1,2**31-1,'original watchdog PID')
    if (heartbeat['schema']!='ovl.external-watchdog-heartbeat.v1' or heartbeat['intent_sha256']!=digest(w)
        or heartbeat['plan_sha256']!=digest(p) or heartbeat['external_terminate_epoch']!=p['external_terminate_epoch']
        or heartbeat['state']!='ARMED' or heartbeat['pod_id'] is not None
        or heartbeat['automatic_provider_termination']!='UNVERIFIED'
        or type(heartbeat['observed_epoch']) is not int or not 0<=now-heartbeat['observed_epoch']<=30):
        raise EvidenceError('original watchdog must remain armed with unchanged identity/deadline')
    return {'schema':'ovl.capacity-rejection-reconciliation.v1','result':'EXPLICIT_REJECTION_AND_LATER_EMPTY_ACCOUNT_VERIFIED',
            'rental_intent_sha256':expected,'attempt_id':rental['payload']['name'],'observed_epoch':now,
            'response_sha256':hashlib.sha256(raw).hexdigest(),'diagnostic_sha256':digest(diagnostic),
            'account_identity_sha256':identity,'observations_sha256':digest(observations),'journal_prefix_sha256':digest(events),
            'watchdog_heartbeat_sha256':digest(heartbeat),'original_external_deadline_epoch':p['external_terminate_epoch'],
            'original_watchdog':'MUST_REMAIN_ARMED_UNCHANGED','creation_fence':'PRESERVE_NEVER_REISSUE_ORIGINAL_REQUEST',
            'billing':'NOT_SETTLED_NO_CEILING_RELEASE','provider_mutation':'NOT_RUN','scope':'Operator-trusted provider response and same-account observations. Allows stopping only the failed creator; a separately admitted future rental still requires fresh capacity, singleton account and full reserved budget. No training or independent-verification credit.'}
