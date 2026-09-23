"""Read-only reconciliation of one fully retained, explicit capacity rejection.

This does not create, stop services, clear a fence, release money or alter any
watchdog deadline. Only the exact provider rejection plus two later no-compute reads
of the same account can produce a receipt. Any retained storage must match the
original separately reserved selection. The original watchdog must remain
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
from retained_volume import baseline_valid

MESSAGE='There are no longer any instances available with the requested specifications. Please refresh and try again.'


def principal(value):
    if type(value) is not str or not 1<=len(value)<=128:raise EvidenceError('missing provider account identity')
    return digest({'runpod_account_id':value})


def retained_window(watch, now):
    selection = watch.get('retained_volume')
    if selection is not None:
        integer(now, selection['created_epoch'], selection['retention_deadline_epoch'] - 1,
                'retained storage recovery window')


def selected_baseline(watch, observation):
    """No compute; only the exact separately reserved storage may remain."""
    try:
        valid = (observation.get('pods') == []
                 and type(observation.get('account_hourly_usd')) is str
                 and baseline_valid(watch, observation))
    except (KeyError, TypeError, ValueError, ArithmeticError):
        valid = False
    if not valid:raise EvidenceError('account differs from selected storage baseline')


def identified_baseline(observation, watch, identity):
    retained = watch.get('retained_volume')
    keys = 'schema account_identity_sha256 response_sha256 http_clock observed_epoch pods volume_ids autopay account_hourly_usd'
    if retained is not None:keys += ' network_volumes retained_volume_sha256'
    fields(observation, keys, 'identified account baseline')
    require_digest(observation['response_sha256'])
    schema = ('ovl.retained-volume-account-identity-observation.v1' if retained is not None
              else 'ovl.empty-account-identity-observation.v1')
    if (observation['schema'] != schema or observation['account_identity_sha256'] != identity
            or retained is not None and observation['retained_volume_sha256'] != digest(retained)):
        raise EvidenceError('account or retained storage identity differs')
    selected_baseline(watch, observation)


def capture(watch=None):
    """One bounded read in this dedicated, single-threaded observation process."""
    import probe_provider_deadline as provider
    original=provider.OPERATIONS['account']
    try:
        provider.OPERATIONS['account']=original.replace('myself {','myself { id',1)
        data,root,clock=provider.request('account')
    finally:provider.OPERATIONS['account']=original
    watch = watch or {}
    v=data['myself'];volumes=[]
    if type(v['networkVolumes']) is not list or v['pods'] != []:
        raise EvidenceError('capacity reconciliation requires no compute')
    for volume in v['networkVolumes']:
        size=volume['size']
        if (type(size) not in (int,Decimal) or not Decimal(size).is_finite()
                or size!=int(size) or not 1<=size<=2**50
                or any(type(volume[k]) is not str or not volume[k] for k in ('id','name','dataCenterId'))):
            raise EvidenceError('invalid retained volume observation')
        volumes.append({**{k:volume[k] for k in ('id','name','dataCenterId')},'size':int(size)})
    result = {'schema':'ovl.empty-account-identity-observation.v1','account_identity_sha256':principal(v['id']),
            'response_sha256':root,'http_clock':clock,'observed_epoch':int(time.time()),
            'pods':v['pods'],'volume_ids':[x['id'] for x in volumes],
            'autopay':v['isAutoPayEnabled'],'account_hourly_usd':provider.amount(v['currentSpendPerHr'])}
    if watch.get('retained_volume') is not None:
        result.update(schema='ovl.retained-volume-account-identity-observation.v1',
                      network_volumes=volumes,retained_volume_sha256=digest(watch['retained_volume']))
    identified_baseline(result, watch, principal(v['id']))
    return result


def verify(rental,expected,diagnostic,request,raw,events,observations,heartbeat,now):
    w,p=validate(rental,expected);integer(now,1,2**53-1,'reconciliation time')
    retained_window(w, now)
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
            selected_baseline(w, account)
    if type(observations) is not list or len(observations)!=2:raise EvidenceError('two complete later observations required')
    for o in observations:
        identified_baseline(o, w, identity)
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
    result = {'schema':'ovl.capacity-rejection-reconciliation.v1','result':'EXPLICIT_REJECTION_AND_LATER_EMPTY_ACCOUNT_VERIFIED',
            'rental_intent_sha256':expected,'attempt_id':rental['payload']['name'],'observed_epoch':now,
            'response_sha256':hashlib.sha256(raw).hexdigest(),'diagnostic_sha256':digest(diagnostic),
            'account_identity_sha256':identity,'observations_sha256':digest(observations),'journal_prefix_sha256':digest(events),
            'watchdog_heartbeat_sha256':digest(heartbeat),'original_external_deadline_epoch':p['external_terminate_epoch'],
            'original_watchdog':'MUST_REMAIN_ARMED_UNCHANGED','creation_fence':'PRESERVE_NEVER_REISSUE_ORIGINAL_REQUEST',
            'billing':'NOT_SETTLED_NO_CEILING_RELEASE','provider_mutation':'NOT_RUN','scope':'Operator-trusted provider response and same-account observations. Allows stopping only the failed creator; a separately admitted future rental still requires fresh capacity, singleton account and full reserved budget. No training or independent-verification credit.'}
    if w.get('retained_volume') is not None:
        result.update(schema='ovl.capacity-rejection-reconciliation.v2',
                      result='EXPLICIT_REJECTION_AND_SELECTED_STORAGE_BASELINE_VERIFIED',
                      retained_volume_sha256=digest(w['retained_volume']),
                      retained_storage_reserved_usd=w['retained_volume']['reserved_usd'])
    return result
