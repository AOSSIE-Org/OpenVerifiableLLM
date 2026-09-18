#!/usr/bin/env python3
"""Bounded operational deadline test, never model training or production admission.

Run as a restartable systemd user service. The first invocation durably records one
creation intent; restarts reconcile it and NEVER send another creation. Observations
and teardown intents live off-pod. No secrets are passed to the pod or logged.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid
from urllib.request import Request, build_opener
from urllib.error import HTTPError, URLError
from email.utils import parsedate_to_datetime
import shutil

from provider_preflight import credential, NoRedirect, ENDPOINT, Refused, unique_pairs, amount
from ovl_pipeline.canonical import EvidenceError, digest, read_json, write_json
from ovl_pipeline.supervision import Journal, rental_plan

ACCOUNT='''query OvlGuardAccount { myself { clientBalance currentSpendPerHr isAutoPayEnabled
 pods { id name createdAt desiredStatus gpuCount costPerHr adjustedCostPerHr imageName containerDiskInGb volumeInGb }
 networkVolumes { id } } }'''
CREATE='''mutation OvlGuardCreate($input: PodFindAndDeployOnDemandInput!) {
 podFindAndDeployOnDemand(input:$input) { id name createdAt gpuCount imageName } }'''
TERMINATE='''mutation OvlGuardTerminate($input: PodTerminateInput!) { podTerminate(input:$input) }'''
IDENTITIES='query OvlGuardIdentities { myself { pods { id name } } }'
OPERATIONS={'account':ACCOUNT,'create':CREATE,'terminate':TERMINATE,'identities':IDENTITIES}
GPU='NVIDIA RTX 2000 Ada Generation'


class ProviderFailure(Refused):
    """Selected nonsecret diagnostics only; never carry reflected HTTP text/URL."""
    def __init__(self, category, *, status=None, transient=False):
        self.category=category;self.status=status;self.transient=transient
        super().__init__(category)


def diagnostic(error):
    return {'error_type':type(error).__name__,
            **({'category':error.category,'http_status':error.status,'transient':error.transient}
               if isinstance(error,ProviderFailure) else {})}


def transient_read_grace(error, now, last_success, monotonic_now, last_success_monotonic, fallback_epoch, fallback):
    # Up to30s since a successful read, followed by at most5s wait +20s request.
    # No grace for auth, malformed data, identity, storage, create or teardown errors.
    return (isinstance(error,ProviderFailure) and error.transient and not fallback
            and last_success is not None and 0<=now-last_success<30
            and 0<=monotonic_now-last_success_monotonic<30 and now+25<fallback_epoch)



def request(operation, variables=None):
    if operation not in OPERATIONS:raise EvidenceError('unknown provider operation')
    q=OPERATIONS[operation]
    req=Request(ENDPOINT,data=json.dumps({'query':q,'variables':variables or {}}).encode(),
        headers={'Authorization':'Bearer '+credential(),'Content-Type':'application/json',
                 'User-Agent':'OpenVerifiableLLM-deadline-probe/1','Accept-Encoding':'identity'},method='POST')
    started=int(time.time())
    try:
        with build_opener(NoRedirect()).open(req,timeout=20) as r:
            if r.url!=ENDPOINT or r.status!=200 or r.headers.get('Content-Encoding','identity')!='identity':
                raise Refused('unexpected provider HTTP response')
            raw=r.read(1024*1024+1)
            server_epoch=int(parsedate_to_datetime(r.headers["Date"]).timestamp())
        if len(raw)>1024*1024:raise Refused('provider response limit')
        obj=json.loads(raw,object_pairs_hook=unique_pairs,parse_float=Decimal,
                       parse_constant=lambda _: (_ for _ in ()).throw(Refused('nonfinite provider response')))
        if type(obj) is not dict or obj.get('errors') or type(obj.get('data')) is not dict:
            raise Refused('provider errors or missing data; raw response withheld')
        return obj['data'],hashlib.sha256(raw).hexdigest(),{'server_epoch':server_epoch,'request_started_epoch':started,'request_completed_epoch':int(time.time())}
    except HTTPError as e:
        raise ProviderFailure('http',status=e.code,transient=e.code==429 or 500<=e.code<=599) from None
    except (URLError,TimeoutError,ConnectionError):
        raise ProviderFailure('transport',transient=True) from None
    except Exception as e:
        raise ProviderFailure('invalid-response-'+type(e).__name__) from None


def account():
    d,h,clock=request('account');v=d['myself']
    if type(v['isAutoPayEnabled']) is not bool or type(v['pods']) is not list or type(v['networkVolumes']) is not list:
        raise EvidenceError('invalid provider account observation')
    pods=[]
    for p in v['pods']:
        if (not re.fullmatch('[A-Za-z0-9_-]{1,96}',p['id']) or type(p['name']) is not str
                or type(p['createdAt']) is not str or type(p['gpuCount']) is not int):
            raise EvidenceError('invalid pod identity')
        selected={k:p[k] for k in ('id','name','createdAt','desiredStatus','gpuCount','imageName','containerDiskInGb','volumeInGb')}
        selected.update(costPerHr=amount(p['costPerHr']),adjustedCostPerHr=amount(p['adjustedCostPerHr']))
        pods.append(selected)
    return {'observed_epoch':int(time.time()),'response_sha256':h,'balance_usd':amount(v['clientBalance']),
            'account_hourly_usd':amount(v['currentSpendPerHr']),'autopay':v['isAutoPayEnabled'],
            'pods':pods,'volume_ids':[p['id'] for p in v['networkVolumes']],'http_clock':clock}


def make_intent(now, initial, quote, image, prior=None):
    clock=initial['http_clock']
    if not clock['request_started_epoch']-5<=clock['server_epoch']<=clock['request_completed_epoch']+5:
        raise EvidenceError('local/provider clock skew exceeds five seconds')
    if initial['pods'] or initial['volume_ids'] or initial['autopay'] or Decimal(initial['account_hourly_usd'])!=0:
        raise EvidenceError('probe requires empty, zero-rate account with autopay disabled')
    if Decimal(initial['balance_usd'])<100:raise EvidenceError('probe requires full reserved balance')
    if prior is not None and Decimal(prior['baseline_balance_usd'])-Decimal(initial['balance_usd'])>Decimal('0.075'):
        raise EvidenceError('observed prior debit exceeds fully reserved predecessor ceiling')
    if not re.fullmatch(r'ubuntu@sha256:[0-9a-f]{64}',image):raise EvidenceError('pinned official minimal image required')
    if not 0<=now-quote['observed_epoch']<=600:raise EvidenceError('stale provider quote')
    choices=[g for g in quote['gpus'] if g.get('secure') and g.get('availability')!='NONE' and Decimal(str(g.get('price',{}).get('secure',0)))>0]
    candidate=min(choices,key=lambda g:Decimal(str(g['price']['secure'])))
    if candidate['id']!=GPU or Decimal(str(candidate['price']['secure']))>Decimal('0.24'):
        raise EvidenceError('fixed cheapest available secure probe quote changed')
    attempt='ovllm-deadline-probe-'+uuid.uuid4().hex
    plan=rental_plan({'schema':'ovl.rental-budget-input.v1','attempt_id':attempt,'now_epoch':now,
        'spent_usd':'0','outstanding_usd':'0.075' if prior is not None else '0','reserved_remaining_usd':'89.75','allowance_usd':'0.175' if prior is not None else '0.25',
        'hourly_upper_usd':'0.30','quote_sha256':digest(quote),'maximum_seconds':600,
        'checkpoint_grace_seconds':300,'billing_slack_seconds':300})
    deadline=plan['provider_terminate_epoch']
    payload={'name':attempt,'cloudType':'SECURE','gpuCount':1,'gpuTypeId':GPU,
        'imageName':image,'containerDiskInGb':4,'volumeInGb':0,'minVcpuCount':1,'minMemoryInGb':1,
        'dockerArgs':'/bin/sleep infinity','startSsh':False,'startJupyter':False,
        'terminateAfter':datetime.fromtimestamp(deadline,timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
    return {'schema':'ovl.deadline-probe-intent.v1','scope':'operational guard probe only; no training',
        'plan':plan,'payload':payload,'baseline':initial,'creation_latest_epoch':now+30,
        'fallback_terminate_epoch':deadline+120,'quote':quote,'training_admission':'NOT_RUN',**({'prior_probe':prior} if prior is not None else {})}


def match_pod(intent, observed, known_id=None):
    p=intent['payload'];matches=[x for x in observed['pods'] if x['name']==p['name'] or x['id']==known_id]
    if len(matches)>1:raise EvidenceError('ambiguous attributed pod; never create another')
    if not matches:return None
    x=matches[0]
    # Attribution follows the unique pre-request UUID on the empty baseline,
    # or the authenticated returned id. Shape errors must trigger cleanup of our
    # misprovisioned resource rather than make its ownership disappear.
    if x['name']!=p['name'] or (known_id is not None and x['id']!=known_id):
        raise EvidenceError('provider pod attribution conflicts with creation intent')
    return x


def provision_errors(intent,pod):
    p=intent['payload'];errors=[]
    for k in ('imageName','gpuCount','containerDiskInGb','volumeInGb'):
        if pod[k]!=p[k]:errors.append(k)
    try:
        created=int(datetime.fromisoformat(pod['createdAt'].replace('Z','+00:00')).timestamp())
        if not intent['plan']['input']['now_epoch']-60<=created<=intent['creation_latest_epoch']+120:errors.append('createdAt')
    except (TypeError,ValueError):errors.append('createdAt')
    return errors


def emergency_terminate(intent, known, journal):
    """Best-effort journal cannot gate a necessary attributed teardown request."""
    if known is None:
        d,_,_=request('identities')
        matches=[p for p in d['myself']['pods'] if p['name']==intent['payload']['name']]
        if len(matches)!=1:return None
        known=matches[0]['id']
    if not re.fullmatch('[A-Za-z0-9_-]{1,96}',known):raise EvidenceError('invalid attributed id')
    # Preserve a requested-action event when possible, but never suppress teardown
    # because evidence storage itself became unavailable. Creation still requires
    # successful durable intent; an emergency abort never creates anything.
    try:journal.append('decision',{'action':'TERMINATE','pod_id':known,'reason':'emergency abort'})
    except Exception:pass
    result=request('terminate',{'input':{'podId':known}})
    try:journal.append('teardown',{'complete':False,'pod_id':known,'termination_response_sha256':result[1]})
    except Exception:pass
    return known


def verdict(intent, observations, fallback_requested):
    live=[o for o in observations if o['pod'] is not None]
    missing=[o for o in observations if o['pod'] is None]
    deadline=intent['plan']['provider_terminate_epoch']
    if not live:return 'INCONCLUSIVE_NEVER_OBSERVED'
    absent_after=[o for o in missing if o['observed_epoch']>live[-1]['observed_epoch']]
    if not absent_after:return 'PENDING'
    if fallback_requested:return 'FAIL_REQUIRED_CALLER_TEARDOWN'
    last=live[-1]['observed_epoch'];gone=absent_after[0]['observed_epoch']
    if deadline-45<=last<gone<=deadline+120:return 'OBSERVED_TERMINATION_IN_DEADLINE_WINDOW'
    return 'INCONCLUSIVE_DISAPPEARED_OUTSIDE_WINDOW'


def run(directory, quote, image, prior=None):
    with Journal(directory).lease() as journal:
        intents=[e['body'] for e in journal.events if e['kind']=='creation-intent']
        if len(intents)>1:raise EvidenceError('multiple creation intents forbidden')
        if any(e['kind']=='teardown' and e['body'].get('complete') is True for e in journal.events):return
        first=not intents
        if first:
            usage=shutil.disk_usage(directory)
            if usage.free<1024**3:raise EvidenceError('insufficient local evidence headroom')
            initial=account();intent=make_intent(int(time.time()),initial,quote,image,prior)
            # Publish all intent/attribution/budget bytes before issuing creation once.
            journal.append('creation-intent',intent)
            try:
                if time.time()>intent['creation_latest_epoch']:raise EvidenceError('creation window expired')
                result,response_sha,_=request('create',{'input':intent['payload']})
                pod=result['podFindAndDeployOnDemand']
                if type(pod) is not dict or type(pod.get('id')) is not str:raise EvidenceError('missing created pod identity')
                try:journal.append('creation-observed',{'id':pod['id'],'response_sha256':response_sha})
                except Exception:
                    emergency_terminate(intent,pod['id'],journal)
                    raise
            except Exception as e:
                try:journal.append('failure',{'stage':'creation',**diagnostic(e),'reissue':'FORBIDDEN'})
                except Exception:
                    emergency_terminate(intent,None,journal)
                    raise
        else:intent=intents[0]
        observed_ids=[e['body']['id'] for e in journal.events if e['kind']=='creation-observed']
        if len(set(observed_ids))>1:raise EvidenceError('multiple observed creation identities')
        known=observed_ids[0] if observed_ids else None
        history=[e['body'] for e in journal.events if e['kind']=='provider-observation']
        fallback=any(e['kind']=='decision' and e['body'].get('action')=='TERMINATE' for e in journal.events)
        monotonic_limit=time.monotonic()+min(720,max(0,intent['fallback_terminate_epoch']-time.time()))
        last_success=history[-1]['observed_epoch'] if history else None
        last_success_monotonic=time.monotonic()-(max(0,time.time()-last_success) if last_success is not None else 60)
        while True:
            stage='account'
            try:
                obs=account();stage='observation';pod=match_pod(intent,obs,known)
                if pod and known is None:
                    known=pod['id'];journal.append('creation-observed',{'id':known,'adopted_from_intent':True})
                current={**obs,'pod':pod};history.append(current);journal.append('provider-observation',current)
                now=obs['observed_epoch'];last_success=now;last_success_monotonic=time.monotonic()
                live=[h for h in history if h['pod'] is not None]
                if pod is None and (live or now>intent['fallback_terminate_epoch']+180):
                    # Two fresh observations 20 seconds apart establish absence; no cached 404.
                    missing=[h for h in history if h['observed_epoch']>(live[-1]['observed_epoch'] if live else 0) and h['pod'] is None]
                    if len(missing)>=2 and missing[-1]['observed_epoch']-missing[-2]['observed_epoch']>=15:
                        report={'complete':True,'result':verdict(intent,history,fallback),'pod_id':known,
                            'intent_sha256':digest(intent),'confirmed_absent_epoch':now,'residual_network_volumes':obs['volume_ids'],
                            'provider_billing_reconciliation':'PENDING','production_guard_admission':'NOT_RUN',
                            'limitations':'Timing observation does not expose scheduler readback or prove causation; other provider termination causes remain possible.'}
                        journal.append('teardown',report);write_json(directory/'result.json',report);return
                excessive=pod is not None and (Decimal(pod['costPerHr'])>Decimal('0.299') or Decimal(pod['adjustedCostPerHr'])>Decimal('0.299'))
                clock=obs['http_clock']
                clock_bad=not clock['request_started_epoch']-5<=clock['server_epoch']<=clock['request_completed_epoch']+5
                if pod and (fallback or now>=intent['fallback_terminate_epoch'] or time.monotonic()>=monotonic_limit
                            or excessive or obs['autopay'] or clock_bad or provision_errors(intent,pod)):
                    journal.append('decision',{'action':'TERMINATE','pod_id':known,'reason':'native deadline overdue or observed resource guard failure'})
                    fallback=True
                    _,h,_=request('terminate',{'input':{'podId':known}})
                    journal.append('teardown',{'complete':False,'pod_id':known,'termination_response_sha256':h})
                write_json(directory/'heartbeat.json',{'observed_epoch':now,'pid':os.getpid(),'pod_id':known,
                    'fallback_terminate_epoch':intent['fallback_terminate_epoch'],'production_admission':'NOT_RUN'})
            except Exception as e:
                if stage=='account' and transient_read_grace(e,time.time(),last_success,time.monotonic(),last_success_monotonic,intent['fallback_terminate_epoch'],fallback):
                    try:journal.append('failure',{'stage':'transient-account-read',**diagnostic(e),'action':'bounded-read-retry'})
                    except Exception:pass
                    else:
                        time.sleep(5)
                        continue
                # Unexpected metadata, clock or disk failures abort the empty
                # probe immediately. Recover attribution with a minimal query if
                # the create response was lost; never wait indefinitely on shape.
                fallback=True
                try:known=emergency_terminate(intent,known,journal) or known
                except Exception:pass
                try:journal.append('failure',{'stage':stage,**diagnostic(e)})
                except Exception:pass
            time.sleep(20)


def prior_probe_receipt(root):
    """Fixed predecessor only: preserve failed attempt and reserve its entire ceiling."""
    directory=root/'.ovllm-cache/provider-deadline-probe-v1'
    with Journal(directory).lease() as journal:
        events=journal.events
        if not events or events[0]['kind']!='creation-intent':raise EvidenceError('missing predecessor intent')
        terminal=events[-1]
        report=read_json(directory/'result.json')
        if (terminal['kind']!='teardown' or terminal['body']!=report or report.get('complete') is not True
                or report.get('result')!='FAIL_REQUIRED_CALLER_TEARDOWN' or report.get('pod_id')!='lrsbh9avgltuxm'
                or report.get('residual_network_volumes')!=[]):raise EvidenceError('predecessor not reconciled absent')
        prior=events[0]['body']
        if (digest(prior)!=report['intent_sha256'] or prior['plan']['maximum_charge_micro_usd']!=75000
                or prior['plan']['input']['spent_usd']!='0' or prior['plan']['input']['outstanding_usd']!='0'):
            raise EvidenceError('predecessor reservation mismatch')
        return {'pod_id':report['pod_id'],'baseline_balance_usd':prior['baseline']['balance_usd'],'intent_sha256':digest(prior),'terminal_event_sha256':digest(terminal),
                'result_sha256':digest(report),'reserved_unsettled_usd':'0.075',
                'scope':'Entire predecessor ceiling reserved until attributed billing settles; observed debit is not final billing.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--quote',type=Path,required=True);p.add_argument('--image',required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    expected=root/'.ovllm-cache/provider-deadline-probe-v2'
    if a.directory.resolve()!=expected:p.error('only linked second probe directory allowed; original must stay terminal')
    prior=prior_probe_receipt(root)
    if a.dry_run:
        observation=account();intent=make_intent(int(time.time()),observation,read_json(a.quote,canonical_required=False),a.image,prior)
        print(json.dumps({'result':'DRY_RUN_NO_MUTATION','intent':intent}));return
    if os.environ.get('OVL_SYSTEMD_DEADLINE_PROBE')!='1':p.error('launch only through prepared restartable service')
    run(a.directory,read_json(a.quote,canonical_required=False),a.image,prior)

if __name__=='__main__':main()
