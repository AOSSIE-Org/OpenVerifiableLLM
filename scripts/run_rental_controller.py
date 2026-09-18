#!/usr/bin/env python3
"""One-shot bounded rental lifecycle, alongside the separately started watchdog.

This controller never launches training or claims production admission. It creates
at most once from a pinned intent, supervises costs and external workload/export
health, requests graceful stop, and explicitly terminates/observes absence. Run
under a restartable off-pod service with the watchdog already armed. Credentials
stay on the controller host. A separate dispatcher must enforce production gates.
"""
import argparse
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
import re
import time

from ovl_pipeline.canonical import EvidenceError,digest,read_json,require_digest,write_json
from ovl_pipeline.schema import fields
from ovl_pipeline.supervision import ControllerBusy,Journal,observe
from probe_provider_deadline import account,request,match_pod,provision_errors,diagnostic
from run_external_watchdog import validate_intent


def validate(value,expected):
    require_digest(expected)
    if digest(value)!=expected:raise EvidenceError('rental intent differs from selected pin')
    fields(value,'schema watchdog_intent payload','rental intent')
    if value['schema']!='ovl.rental-controller-intent.v1':raise EvidenceError('unsupported rental controller intent')
    w=value['watchdog_intent'];p=validate_intent(w,digest(w));payload=value['payload']
    fields(payload,'name gpuCount imageName containerDiskInGb volumeInGb terminateAfter cloudType gpuTypeId minVcpuCount minMemoryInGb dockerArgs startSsh startJupyter ports','creation payload')
    if any(payload[k]!=v for k,v in w['payload'].items()):raise EvidenceError('creation differs from watchdog identity/deadline')
    if payload['cloudType']!='SECURE' or payload['startSsh'] is not True or payload['startJupyter'] is not False:
        raise EvidenceError('secure SSH-only rental required')
    if payload['ports']!='22/tcp':raise EvidenceError('only SSH port may be exposed')
    if payload['dockerArgs']!='':raise EvidenceError('only immutable image entrypoint may start')
    if type(payload['gpuTypeId']) is not str or not re.fullmatch(r'[A-Za-z0-9 ._-]{1,96}',payload['gpuTypeId']):raise EvidenceError('invalid selected GPU')
    for k in ('minVcpuCount','minMemoryInGb'):
        if type(payload[k]) is not int or not 1<=payload[k]<=1024:raise EvidenceError('invalid selected resource minimum')
    # No env/key/registry credential fields are accepted in the published intent.
    b=w['baseline'];balance=Decimal(b['balance_usd'])
    needed=sum(Decimal(p['input'][k]) for k in ('outstanding_usd','reserved_remaining_usd'))+Decimal(p['maximum_charge_micro_usd']+p['protected_reserve_micro_usd'])/10**6
    if not balance.is_finite() or balance<needed:raise EvidenceError('account cannot fund bounded rental and protected work')
    return w,p


def watchdog_heartbeat(path,w,now,known=None):
    h=read_json(path)
    if (h.get('schema')!='ovl.external-watchdog-heartbeat.v1' or h.get('intent_sha256')!=digest(w)
        or h.get('plan_sha256')!=digest(w['plan']) or h.get('external_terminate_epoch')!=w['plan']['external_terminate_epoch']
        or type(h.get('observed_epoch')) is not int or not 0<=now-h['observed_epoch']<=30
        or h.get('state')!='ARMED' or h.get('pod_id') not in (None,known)):
        raise EvidenceError('missing/stale/inconsistent armed external watchdog')
    return h


def dollars(value,*,balance=False):
    return format(value.quantize(Decimal('0.000001'),rounding=ROUND_FLOOR if balance else ROUND_CEILING),'f')


def normalized(w,obs,pod,h,health,now):
    """Conservative unsettled exposure, retaining all prior-work reservations.

    Account debit may include prior settlements; taking its maximum with the
    lifetime upper-rate accrual can overreserve, but cannot intentionally hide it.
    Final provider-attributed bills still require independent reconciliation.
    """
    p=w['plan'];baseline=Decimal(w['baseline']['balance_usd']);balance=Decimal(obs['balance_usd'])
    debit=max(Decimal(0),baseline-balance)
    elapsed=max(0,now-p['input']['now_epoch'])
    accrual=Decimal(elapsed)*Decimal(p['input']['hourly_upper_usd'])/3600
    outstanding=Decimal(p['input']['outstanding_usd'])+max(debit,accrual)
    if health is None:progress=checkpoint=p['input']['now_epoch']
    else:
        fields(health,'schema intent_sha256 pod_id observed_epoch progress_epoch exported_checkpoint_epoch complete','workload health')
        if (health['schema']!='ovl.rental-workload-health.v1' or health['intent_sha256']!=digest(w)
            or health['pod_id']!=pod['id'] or type(health['observed_epoch']) is not int
            or not 0<=now-health['observed_epoch']<=60 or type(health['complete']) is not bool):
            raise EvidenceError('invalid/stale workload and export health')
        progress=health['progress_epoch'];checkpoint=health['exported_checkpoint_epoch']
        if health['complete'] and (type(progress) is not int or type(checkpoint) is not int or checkpoint<progress):
            raise EvidenceError('completed work must have its final state exported')
    return {'schema':'ovl.supervisor-observation.v2','now_epoch':now,'observed_epoch':obs['observed_epoch'],
        'attributed_pod_ids':[pod['id']],'active_pod_ids':[x['id'] for x in obs['pods']],
        'pod_id':pod['id'],'gpu_count':pod['gpuCount'],
        'hourly_usd':dollars(max(Decimal(pod['costPerHr']),Decimal(pod['adjustedCostPerHr']),Decimal(obs['account_hourly_usd']))),
        'actual_project_spend_usd':p['input']['spent_usd'],'outstanding_usd':dollars(outstanding),
        'reserved_remaining_usd':p['input']['reserved_remaining_usd'],'account_balance_usd':dollars(balance,balance=True),
        'progress_epoch':progress,'last_checkpoint_epoch':checkpoint,
        'terminate_after_request_epoch':p['provider_terminate_epoch'],'watchdog_observed_epoch':h['observed_epoch'],
        'watchdog_plan_sha256':h['plan_sha256'],'watchdog_external_terminate_epoch':h['external_terminate_epoch'],'watchdog_state':h['state']}


def run(directory,value,expected,heartbeat,health_path,*,get_account=account,provider_request=request,
        wall=time.time,monotonic=time.monotonic,sleep=time.sleep):
    w,p=validate(value,expected)
    with Journal(directory).lease() as j:
        intents=[e['body'] for e in j.events if e['kind']=='creation-intent']
        if intents and intents!=[value]:raise EvidenceError('controller journal intent mismatch')
        if any(e['kind']=='teardown' and e['body'].get('complete') is True for e in j.events):return
        knowns={e['body']['id'] for e in j.events if e['kind']=='creation-observed'}
        if len(knowns)>1:raise EvidenceError('ambiguous recorded identity')
        known=next(iter(knowns),None)
        stopping=next((e['body']['observed_epoch'] for e in j.events if e['kind']=='decision' and e['body'].get('action')=='CHECKPOINT_AND_STOP'),None)
        terminating=any(e['kind']=='decision' and e['body'].get('action')=='TERMINATE' for e in j.events)
        missing_since=None
        mono_deadline=monotonic()+max(0,min(p['external_terminate_epoch']-wall(),p['external_terminate_epoch']-p['input']['now_epoch']))
        def log(kind,body):
            try:j.append(kind,body)
            except Exception:return False
            return True
        def terminate(reason):
            nonlocal terminating,known
            terminating=True;log('decision',{'action':'TERMINATE','reason':reason,'observed_epoch':int(wall()),'pod_id':known})
            if known is None:
                try:
                    d,_,_=provider_request('identities');matches=[x for x in d['myself']['pods'] if x['name']==w['payload']['name']]
                    if len(matches)==1 and re.fullmatch('[A-Za-z0-9_-]{1,96}',matches[0]['id']):
                        known=matches[0]['id'];log('creation-observed',{'id':known,'adopted_from_unique_intent':True})
                except Exception as e:log('failure',{'stage':'identity-reconciliation',**diagnostic(e)})
            if known is not None:
                try:
                    _,h,_=provider_request('terminate',{'input':{'podId':known}})
                    log('teardown',{'complete':False,'pod_id':known,'response_sha256':h})
                except Exception as e:log('failure',{'stage':'termination',**diagnostic(e)})
        if not intents:
            # Creation is authorized only within the short fixed window, with a
            # fresh empty account read and separately armed watchdog. Write once
            # before sending; a crash after this event can NEVER reissue creation.
            now=int(wall());watchdog_heartbeat(heartbeat,w,now)
            obs=get_account()
            if (obs['pods'] or obs['volume_ids'] or obs['autopay'] or Decimal(obs['account_hourly_usd'])!=0
                or not 0<=wall()-obs['observed_epoch']<=25 or Decimal(obs['balance_usd'])<Decimal(w['baseline']['balance_usd'])):
                raise EvidenceError('account changed before one-shot creation')
            clock=obs['http_clock']
            if not clock['request_started_epoch']-5<=clock['server_epoch']<=clock['request_completed_epoch']+5:
                raise EvidenceError('provider clock differs before creation')
            j.append('creation-intent',value)
            try:
                if not p['input']['now_epoch']<=wall()<=w['creation_latest_epoch']:raise EvidenceError('creation window expired')
                watchdog_heartbeat(heartbeat,w,int(wall()))
                d,h,_=provider_request('create',{'input':value['payload']})
                pod=d['podFindAndDeployOnDemand']
                if not re.fullmatch('[A-Za-z0-9_-]{1,96}',pod['id']):raise EvidenceError('invalid created ID')
                known=pod['id'];j.append('creation-observed',{'id':known,'response_sha256':h})
            except Exception as e:
                log('failure',{'stage':'creation','reissue':'FORBIDDEN',**diagnostic(e)});terminate('uncertain-or-failed-creation')
        while True:
            now=int(wall())
            if terminating or now>=p['external_terminate_epoch'] or monotonic()>=mono_deadline or now<p['input']['now_epoch']-5:
                terminate('deadline-or-prior-abort-or-clock-rollback')
            left=min(p['external_terminate_epoch']-wall(),mono_deadline-monotonic())
            if known is not None and not terminating and 0<left<=21:sleep(min(5,left));continue
            try:
                obs=get_account();pod=match_pod(w,obs,known)
                if not 0<=wall()-obs['observed_epoch']<=25:raise EvidenceError('stale provider read')
                clock=obs['http_clock']
                if not clock['request_started_epoch']-5<=clock['server_epoch']<=clock['request_completed_epoch']+5:raise EvidenceError('provider clock differs')
                if pod is None:
                    if known is not None or wall()>p['external_terminate_epoch']+180:
                        if missing_since is None:missing_since=obs['observed_epoch']
                        elif obs['observed_epoch']-missing_since>=15:
                            report={'schema':'ovl.rental-controller-result.v1','complete':True,'pod_id':known,'intent_sha256':expected,
                                'confirmed_absent_epoch':obs['observed_epoch'],'residual_network_volumes':obs['volume_ids'],
                                'automatic_provider_termination':'UNVERIFIED','provider_billing_reconciliation':'PENDING','training_admission':'NOT_RUN'}
                            j.append('teardown',report);write_json(directory/'result.json',report);return
                else:
                    missing_since=None
                    if known is None:
                        known=pod['id'];j.append('creation-observed',{'id':known,'adopted_from_unique_intent':True})
                    if provision_errors(w,pod) or obs['volume_ids'] or obs['autopay'] or len(obs['pods'])!=1:
                        raise EvidenceError('resource shape/account singleton changed')
                    h=watchdog_heartbeat(heartbeat,w,int(wall()),known)
                    health=read_json(health_path) if health_path.exists() else None
                    v=normalized(w,obs,pod,h,health,int(wall()));decision=observe(p,v)
                    decision['observed_epoch']=int(wall())
                    debit=max(Decimal(0),Decimal(w['baseline']['balance_usd'])-Decimal(obs['balance_usd']))
                    if debit>Decimal(p['maximum_charge_micro_usd'])/10**6:raise EvidenceError('observed rental debit exceeded reserved ceiling')
                    if not log('provider-observation',{'account':obs,'normalized':v}):raise EvidenceError('cannot preserve cost observation')
                    if health is not None and health['complete']:terminate('workload-complete-exported')
                    elif decision['action']=='TERMINATE':terminate('supervision-hard-deadline')
                    elif decision['action']=='CHECKPOINT_AND_STOP':
                        if stopping is None:
                            stopping=int(wall());j.append('decision',decision)
                            write_json(directory/'stop-request.json',{'schema':'ovl.rental-stop-request.v1','intent_sha256':expected,'pod_id':known,'observed_epoch':stopping,'reasons':decision['reasons']})
                        # Dispatcher observes stop request and exports, but cannot
                        # renew the fixed graceful-stop interval by refreshing health.
                        if wall()>=min(stopping+p['input']['checkpoint_grace_seconds'],p['provider_terminate_epoch']):
                            terminate('graceful-stop-interval-exhausted')
                    elif stopping is not None and wall()>=min(stopping+p['input']['checkpoint_grace_seconds'],p['provider_terminate_epoch']):
                        terminate('prior-stop-remains-binding')
                if not log('provider-observation',{'account':obs,'pod_id':known}):raise EvidenceError('cannot preserve account observation')
            except Exception as e:
                log('failure',{'stage':'supervision',**diagnostic(e)});terminate('invalid-observation-or-evidence')
            sleep(5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('intent','journal','watchdog-heartbeat','workload-health'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--intent-sha256',required=True);a=p.parse_args()
    run(a.journal,read_json(a.intent),a.intent_sha256,a.watchdog_heartbeat,a.workload_health)

if __name__=='__main__':main()
