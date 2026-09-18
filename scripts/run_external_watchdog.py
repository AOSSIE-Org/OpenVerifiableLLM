#!/usr/bin/env python3
"""Persistent off-pod teardown controller; never creates a resource or trains.

Start under systemd Restart=always before issuing the separately journalled single
creation request. A fresh ARMED heartbeat is a prerequisite for that request. The
creation controller must also supervise spend/progress and may request earlier
teardown. This watchdog enforces an absolute lifetime even if that controller dies.
Provider automatic termination stays UNVERIFIED. Loss of this entire external host
or all provider API connectivity can prevent teardown; this is not a provider SLA.
"""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
import os
from pathlib import Path
import re
import time

from ovl_pipeline.canonical import EvidenceError,digest,read_json,require_digest,write_json
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.supervision import ControllerBusy,Journal,rental_plan
from probe_provider_deadline import account,request,diagnostic,match_pod,provision_errors,transient_read_grace
from rental_safety import Lifetime,boot_clock,attributed_ids


def validate_intent(value,expected):
    require_digest(expected)
    if digest(value)!=expected:raise EvidenceError('watchdog intent differs from caller pin')
    fields(value,'schema plan payload creation_latest_epoch baseline','watchdog intent')
    if value['schema']!='ovl.external-watchdog-intent.v1':raise EvidenceError('unsupported watchdog intent')
    p=value['plan']
    if p!=rental_plan(p['input']) or p['schema']!='ovl.rental-budget-plan.v2':raise EvidenceError('revised bounded rental plan required')
    payload=value['payload']
    fields(payload,'name gpuCount imageName containerDiskInGb volumeInGb terminateAfter','watchdog resource identity')
    if payload['name']!=p['input']['attempt_id']:raise EvidenceError('resource name differs from budget attempt')
    if not re.fullmatch(r'ovllm-[a-z0-9-]*[0-9a-f]{32}',payload['name']):raise EvidenceError('unique UUID-suffixed project attempt name required')
    if type(payload['gpuCount']) is not int or payload['gpuCount']!=1:raise EvidenceError('one GPU required')
    if type(payload['imageName']) is not str or not re.fullmatch(r'[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}',payload['imageName']):raise EvidenceError('immutable image required')
    for k in ('containerDiskInGb','volumeInGb'):integer(payload[k],0,1024,k)
    if payload['terminateAfter']!=datetime.fromtimestamp(p['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'):
        raise EvidenceError('terminateAfter must request exact planned deadline')
    integer(value['creation_latest_epoch'],p['input']['now_epoch']+1,p['input']['now_epoch']+120,'creation window')
    b=value['baseline']
    if b['pods'] or b['volume_ids'] or b['autopay'] is not False or Decimal(b['account_hourly_usd'])!=0:
        raise EvidenceError('empty zero-rate account baseline required')
    if not 0<=p['input']['now_epoch']-b['observed_epoch']<=60:raise EvidenceError('fresh baseline required')
    clock=b['http_clock']
    if not clock['request_started_epoch']-5<=clock['server_epoch']<=clock['request_completed_epoch']+5:
        raise EvidenceError('baseline provider/local clock disagreement')
    return p


def run(directory,intent,expected,*,get_account=account,provider_request=request,
        wall=time.time,monotonic=time.monotonic,sleep=time.sleep,clock=boot_clock):
    p=validate_intent(intent,expected)
    with Journal(directory).lease() as j:
        prior=[e['body'] for e in j.events if e['kind']=='creation-intent']
        if not prior:j.append('creation-intent',intent)
        elif prior!=[intent]:raise EvidenceError('watchdog journal has different or multiple intent')
        if any(e['kind']=='teardown' and e['body'].get('complete') is True for e in j.events):return
        identities={e['body']['id'] for e in j.events if e['kind']=='creation-observed'}
        if len(identities)>1:raise EvidenceError('multiple attributed resource identities')
        known=next(iter(identities),None)
        terminating=any(e['kind']=='decision' and e['body'].get('action')=='TERMINATE' for e in j.events)
        lifetime=Lifetime(j,p,wall=wall,clock=clock,initialize=not prior)
        remaining=lifetime.remaining()
        mono_deadline=monotonic()+remaining
        missing_since=None
        last_success=None;last_success_monotonic=None
        def log(kind,body):
            # Evidence storage failure must not suppress attributed teardown.
            try:j.append(kind,body)
            except Exception:return False
            return True
        def terminate(reason,*,reconcile=False):
            nonlocal terminating,known
            terminating=True;log('decision',{'action':'TERMINATE','reason':reason,'pod_id':known})
            if known is not None:
                try:
                    _,h,_=provider_request('terminate',{'input':{'podId':known}})
                    log('teardown',{'complete':False,'pod_id':known,'response_sha256':h})
                except Exception as e:log('failure',{'stage':'termination',**diagnostic(e)})
            if known is None or reconcile:
                try:
                    matches=attributed_ids(provider_request,intent['payload']['name'])
                    if known is None and matches:
                        known=matches[0];log('creation-observed',{'id':known,'adopted_from_unique_intent':True})
                    for pod_id in matches:
                        # Repeating the first ID is safe; do not let ambiguous
                        # provider duplicates disable cleanup of all exact names.
                        _,h,_=provider_request('terminate',{'input':{'podId':pod_id}})
                        log('teardown',{'complete':False,'pod_id':pod_id,'response_sha256':h})
                except Exception as e:log('failure',{'stage':'identity-reconciliation',**diagnostic(e)})
        while True:
            now=wall()
            # Do not wait for a potentially slow read before a due known-ID teardown.
            if terminating or lifetime.remaining()<=0 or now>=p['external_terminate_epoch'] or monotonic()>=mono_deadline or now<p['input']['now_epoch']-5:
                terminate('deadline-or-prior-abort-or-clock-rollback')
            # Avoid starting a 20s account read that crosses the deadline. The
            # loop wakes at the deadline itself; its monotonic limit cannot extend.
            left=min(lifetime.remaining(),p['external_terminate_epoch']-wall(),mono_deadline-monotonic())
            if known is not None and not terminating and 0<left<=21:
                sleep(min(5,left));continue
            stage='account'
            try:
                obs=get_account();stage='observation';pod=match_pod(intent,obs,known)
                clock=obs['http_clock']
                if not clock['request_started_epoch']-5<=clock['server_epoch']<=clock['request_completed_epoch']+5:
                    raise EvidenceError('provider clock mismatch')
                if not 0<=wall()-obs['observed_epoch']<=25:raise EvidenceError('stale account read')
                last_success=obs['observed_epoch'];last_success_monotonic=monotonic()
                if pod is not None:
                    missing_since=None
                    if known is None:
                        known=pod['id']
                        if not log('creation-observed',{'id':known,'adopted_from_unique_intent':True}):
                            terminate('identity-journal-failure')
                    unrelated=[x['id'] for x in obs['pods'] if x['id']!=known]
                    excessive=max(Decimal(pod['costPerHr']),Decimal(pod['adjustedCostPerHr']),Decimal(obs['account_hourly_usd']))>Decimal(p['input']['hourly_upper_usd'])
                    if provision_errors(intent,pod) or unrelated or obs['volume_ids'] or obs['autopay'] or excessive:
                        terminate('resource-shape-or-budget-guard')
                    if terminating or wall()>=p['external_terminate_epoch'] or monotonic()>=mono_deadline:
                        terminate('late-discovered-resource-or-deadline')
                elif known is not None or wall()>p['external_terminate_epoch']+180:
                    # Never interpret pre-creation absence as completed termination.
                    if missing_since is None:missing_since=obs['observed_epoch']
                    elif obs['observed_epoch']-missing_since>=15:
                        report={'schema':'ovl.external-watchdog-result.v1','complete':True,
                                'pod_id':known,'intent_sha256':expected,'confirmed_absent_epoch':obs['observed_epoch'],
                                'automatic_provider_termination':'UNVERIFIED',
                                'external_termination_requested':terminating,'residual_network_volumes':obs['volume_ids'],
                                'provider_billing_reconciliation':'PENDING','execution_admission':'NOT_RUN'}
                        j.append('teardown',report);write_json(directory/'result.json',report);return
                if not log('provider-observation',obs):terminate('observation-journal-failure')
                write_json(directory/'heartbeat.json',{'schema':'ovl.external-watchdog-heartbeat.v1',
                           'observed_epoch':int(wall()),'pid':os.getpid(),'pod_id':known,'intent_sha256':expected,
                           'plan_sha256':digest(p),'external_terminate_epoch':p['external_terminate_epoch'],
                           'state':'TERMINATING' if terminating else 'ARMED',
                           'automatic_provider_termination':'UNVERIFIED'})
            except Exception as e:
                if (stage=='account' and lifetime.remaining()>25 and transient_read_grace(e,wall(),last_success,monotonic(),last_success_monotonic,p['external_terminate_epoch'],terminating)):
                    if log('failure',{'stage':'transient-account-read',**diagnostic(e),'action':'bounded-read-retry'}):
                        write_json(directory/'heartbeat.json',{'schema':'ovl.external-watchdog-heartbeat.v1','observed_epoch':int(wall()),
                            'provider_observed_epoch':last_success,'pid':os.getpid(),'pod_id':known,'intent_sha256':expected,
                            'plan_sha256':digest(p),'external_terminate_epoch':p['external_terminate_epoch'],'state':'ARMED',
                            'automatic_provider_termination':'UNVERIFIED'})
                        sleep(5);continue
                terminate('observation-or-evidence-failure',reconcile=True);log('failure',{'stage':'observation',**diagnostic(e)})
            sleep(max(.01,min(10,p['external_terminate_epoch']-wall())) if not terminating else 5)


def run_guarded(directory,intent,expected,*,provider_request=request,**kwargs):
    """A damaged/unwritable journal must not strand a uniquely attributed pod."""
    validate_intent(intent,expected)  # Never derive authority from broken evidence.
    try:return run(directory,intent,expected,provider_request=provider_request,**kwargs)
    except ControllerBusy:raise  # A duplicate must leave the live owner alone.
    except Exception:
        try:
            for pod_id in attributed_ids(provider_request,intent['payload']['name']):
                provider_request('terminate',{'input':{'podId':pod_id}})
        except Exception:pass
        # Restarting service continues reconciliation; never claim a request is
        # successful termination or overwrite the damaged journal to make it pass.
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intent',required=True,type=Path);parser.add_argument('--intent-sha256',required=True)
    parser.add_argument('--journal',required=True,type=Path);a=parser.parse_args()
    run_guarded(a.journal,read_json(a.intent),a.intent_sha256)

if __name__=='__main__':main()
