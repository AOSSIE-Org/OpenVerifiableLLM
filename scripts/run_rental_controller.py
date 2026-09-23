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
from ovl_pipeline.schema import fields,integer
from ovl_pipeline.supervision import ControllerBusy,Journal,observe
from probe_provider_deadline import account,request,match_pod,provision_errors,diagnostic,transient_read_grace
from run_external_watchdog import validate_intent
from rental_safety import Lifetime,boot_clock,attributed_ids,account_lease,fence
from rental_quote import validate_quote
import retained_volume


def validate(value,expected):
    require_digest(expected)
    if digest(value)!=expected:raise EvidenceError('rental intent differs from selected pin')
    fields(value,'schema watchdog_intent payload quote','rental intent')
    if value['schema'] not in ('ovl.rental-controller-intent.v1','ovl.rental-controller-intent.v2','ovl.rental-controller-intent.v3','ovl.rental-controller-intent.v4','ovl.rental-controller-intent.v5','ovl.rental-controller-intent.v6'):raise EvidenceError('unsupported rental controller intent')
    w=value['watchdog_intent'];p=validate_intent(w,digest(w));payload=value['payload']
    retained=value['schema']=='ovl.rental-controller-intent.v6'
    if (w['schema']=='ovl.external-watchdog-intent.v2')!=retained:raise EvidenceError('watchdog storage version differs')
    names='name gpuCount imageName containerDiskInGb volumeInGb terminateAfter cloudType gpuTypeId minVcpuCount minMemoryInGb dockerArgs startSsh startJupyter ports'
    explicit_cuda=value['schema']!='ovl.rental-controller-intent.v1'
    if explicit_cuda:names+=' allowedCudaVersions'
    network_minimums=value['schema'] in ('ovl.rental-controller-intent.v4','ovl.rental-controller-intent.v5')
    if network_minimums:names+=' minDownload minUpload'
    country_selection=value['schema']=='ovl.rental-controller-intent.v5'
    if country_selection:names+=' countryCode'
    if retained:names+=' networkVolumeId dataCenterId volumeMountPath'
    fields(payload,names,'creation payload')
    if country_selection and (type(payload['countryCode']) is not str or not re.fullmatch(r'[A-Z]{2}',payload['countryCode'])):
        raise EvidenceError('explicit two-letter country placement required')
    if network_minimums:
        # Provider placement requests, never measured transport or runtime credit.
        for key in ('minDownload','minUpload'):
            integer(payload[key],1,100000,key)
    if explicit_cuda:
        versions=payload['allowedCudaVersions']
        if (type(versions) is not list or not 1<=len(versions)<=16
            or any(type(v) is not str or not re.fullmatch(r'13\.[0-9]{1,2}',v) for v in versions)
            or versions!=sorted(set(versions),key=lambda v:int(v.split('.')[1]))):
            raise EvidenceError('explicit unique ordered CUDA13 host versions required')
    if any(payload[k]!=v for k,v in w['payload'].items()):raise EvidenceError('creation differs from watchdog identity/deadline')
    clouds=('SECURE','COMMUNITY') if value['schema'] in ('ovl.rental-controller-intent.v3','ovl.rental-controller-intent.v4','ovl.rental-controller-intent.v5') else ('SECURE',)
    if payload['cloudType'] not in clouds or payload['startSsh'] is not True or payload['startJupyter'] is not False:
        raise EvidenceError('selected cloud and SSH-only rental required')
    quote_schema='ovl.rental-quote.v2' if value['schema'] in ('ovl.rental-controller-intent.v3','ovl.rental-controller-intent.v4','ovl.rental-controller-intent.v5') else 'ovl.rental-quote.v1'
    if retained:quote_schema='ovl.rental-quote.v3'
    if value['quote'].get('schema')!=quote_schema:raise EvidenceError('quote version differs from rental intent')
    if payload['ports']!='22/tcp':raise EvidenceError('only SSH port may be exposed')
    if payload['dockerArgs']!='':raise EvidenceError('only immutable image entrypoint may start')
    if type(payload['gpuTypeId']) is not str or not re.fullmatch(r'[A-Za-z0-9 ._-]{1,96}',payload['gpuTypeId']):raise EvidenceError('invalid selected GPU')
    for k in ('minVcpuCount','minMemoryInGb'):
        if type(payload[k]) is not int or not 1<=payload[k]<=1024:raise EvidenceError('invalid selected resource minimum')
    validate_quote(value['quote'],payload,p,network_volume=w.get('retained_volume'))
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


def restore_storage_stop(directory,events,expected,known):
    """Recover a journaled local stop without renewing its original timestamp."""
    stops=[e['body'] for e in events if e['kind']=='decision'
           and e['body'].get('action')=='CHECKPOINT_AND_STOP'
           and e['body'].get('reasons')==['local-storage-headroom']]
    if not stops:return
    if len(stops)!=1 or known is None or stops[0].get('pod_id')!=known:
        raise EvidenceError('local storage stop identity differs')
    epoch=stops[0]['observed_epoch'];integer(epoch,1,2**53-1,'original storage stop epoch')
    desired={'schema':'ovl.rental-stop-request.v1','intent_sha256':expected,'pod_id':known,
             'observed_epoch':epoch,'reasons':['local-storage-headroom']}
    path=directory/'stop-request.json'
    if path.exists():
        if read_json(path)!=desired:raise EvidenceError('original storage stop request differs')
    else:write_json(path,desired)


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
    outstanding=max(Decimal(p['input']['outstanding_usd'])+accrual,debit)
    if health is None:progress=checkpoint=p['input']['now_epoch']
    else:
        fields(health,'schema intent_sha256 pod_id observed_epoch progress_epoch exported_checkpoint_epoch complete','workload health')
        if (health['schema']!='ovl.rental-workload-health.v1' or health['intent_sha256']!=digest(w)
            or health['pod_id']!=pod['id'] or type(health['observed_epoch']) is not int
            or not 0<=now-health['observed_epoch']<=60 or type(health['complete']) is not bool):
            raise EvidenceError('invalid/stale workload and export health')
        progress=health['progress_epoch'];checkpoint=health['exported_checkpoint_epoch']
        integer(progress,1,2**53-1,'workload progress epoch')
        integer(checkpoint,1,2**53-1,'workload export epoch')
        if health['complete'] and (type(progress) is not int or type(checkpoint) is not int or checkpoint<progress):
            raise EvidenceError('completed work must have its final state exported')
    return {'schema':'ovl.supervisor-observation.v2','now_epoch':now,'observed_epoch':obs['observed_epoch'],
        'attributed_pod_ids':[pod['id']],'active_pod_ids':[x['id'] for x in obs['pods']],
        'pod_id':pod['id'],'gpu_count':pod['gpuCount'],
        'hourly_usd':dollars(max(Decimal(pod['costPerHr']),Decimal(pod['adjustedCostPerHr']),retained_volume.compute_hourly(w,obs))),
        'actual_project_spend_usd':p['input']['spent_usd'],'outstanding_usd':dollars(outstanding),
        'reserved_remaining_usd':p['input']['reserved_remaining_usd'],'account_balance_usd':dollars(balance,balance=True),
        'progress_epoch':progress,'last_checkpoint_epoch':checkpoint,
        'terminate_after_request_epoch':p['provider_terminate_epoch'],'watchdog_observed_epoch':h['observed_epoch'],
        'watchdog_plan_sha256':h['plan_sha256'],'watchdog_external_terminate_epoch':h['external_terminate_epoch'],'watchdog_state':h['state']}


def run(directory,value,expected,heartbeat,health_path,*,get_account=account,provider_request=request,
        wall=time.time,monotonic=time.monotonic,sleep=time.sleep,boot=boot_clock,fence_root=None,storage_budget=None):
    w,p=validate(value,expected)
    with account_lease(fence_root) as fences,Journal(directory).lease() as j:
        intents=[e['body'] for e in j.events if e['kind']=='creation-intent']
        if intents and intents!=[value]:raise EvidenceError('controller journal intent mismatch')
        if any(e['kind']=='teardown' and e['body'].get('complete') is True for e in j.events):return
        storage_selection={'schema':'ovl.rental-local-storage-selection.v1','intent_sha256':expected,
                           'budget_sha256':None if storage_budget is None else digest(storage_budget.value)}
        storage_file=directory/'local-storage-selection.json'
        bindings=[e['body']['selection'] for e in j.events if e['kind']=='decision'
                  and e['body'].get('action')=='LOCAL_STORAGE_SELECTION']
        if bindings and bindings!=[storage_selection]:
            raise EvidenceError('original rental storage selection differs')
        if not bindings and not intents:
            j.append('decision',{'action':'LOCAL_STORAGE_SELECTION','selection':storage_selection})
            bindings=[storage_selection]
        if bindings and not storage_file.exists():write_json(storage_file,storage_selection)
        if storage_file.exists():
            if read_json(storage_file)!=storage_selection:raise EvidenceError('original rental storage selection differs')
        elif intents and storage_budget is not None:
            raise EvidenceError('existing rental cannot acquire an unbound storage policy')
        elif not intents:write_json(storage_file,storage_selection)
        if storage_budget is not None:
            storage_budget.covers(directory);storage_budget.covers(health_path)
        knowns={e['body']['id'] for e in j.events if e['kind']=='creation-observed'}
        if len(knowns)>1:raise EvidenceError('ambiguous recorded identity')
        known=next(iter(knowns),None)
        stopping=next((e['body']['observed_epoch'] for e in j.events if e['kind']=='decision' and e['body'].get('action')=='CHECKPOINT_AND_STOP'),None)
        terminating=any(e['kind']=='decision' and e['body'].get('action')=='TERMINATE' for e in j.events)
        missing_since=None
        last_success=None;last_success_monotonic=None
        storage_errors={reason for e in j.events if e['kind']=='failure' and e['body'].get('stage')=='retained-account-guard' for reason in e['body']['reasons']}
        def log(kind,body):
            try:j.append(kind,body)
            except Exception:return False
            return True
        def terminate(reason,*,reconcile=False):
            nonlocal terminating,known
            terminating=True;log('decision',{'action':'TERMINATE','reason':reason,'observed_epoch':int(wall()),'pod_id':known})
            if known is not None:
                try:
                    _,h,_=provider_request('terminate',{'input':{'podId':known}})
                    log('teardown',{'complete':False,'pod_id':known,'response_sha256':h})
                except Exception as e:log('failure',{'stage':'termination',**diagnostic(e)})
            if known is None or reconcile:
                try:
                    matches=attributed_ids(provider_request,w['payload']['name'])
                    if known is None and matches:
                        known=matches[0];log('creation-observed',{'id':known,'adopted_from_unique_intent':True})
                    for pod_id in matches:
                        _,h,_=provider_request('terminate',{'input':{'podId':pod_id}})
                        log('teardown',{'complete':False,'pod_id':pod_id,'response_sha256':h})
                except Exception as e:log('failure',{'stage':'identity-reconciliation',**diagnostic(e)})
        first=not intents and not (fences/(value['payload']['name']+'.json')).exists()
        if first:
            # Creation is authorized only within the short fixed window, with a
            # fresh empty account read and separately armed watchdog. Write once
            # before sending; a crash after this event can NEVER reissue creation.
            if storage_budget is not None:storage_budget.admit()
            now=int(wall());watchdog_heartbeat(heartbeat,w,now)
            obs=get_account()
            if 'retained_volume' in w:
                # Storage can legitimately debit the balance between two reads.
                # Recheck full funding; neither lower reserves nor renew deadlines.
                minimum_balance=sum(Decimal(p['input'][k]) for k in ('outstanding_usd','reserved_remaining_usd'))+Decimal(p['maximum_charge_micro_usd']+p['protected_reserve_micro_usd'])/10**6
            else:minimum_balance=Decimal(w['baseline']['balance_usd'])
            if (not retained_volume.baseline_valid(w,obs)
                or not 0<=wall()-obs['observed_epoch']<=25 or Decimal(obs['balance_usd'])<minimum_balance):
                raise EvidenceError('account changed before one-shot creation')
            if 'retained_volume' in w:
                debit=max(Decimal(0),Decimal(w['baseline']['balance_usd'])-Decimal(obs['balance_usd']))
                prior=Decimal(p['input']['outstanding_usd']);ceiling=Decimal(p['maximum_charge_micro_usd'])/10**6
                if debit>prior+ceiling+Decimal(w['retained_volume']['reserved_usd']):
                    raise EvidenceError('observed pre-creation debit exceeds reserved exposure')
                from ovl_pipeline.budget import OPERATING_LIMIT
                if Decimal(p['input']['spent_usd'])+max(prior,debit)+Decimal(p['input']['reserved_remaining_usd'])+ceiling>Decimal(OPERATING_LIMIT)/10**6:
                    raise EvidenceError('fresh account debit leaves insufficient operating budget')
            clock=obs['http_clock']
            if not clock['request_started_epoch']-5<=clock['server_epoch']<=clock['request_completed_epoch']+5:
                raise EvidenceError('provider clock differs before creation')
            j.append('creation-intent',value)
            lifetime=Lifetime(j,p,wall=wall,clock=boot,initialize=True)
            try:
                if not p['input']['now_epoch']<=wall()<=w['creation_latest_epoch']:raise EvidenceError('creation window expired')
                watchdog_heartbeat(heartbeat,w,int(wall()))
                if not fence(fences,value,j):raise EvidenceError('creation request already fenced; never repeat')
                d,h,_=provider_request('create',{'input':value['payload']})
                pod=d['podFindAndDeployOnDemand']
                if not re.fullmatch('[A-Za-z0-9_-]{1,96}',pod['id']):raise EvidenceError('invalid created ID')
                known=pod['id'];j.append('creation-observed',{'id':known,'response_sha256':h})
            except Exception as e:
                log('failure',{'stage':'creation','reissue':'FORBIDDEN',**diagnostic(e)});terminate('uncertain-or-failed-creation')
        else:
            if not intents:j.append('creation-intent',value)
            # A second journal may adopt/stop the same attempt, never authorize
            # another request. Legacy missing clock anchors stop immediately.
            if (fences/(value['payload']['name']+'.json')).exists():fence(fences,value,j)
            lifetime=Lifetime(j,p,wall=wall,clock=boot,initialize=False)
        mono_deadline=monotonic()+lifetime.remaining()
        restore_storage_stop(directory,j.events,expected,known)
        while True:
            now=int(wall())
            storage=None if storage_budget is None else storage_budget.observation()
            # A local low-space observation must not wait for provider recovery.
            # This uses the original stop protocol and original grace interval;
            # no health/deadline refresh, deletion or new rental is authorized.
            if storage is not None and storage['shutdown_needed'] and known is not None and stopping is None and not terminating:
                stopping=now
                j.append('decision',{'action':'CHECKPOINT_AND_STOP','reasons':['local-storage-headroom'],'observed_epoch':now,'pod_id':known,'storage':storage})
                restore_storage_stop(directory,j.events,expected,known)
            if storage is not None:
                try:write_json(directory/'local-storage-observation.json',{**storage,'observed_epoch':now})
                except OSError:
                    # This optional latest observation must not prevent delivery
                    # of a durable stop. Required cost/identity journaling below
                    # still fails closed if the filesystem cannot preserve it.
                    log('failure',{'stage':'local-storage-telemetry','action':'preserve-original-stop-and-guards'})
            stop_limit=None if stopping is None else min(stopping+p['input']['checkpoint_grace_seconds'],p['provider_terminate_epoch'])
            if (stop_limit is not None and now>=stop_limit) or terminating or lifetime.remaining()<=0 or now>=p['external_terminate_epoch'] or monotonic()>=mono_deadline or now<p['input']['now_epoch']-5:
                terminate('deadline-or-prior-abort-or-clock-rollback')
            left=min(lifetime.remaining(),p['external_terminate_epoch']-wall(),mono_deadline-monotonic())
            if stop_limit is not None:left=min(left,stop_limit-wall())
            if known is not None and not terminating and 0<left<=21:sleep(min(5,left));continue
            stage='account'
            try:
                obs=get_account();stage='account-identity'
                if 'retained_volume' in w:
                    violations=retained_volume.independent_errors(w,obs)
                    if violations:
                        storage_errors.update(violations)
                        log('failure',{'stage':'retained-account-guard','reasons':violations})
                stage='observation-journal'
                # Preserve independent storage failures before a serialization
                # failure in another field. All account journals remain private.
                if not log('provider-observation',{'account':obs,'validation':'PENDING'}):
                    raise EvidenceError('cannot preserve provider observation')
                stage='pod-attribution'
                pod=match_pod(w,obs,known)
                stage='provider-clock'
                if not 0<=wall()-obs['observed_epoch']<=25:raise EvidenceError('stale provider read')
                clock=obs['http_clock']
                if not clock['request_started_epoch']-5<=clock['server_epoch']<=clock['request_completed_epoch']+5:raise EvidenceError('provider clock differs')
                last_success=obs['observed_epoch'];last_success_monotonic=monotonic()
                stage='retained-account-guard'
                if 'retained_volume' in w:
                    violations=retained_volume.account_errors(w,obs,pod)
                    if violations:
                        storage_errors.update(violations)
                        log('failure',{'stage':'retained-account-guard','reasons':violations})
                        terminate('retained-account-guard',reconcile=known is None)
                if pod is None:
                    if known is not None or wall()>p['external_terminate_epoch']+180:
                        if missing_since is None:missing_since=obs['observed_epoch']
                        elif obs['observed_epoch']-missing_since>=15:
                            report={'schema':'ovl.rental-controller-result.v1','complete':True,'pod_id':known,'intent_sha256':expected,
                                'confirmed_absent_epoch':obs['observed_epoch'],'residual_network_volumes':obs['volume_ids'],
                                'provider_requested_only_fields':['cloudType','gpuTypeId','ports','startSsh','startJupyter','minVcpuCount','minMemoryInGb']+(['allowedCudaVersions'] if 'allowedCudaVersions' in value['payload'] else [])+(['minDownload','minUpload'] if value['schema'] in ('ovl.rental-controller-intent.v4','ovl.rental-controller-intent.v5') else [])+(['countryCode'] if value['schema']=='ovl.rental-controller-intent.v5' else []),
                                'runtime_identity_admission':'NOT_RUN',
                                'automatic_provider_termination':'UNVERIFIED','provider_billing_reconciliation':'PENDING','training_admission':'NOT_RUN'}
                            if 'retained_volume' in w:report.update(retained_storage_verification='FAIL' if storage_errors else 'PASS',account_guard_violations=sorted(storage_errors))
                            j.append('teardown',report);write_json(directory/'result.json',report);return
                else:
                    missing_since=None
                    if known is None:
                        known=pod['id'];j.append('creation-observed',{'id':known,'adopted_from_unique_intent':True})
                    stage='resource-shape';shape_errors=provision_errors(w,pod)
                    if shape_errors:
                        log('failure',{'stage':stage,'fields':shape_errors})
                    if shape_errors or not retained_volume.matches(w,obs) or obs['autopay'] or len(obs['pods'])!=1:
                        raise EvidenceError('resource shape/account singleton changed')
                    stage='watchdog-heartbeat'
                    h=watchdog_heartbeat(heartbeat,w,int(wall()),known)
                    stage='workload-health'
                    health_error=False
                    try:
                        health=read_json(health_path) if health_path.exists() else None
                        v=normalized(w,obs,pod,h,health,int(wall()))
                    except (EvidenceError,OSError,KeyError,TypeError):
                        health=None;health_error=True;v=normalized(w,obs,pod,h,None,int(wall()))
                    stage='cost-policy';decision=observe(p,v)
                    if health_error and decision['action']!='TERMINATE':
                        decision['action']='CHECKPOINT_AND_STOP';decision['reasons'].append('invalid-workload-export-health')
                    decision['observed_epoch']=int(wall())
                    debit=max(Decimal(0),Decimal(w['baseline']['balance_usd'])-Decimal(obs['balance_usd']))
                    storage_reserve=Decimal(w['retained_volume']['reserved_usd']) if 'retained_volume' in w else Decimal(0)
                    stage='account-debit'
                    if debit>Decimal(p['input']['outstanding_usd'])+Decimal(p['maximum_charge_micro_usd'])/10**6+storage_reserve:
                        raise EvidenceError('observed debit exceeded prior unsettled plus rental ceiling')
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
                if pod is None and not log('provider-observation',{'account':obs,'pod_id':known}):raise EvidenceError('cannot preserve account observation')
            except Exception as e:
                # Observation recovery must not postpone scheduled checkpointing
                # or an already requested graceful stop, even with no new reads.
                recovery_limit = min(p['external_terminate_epoch'],
                    p['request_checkpoint_epoch'] if stopping is None else
                    min(stopping+p['input']['checkpoint_grace_seconds'],p['provider_terminate_epoch']))
                if (stage=='account' and lifetime.remaining()>25 and transient_read_grace(e,wall(),last_success,monotonic(),last_success_monotonic,recovery_limit,terminating)):
                    if log('failure',{'stage':'transient-account-read',**diagnostic(e),'action':'bounded-read-retry',
                                     'provider_observed_epoch':last_success,'stop_limit_epoch':recovery_limit}):sleep(5);continue
                if stage=='account' and 'retained_volume' in w:
                    storage_errors.add('provider-observation-unverified')
                    log('failure',{'stage':'retained-account-guard','reasons':['provider-observation-unverified']})
                log('failure',{'stage':'supervision','failed_check':stage,**diagnostic(e)});terminate('invalid-observation-or-evidence',reconcile=True)
            sleep(5 if terminating else 10)


def run_guarded(directory,value,expected,heartbeat,health_path,*,provider_request=request,**kwargs):
    validate(value,expected)
    try:return run(directory,value,expected,heartbeat,health_path,provider_request=provider_request,**kwargs)
    except ControllerBusy:raise
    except Exception:
        try:
            for pod_id in attributed_ids(provider_request,value['payload']['name']):
                provider_request('terminate',{'input':{'podId':pod_id}})
        except Exception:pass
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('intent','journal','watchdog-heartbeat','workload-health'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--intent-sha256',required=True)
    p.add_argument('--local-storage-budget',type=Path);p.add_argument('--local-storage-budget-sha256');a=p.parse_args()
    from local_storage import Budget
    if (a.local_storage_budget is None)!=(a.local_storage_budget_sha256 is None):p.error('both local storage arguments required')
    budget=None if a.local_storage_budget is None else Budget(a.local_storage_budget,a.local_storage_budget_sha256)
    run_guarded(a.journal,read_json(a.intent),a.intent_sha256,a.watchdog_heartbeat,a.workload_health,storage_budget=budget)

if __name__=='__main__':main()
