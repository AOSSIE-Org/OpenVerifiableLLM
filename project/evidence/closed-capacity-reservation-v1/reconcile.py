"""Reconcile only an explicit rejected creation after its ORIGINAL guard closes.

Operator/provider observations are trusted; this is not a final invoice or a
cryptographic proof of noncreation. The exact original private provider response
is checked locally and its existing public redacted receipt is linked. Never
publish that private response/account identifier. No provider/service mutations.
"""
from decimal import Decimal
from ovl_pipeline.canonical import EvidenceError,digest
from ovl_pipeline.schema import fields,integer
from reconcile_capacity_rejection import verify as original_verify
from run_external_watchdog import match_pod


def insist(value,why):
    if not value:raise EvidenceError(why)


def verify(original_args,receipt,controller_events,watchdog_events,result,observations,services,now):
    # Recheck the original complete rejection, request, identity, journal prefix
    # and then-live unchanged watchdog. A stored success assertion is insufficient.
    insist(original_verify(*original_args)==receipt,'original capacity evidence differs')
    rental=original_args[0];w=rental['watchdog_intent'];p=w['plan']
    integer(now,p['external_terminate_epoch']+196,2**53-1,'post-original-closure time')
    prefix=original_args[5]
    insist(controller_events[:len(prefix)]==prefix,'creator journal lost original prefix')
    insist([e['body'] for e in controller_events if e['kind']=='creation-intent']==[rental],'creator identity changed')
    insist(not any(e['kind']=='creation-observed' for e in controller_events),'creator observed a resource')
    insist(len([e for e in controller_events if e['kind']=='decision' and e['body'].get('action')=='CREATION_FENCED'])==1,'creation fence changed')
    insist([e['body'] for e in watchdog_events if e['kind']=='creation-intent']==[w],'watchdog intent changed')
    insist(not any(e['kind']=='creation-observed' for e in watchdog_events),'watchdog observed a resource')
    insist(watchdog_events[-1]['kind']=='teardown' and watchdog_events[-1]['body']==result,'missing terminal watchdog journal result')
    fields(result,'schema complete pod_id intent_sha256 confirmed_absent_epoch automatic_provider_termination external_termination_requested residual_network_volumes provider_billing_reconciliation execution_admission','closed no-ID watchdog')
    insist(result['schema']=='ovl.external-watchdog-result.v1' and result['complete'] is True
       and result['pod_id'] is None and result['intent_sha256']==digest(w)
       and result['automatic_provider_termination']=='UNVERIFIED'
       and result['external_termination_requested'] is True and result['residual_network_volumes']==[]
       and result['provider_billing_reconciliation']=='PENDING' and result['execution_admission']=='NOT_RUN','invalid closed no-ID result')
    closed=result['confirmed_absent_epoch'];integer(closed,p['external_terminate_epoch']+196,now,'original watchdog closure')
    insist(any(e['kind']=='decision' and e['body'].get('action')=='TERMINATE' for e in watchdog_events),'missing original guard deadline action')
    absences=[]
    for event in watchdog_events:
        if event['kind']=='provider-observation':
            insist(match_pod(w,event['body'],None) is None,'matching resource appeared in original watchdog history')
            o=event['body']
            if o['pods']==[] and o['volume_ids']==[] and Decimal(o['account_hourly_usd'])==0:
                absences.append(o['observed_epoch'])
    insist(any(p['external_terminate_epoch']+180<t<=closed-15 for t in absences),'missing preceding post-deadline empty read')
    insist(services=={'controller':'inactive','watchdog':'inactive'},'both original services must be inactive')
    insist(type(observations) is list and len(observations)==2,'two fresh same-account reads required')
    for o in observations:
        fields(o,'schema account_identity_sha256 response_sha256 http_clock observed_epoch pods volume_ids autopay account_hourly_usd','post-closure account')
        insist(o['schema']=='ovl.empty-account-identity-observation.v1'
            and o['account_identity_sha256']==receipt['account_identity_sha256']
            and o['pods']==[] and o['volume_ids']==[] and o['autopay'] is False and o['account_hourly_usd']=='0','post-closure identity/resource state differs')
        from ovl_pipeline.canonical import require_digest
        require_digest(o['response_sha256']);integer(o['observed_epoch'],closed,now,'post-closure observation')
        c=o['http_clock'];fields(c,'server_epoch request_started_epoch request_completed_epoch','provider clock')
        for value in c.values():integer(value,1,2**53-1,'provider clock value')
        insist(closed<=c['request_started_epoch']<=c['request_completed_epoch']<=o['observed_epoch'],'read predates closure')
        insist(c['request_started_epoch']-5<=c['server_epoch']<=c['request_completed_epoch']+5,'provider clock mismatch')
    insist(observations[1]['observed_epoch']-observations[0]['observed_epoch']>=15 and now-observations[1]['observed_epoch']<=30,'fresh separated closure observations required')
    return {'schema':'ovl.closed-capacity-reservation-reconciliation.v1','result':'PASS',
        'rental_intent_sha256':digest(rental),'original_rejection_sha256':digest(receipt),
        'original_external_deadline_epoch':p['external_terminate_epoch'],'confirmed_absent_epoch':closed,
        'controller_events_sha256':digest(controller_events),'watchdog_events_sha256':digest(watchdog_events),
        'watchdog_result_sha256':digest(result),'post_closure_observations_sha256':digest(observations),
        'released_unused_creation_allowance_usd':format(Decimal(p['maximum_charge_micro_usd'])/10**6,'f'),
        'remaining_creation_reservation_usd':'0','provider_final_settlement':'NOT_ASSERTED',
        'accounting_basis':'Exact authenticated provider capacity rejection, no matching resource in original retained guard history, original no-ID deadline closure and fresh same-account empty/zero-rate reads. Operator/provider trust; not proof against a dishonest provider or final invoice.',
        'creation_fence':'PRESERVE_NEVER_REISSUE_ORIGINAL_REQUEST','provider_mutation':'NOT_RUN',
        'training_verification_credit':False}
