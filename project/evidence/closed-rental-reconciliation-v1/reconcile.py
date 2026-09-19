"""Recompute reserved exposure for identified, confirmed-closed project rentals.

Provider/control-plane and operator clock observations are explicit trust inputs.
This is not a final invoice claim. Original rate margin and billing slack remain;
no unknown-ID attempt or active rental can receive a reduction from this checker.
Run from the trusted repository after extracting the separately pinned archive.
"""
from pathlib import Path
from decimal import Decimal,ROUND_CEILING
import json,sys
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json,file_hash
from ovl_pipeline.supervision import Journal
from run_rental_controller import validate


def insist(condition,reason):
    if not condition:raise EvidenceError(reason)


def unique(items):
    result={}
    for k,v in items:
        insist(k not in result,'duplicate billing key');result[k]=v
    return result


def dollars(value):
    insist(type(value) in (int,Decimal) and value>=0 and Decimal(value).is_finite(),'invalid billing amount')
    return Decimal(value)


def check(directory):
    directory=Path(directory);r=read_json(directory/'rental-intent.json');w,p=validate(r,digest(r))
    results=[];heads={};sources={};absences=[];zero_epochs=[]
    for owner,selected in [('controller',r),('watchdog',w)]:
        events=Journal(directory/owner)._read();report=read_json(directory/owner/'result.json')
        insist([e['body'] for e in events if e['kind']=='creation-intent']==[selected],'wrong original intent')
        insist(report['intent_sha256']==digest(selected),'wrong result intent')
        insist(events[-1]['kind']=='teardown' and events[-1]['body']==report,'result not final journal teardown')
        insist(report['complete'] is True and report['pod_id'] and not report['residual_network_volumes'],'identified closed resource without storage required')
        known={e['body']['id'] for e in events if e['kind']=='creation-observed'}
        insist(known=={report['pod_id']},'resource identity not unique')
        epoch=report['confirmed_absent_epoch']
        insist(type(epoch) is int and p['input']['now_epoch']<epoch<=p['external_terminate_epoch']+180,'closing clock outside original lifetime')
        missing=[];empty_zero=[]
        for e in events:
            if e['kind']!='provider-observation':continue
            account=e['body'].get('account',e['body'])
            if any(pod['id']==report['pod_id'] for pod in account['pods']):
                own=next(pod for pod in account['pods'] if pod['id']==report['pod_id'])
                insist(max(Decimal(own['costPerHr']),Decimal(own['adjustedCostPerHr']),Decimal(account['account_hourly_usd']))<=Decimal(p['input']['hourly_upper_usd']),'observed rate exceeded original bound')
                missing=[];empty_zero=[]
            elif not account['pods'] and not account['volume_ids']:
                missing.append(account['observed_epoch'])
                if Decimal(account['account_hourly_usd'])==0:empty_zero.append(account['observed_epoch'])
        insist(any(p['input']['now_epoch']<=t<=epoch-15 for t in missing),'no preceding empty-account observation')
        zero_epochs.extend(empty_zero)
        results.append(report);heads[owner]=digest(events[-1]);sources[owner]=file_hash(directory/owner/'result.json');absences.append(epoch)
    insist(results[0]['pod_id']==results[1]['pod_id'],'guards selected different resources')
    insist(w['payload']['volumeInGb']==0,'remaining local volume exposure is not covered')
    pod=results[0]['pod_id'];raw=(directory/'billing.json').read_bytes()
    def nonfinite(value):raise EvidenceError('nonfinite billing JSON')
    bill=json.loads(raw,parse_float=Decimal,object_pairs_hook=unique,parse_constant=nonfinite)
    insist(bill['metadata']['query']['podId']==pod and bill['metadata']['recordCount']==len(bill['records']),'foreign or incomplete billing selection')
    paid=dollars(bill['metadata']['totals']['totalAmount'])
    row_total=Decimal(0)
    for row in bill['records']:
        insist(row['podId']==pod,'foreign billed resource')
        row_total+=dollars(row['totalAmount'])
    # Provider aggregates can round binary decimal sums; reserve the larger
    # value, rounded upward, and retain the original reported amount separately.
    paid_upper=max(paid,row_total).quantize(Decimal('.000001'),rounding=ROUND_CEILING)
    closure_epoch=max(absences);zero_source={'kind':'closing-guard-journal','epochs':zero_epochs}
    if not zero_epochs:
        later=read_json(directory/'zero-rate-rental-intent.json');validate(later,digest(later))
        baseline=later['watchdog_intent']['baseline']
        insist(baseline['observed_epoch']>=closure_epoch,'zero-rate baseline precedes closure')
        insist(not baseline['pods'] and not baseline['volume_ids'] and Decimal(baseline['account_hourly_usd'])==0,'missing later zero-rate account')
        closure_epoch=baseline['observed_epoch']
        zero_source={'kind':'later-rental-empty-baseline','intent_sha256':digest(later),'observed_epoch':closure_epoch}
    elapsed=closure_epoch-p['input']['now_epoch']+p['input']['billing_slack_seconds']
    bound=(Decimal(p['input']['hourly_upper_usd'])*elapsed/3600).quantize(Decimal('.000001'),rounding=ROUND_CEILING)
    original=Decimal(p['maximum_charge_micro_usd'])/10**6
    elapsed_bound=bound;bound=min(bound,original)
    insist(paid_upper<=bound,'billing exceeds retained charge bound')
    return {'directory':directory.name,'pod_id':pod,'intent_sha256':digest(r),'journal_heads':heads,'result_sha256':sources,
        'billing_sha256':file_hash(directory/'billing.json'),'confirmed_absent_epoch':max(absences),'zero_rate_basis':zero_source,
        'conservative_closure_epoch':closure_epoch,'unclipped_elapsed_upper_usd':str(elapsed_bound),
        'full_original_allowance_retained':elapsed_bound>=original,
        'original_allowance_usd':str(original),'elapsed_including_original_billing_slack_seconds':elapsed,
        'unchanged_hourly_upper_usd':p['input']['hourly_upper_usd'],'closed_charge_upper_usd':str(bound),
        'reported_paid_usd':str(paid),'paid_upper_usd':str(paid_upper),'remaining_unposted_reserve_usd':str(bound-paid_upper),
        'unused_future_allowance_usd':str(original-bound),'provider_final_settlement':'NOT_ASSERTED'}


def reconcile(root):
    entries=[check(p) for p in sorted(Path(root).iterdir())]
    insist(len(entries)==10 and len({e['pod_id'] for e in entries})==10,'exact ten independently identified closed rentals required')
    total=lambda key:format(sum(Decimal(e[key]) for e in entries),'f')
    return {'schema':'ovl.closed-rental-reservation-reconciliation.v1','result':'PASS','entries':entries,
        'original_allowances_usd':total('original_allowance_usd'),'closed_charge_upper_usd':total('closed_charge_upper_usd'),
        'reported_paid_usd':total('reported_paid_usd'),'remaining_unposted_reserve_usd':total('remaining_unposted_reserve_usd'),
        'unused_future_allowance_usd':total('unused_future_allowance_usd'),
        'scope':'Reclassification of unused future rental time after both guards confirmed absence; pending bills remain bounded and reserved. No change to active deadlines, rates, cap, stop guard, reserve or verification requirements.',
        'trust':'Recorded provider/control-plane observations, original selected quotes and operator clocks; not cryptographic billing attestation or final invoice settlement.',
        'excluded':'Both original deadline probes, all no-ID/uncertain attempts and current active5090 retain their prior accounting and ceilings.'}

if __name__=='__main__':
    value=reconcile(Path(sys.argv[1]));out=Path(sys.argv[2]);insist(not out.exists(),'fresh output required');write_json(out,value);print(digest(value))
