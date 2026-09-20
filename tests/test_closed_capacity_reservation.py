"""Explicit provider doubles. No actual closure or money-release credit."""
import copy,importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
import pytest
from test_reconcile_capacity_rejection import fixture
import reconcile_capacity_rejection as old
from ovl_pipeline.canonical import EvidenceError,digest
spec=importlib.util.spec_from_file_location('closed_capacity',Path(__file__).parents[1]/'project/evidence/closed-capacity-reservation-v1/reconcile.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def args():
    original=fixture();receipt=old.verify(*original);w=original[0]['watchdog_intent'];closed=w['plan']['external_terminate_epoch']+200
    report={'schema':'ovl.external-watchdog-result.v1','complete':True,'pod_id':None,'intent_sha256':digest(w),
       'confirmed_absent_epoch':closed,'automatic_provider_termination':'UNVERIFIED','external_termination_requested':True,
       'residual_network_volumes':[],'provider_billing_reconciliation':'PENDING','execution_admission':'NOT_RUN'}
    events=[{'kind':'creation-intent','body':w},
            {'kind':'decision','body':{'action':'TERMINATE','pod_id':None}},
            {'kind':'provider-observation','body':{'pods':[],'volume_ids':[],'account_hourly_usd':'0','observed_epoch':closed-15}},
            {'kind':'teardown','body':report}]
    observations=copy.deepcopy(original[6]);now=closed+20
    for o,t in zip(observations,[closed,now]):
        o['observed_epoch']=t;o['http_clock']={k:t for k in o['http_clock']}
    return [original,receipt,copy.deepcopy(original[5]),events,report,observations,{'controller':'inactive','watchdog':'inactive'},now]


def test_exact_rejection_and_original_closed_guard_can_release_unused_creation_allowance():
    result=m.verify(*args());assert result['result']=='PASS'
    assert result['remaining_creation_reservation_usd']=='0'
    assert result['provider_final_settlement']=='NOT_ASSERTED' and result['training_verification_credit'] is False


@pytest.mark.parametrize('damage',['unverified-receipt','private-response','prefix','observed-creator','observed-watchdog','terminal','identity','deadline','resource','volumes','automatic-proof','missing-termination','running-creator','running-watchdog','new-account','pods','rate','autopay','unseparated','stale','preclosure-read','clock','extra-field','missing-action','missing-absence'])
def test_missing_or_changed_guard_identity_evidence_keeps_entire_reservation(damage):
    a=args();original,receipt,controller,events,result,obs,services,now=a
    if damage=='unverified-receipt':receipt['result']='unverified'
    elif damage=='private-response':original[4]+=b' '
    elif damage=='prefix':controller[0]['body']={}
    elif damage=='observed-creator':controller.append({'kind':'creation-observed','body':{'id':'unexpected'}})
    elif damage=='observed-watchdog':events.insert(1,{'kind':'creation-observed','body':{'id':'unexpected'}})
    elif damage=='terminal':events.pop()
    elif damage=='identity':result['intent_sha256']='f'*64
    elif damage=='deadline':result['confirmed_absent_epoch']=original[0]['watchdog_intent']['plan']['external_terminate_epoch']
    elif damage=='resource':result['pod_id']='late-pod'
    elif damage=='volumes':result['residual_network_volumes']=['disk']
    elif damage=='automatic-proof':result['automatic_provider_termination']='VERIFIED'
    elif damage=='missing-termination':result['external_termination_requested']=False
    elif damage=='running-creator':services['controller']='active'
    elif damage=='running-watchdog':services['watchdog']='active'
    elif damage=='new-account':obs[1]['account_identity_sha256']='f'*64
    elif damage=='pods':obs[1]['pods']=[{'id':'another'}]
    elif damage=='rate':obs[1]['account_hourly_usd']='1'
    elif damage=='autopay':obs[1]['autopay']=True
    elif damage=='unseparated':obs[0]=copy.deepcopy(obs[1])
    elif damage=='stale':a[-1]+=31
    elif damage=='preclosure-read':obs[0]['http_clock']['request_started_epoch']-=1
    elif damage=='clock':obs[0]['http_clock']['server_epoch']+=100
    elif damage=='extra-field':result['invented']=True
    elif damage=='missing-action':events.pop(1)
    elif damage=='missing-absence':events.pop(2)
    with pytest.raises(EvidenceError):m.verify(*a)
