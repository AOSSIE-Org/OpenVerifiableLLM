"""Exact retained network-volume identity and separately reserved lifetime cost.

No creation, deletion, cache trust or scientific acceptance is granted here.
The selected volume survives pod teardown; its owner must verify safe retention
and reconcile its actual bills before releasing this reservation.
"""
from decimal import Decimal, ROUND_CEILING
import re

from ovl_pipeline.canonical import EvidenceError
from ovl_pipeline.schema import fields, integer
from ovl_pipeline.budget import money


def hourly(selection):
    return (Decimal(selection['size_gb'])*Decimal(selection['monthly_gb_usd'])/selection['monthly_hours']).quantize(Decimal('.000001'),rounding=ROUND_CEILING)


def validate(selection, plan, payload):
    fields(selection,'schema id name size_gb data_center_id created_epoch retention_deadline_epoch billing_ceiling_epoch monthly_gb_usd monthly_hours reserved_usd','retained volume')
    if selection['schema']!='ovl.retained-network-volume.v1':raise EvidenceError('unsupported retained volume')
    for key in ('id','name','data_center_id'):
        if type(selection[key]) is not str or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',selection[key]):
            raise EvidenceError('invalid retained volume identity')
    integer(selection['size_gb'],10,4096,'network volume GB')
    if selection['monthly_gb_usd']!='0.07' or type(selection['monthly_hours']) is not int or selection['monthly_hours']!=672:
        raise EvidenceError('network volume price bound differs')
    created=selection['created_epoch'];retain=selection['retention_deadline_epoch'];ceiling=selection['billing_ceiling_epoch']
    integer(created,1,plan['input']['now_epoch'],'volume creation epoch')
    integer(retain,created+1,created+7*86400,'volume retention deadline')
    integer(ceiling,retain+3600,created+8*86400,'volume billing and recovery ceiling')
    if plan['billing_ceiling_epoch']>retain:raise EvidenceError('rental extends beyond retained volume deadline')
    cost=(hourly(selection)*Decimal(ceiling-created)/3600).quantize(Decimal('.000001'),rounding=ROUND_CEILING)
    if money(selection['reserved_usd'])<int(cost*10**6) or money(plan['input']['reserved_remaining_usd'])<money(selection['reserved_usd']):
        raise EvidenceError('complete retained volume cost is not reserved')
    if (payload.get('networkVolumeId')!=selection['id'] or payload.get('dataCenterId')!=selection['data_center_id']
        or payload.get('volumeMountPath')!='/workspace' or payload['volumeInGb']!=selection['size_gb']):
        raise EvidenceError('attached volume differs from retained selection')
    return selection


def matches(intent, observation):
    selection=intent.get('retained_volume')
    if selection is None:return observation['volume_ids']==[]
    expected={'id':selection['id'],'name':selection['name'],'size':selection['size_gb'],'dataCenterId':selection['data_center_id']}
    return observation['volume_ids']==[selection['id']] and observation.get('network_volumes')==[expected]


def compute_hourly(intent, observation):
    """Subtract at most the separately fully reserved volume rate from account rate."""
    total=Decimal(observation['account_hourly_usd'])
    if not total.is_finite() or total<0:raise EvidenceError('invalid account rate')
    selected=intent.get('retained_volume')
    return max(Decimal(0),total-(hourly(selected) if selected is not None else Decimal(0)))


def baseline_valid(intent, observation):
    return (not observation['pods'] and matches(intent,observation)
            and observation['autopay'] is False and compute_hourly(intent,observation)==0)


def account_errors(intent, observation, pod):
    """Check retained storage even before discovery and after compute disappears."""
    errors=[]
    if not matches(intent,observation):errors.append('retained-volume-identity')
    if observation['autopay'] is not False:errors.append('autopay')
    if len(observation['pods'])!=(0 if pod is None else 1):errors.append('unrelated-pods')
    ceiling=Decimal(0) if pod is None else Decimal(intent['plan']['input']['hourly_upper_usd'])
    if compute_hourly(intent,observation)>ceiling:errors.append('account-rate')
    return errors
