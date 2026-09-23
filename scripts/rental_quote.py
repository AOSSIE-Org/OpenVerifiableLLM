"""Bind an authored rental ceiling to retained quote bytes and owner authorization.

The catalog is an operator-captured provider observation, not a signed price offer.
Live actual rates, runtime device selection, capacity and final billing remain
separate checks. Storage uses the documented higher stopped-volume rate and a
short28-day denominator, then a25% all-in rate margin.
"""
from decimal import Decimal,ROUND_CEILING
import hashlib
import json
from pathlib import Path
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,require_digest
from ovl_pipeline.schema import fields,integer

AUTHORIZATION_SHA256='12bbf07875add059c338ae35255fbb6c2bfbda16c102310395312a7474f93e80'
AUTHORIZATION=Path.home()/'.local/share/openverifiablellm/authorizations/cost-guard-revision-20260918.json'
STORAGE_URL='https://docs.runpod.io/pods/pricing'


def rate(value):
    from ovl_pipeline.budget import money
    return Decimal(money(value))/10**6


def validate_quote(quote,payload,plan,*,network_volume=None):
    retained=quote.get('schema')=='ovl.rental-quote.v3'
    fields(quote,'schema observed_epoch catalog_response catalog_response_sha256 selected_gpu storage_source storage_page_sha256 storage_observed_epoch container_gb_month_usd volume_gb_month_upper_usd monthly_hours rate_margin_percent'+(' network_volume_sha256' if retained else ''),'rental price evidence')
    if quote['schema'] not in ('ovl.rental-quote.v1','ovl.rental-quote.v2','ovl.rental-quote.v3') or quote['storage_source']!=STORAGE_URL:raise EvidenceError('unsupported rental quote source')
    if retained:
        from retained_volume import validate
        validate(network_volume,plan,payload)
        if payload['cloudType']!='SECURE' or quote['network_volume_sha256']!=digest(network_volume):raise EvidenceError('retained volume quote identity differs')
    elif network_volume is not None or 'networkVolumeId' in payload:raise EvidenceError('retained storage requires separately reserved quote')
    require_digest(quote['catalog_response_sha256']);require_digest(quote['storage_page_sha256'])
    g=quote['selected_gpu']
    if quote['schema']=='ovl.rental-quote.v1':
        fields(g,'id secure secure_hourly_usd','selected secure quote')
        if g['secure'] is not True:raise EvidenceError('historical quote requires secure cloud')
        cloud='secure';selected_rate=g['secure_hourly_usd']
    else:
        fields(g,'id cloud hourly_usd','selected cloud quote')
        if g['cloud'] not in ('SECURE','COMMUNITY'):raise EvidenceError('explicit supported quote cloud required')
        cloud=g['cloud'].lower();selected_rate=g['hourly_usd']
    if payload['cloudType']!=cloud.upper():raise EvidenceError('quote cloud differs from creation selection')
    raw=quote['catalog_response']
    if type(raw) is not str or len(raw.encode())>256*1024 or hashlib.sha256(raw.encode()).hexdigest()!=quote['catalog_response_sha256']:
        raise EvidenceError('retained catalog response differs from selected root')
    from provider_preflight import unique_pairs
    catalog=json.loads(raw,object_pairs_hook=unique_pairs,parse_float=Decimal)
    matches=[x for x in catalog['gpus'] if x['id']==g['id']]
    if (len(matches)!=1 or matches[0].get(cloud) is not True
        or type(matches[0].get('maxCount',{}).get(cloud)) is not int or matches[0]['maxCount'][cloud]<1
        or Decimal(matches[0]['price'][cloud])!=rate(selected_rate)):
        raise EvidenceError('selected GPU/rate differs from actual retained catalog bytes')
    if g['id']!=payload['gpuTypeId'] or rate(selected_rate)<=0:
        raise EvidenceError('selected GPU differs from retained cloud quote')
    for k,age in [('observed_epoch',600),('storage_observed_epoch',86400)]:
        integer(quote[k],1,2**53-1,k)
        if not 0<=plan['input']['now_epoch']-quote[k]<=age:raise EvidenceError('stale quote evidence')
    if (quote['container_gb_month_usd']!='0.10' or quote['volume_gb_month_upper_usd']!='0.20'
        or type(quote['monthly_hours']) is not int or quote['monthly_hours']!=672
        or type(quote['rate_margin_percent']) is not int or quote['rate_margin_percent']!=125):
        raise EvidenceError('storage price bound or rate margin differs from selected policy')
    computed=(rate(selected_rate)+(Decimal(payload['containerDiskInGb'])*rate(quote['container_gb_month_usd'])+
                  Decimal(0 if retained else payload['volumeInGb'])*rate(quote['volume_gb_month_upper_usd']))/quote['monthly_hours'])*Decimal(quote['rate_margin_percent'])/100
    upper=format(computed.quantize(Decimal('0.000001'),rounding=ROUND_CEILING),'f')
    if plan['input']['hourly_upper_usd']!=upper or plan['input']['quote_sha256']!=digest(quote):
        raise EvidenceError('rental rate/quote root differs from complete compute and storage arithmetic')
    if (not AUTHORIZATION.is_file() or AUTHORIZATION.is_symlink()
            or plan['input']['authorization_sha256']!=AUTHORIZATION_SHA256
            or file_hash(AUTHORIZATION)!=AUTHORIZATION_SHA256):
        raise EvidenceError('rental authorization differs from retained owner instruction')
    return upper
