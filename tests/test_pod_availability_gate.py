import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from pod_availability_gate import check,check_cloud
from ovl_pipeline.canonical import EvidenceError


def sample():
    return {'observed_epoch':1000,'request':{'id':'test-gpu','include':['AVAILABILITY'],
      'product':['POD'],'cloud':'SECURE','count':1,'cudaVersions':['13.0','13.2']},
      'response':{'id':'test-gpu','secure':True,'availability':'LOW','price':{'secure':.74},
        'cudaVersions':[{'version':'13.0','available':False},{'version':'13.2','available':True}]}}


def checked(o,now=1001):
    return check(o,now=now,gpu_id='test-gpu',cuda_versions=['13.0','13.2'],maximum_hourly_usd='.74')


def test_only_actual_available_versions_returned():
    o=sample();before=copy.deepcopy(o)
    assert checked(o)['available_cuda_versions']==['13.2']
    assert o==before


@pytest.mark.parametrize('field,value',[('product',['SERVERLESS']),('product',['POD','SERVERLESS']),
    ('cloud','COMMUNITY'),('count',2),('count',True),('cudaVersions',['13.0']),('id','another')])
def test_wrong_product_or_shape_refused(field,value):
    o=sample();o['request'][field]=value
    with pytest.raises(EvidenceError):checked(o)


@pytest.mark.parametrize('field,value',[('availability','NONE'),('availability','unknown'),
    ('id','another'),('secure',False),('price',{'secure':'NaN'}),('price',{'secure':.75}),
    ('cudaVersions',[{'version':'13.0','available':True}]),
    ('cudaVersions',[{'version':'13.0','available':False},{'version':'13.2','available':False}]),
    ('cudaVersions',[{'version':'13.0','available':True},{'version':'13.0','available':True}])])
def test_unavailable_malformed_or_over_budget_refused(field,value):
    o=sample();o['response'][field]=value
    with pytest.raises(EvidenceError):checked(o)


@pytest.mark.parametrize('now',[999,1601,True])
def test_stale_or_invalid_observation_refused(now):
    with pytest.raises(EvidenceError):checked(sample(),now)


def community():
    o=sample();o['request']['cloud']='COMMUNITY'
    o['response'].update(community=True,price={'secure':'.99','community':'.69'})
    return o


def community_checked(o):
    return check_cloud(o,now=1001,gpu_id='test-gpu',cuda_versions=['13.0','13.2'],
                       maximum_hourly_usd='.69',cloud='COMMUNITY')


def test_community_requires_explicit_selection_and_uses_only_its_lane():
    o=community();before=copy.deepcopy(o)
    result=community_checked(o)
    assert result['cloud']=='COMMUNITY' and result['hourly_usd']=='0.69'
    assert result['available_cuda_versions']==['13.2'] and o==before
    with pytest.raises(EvidenceError):checked(o)
    with pytest.raises(EvidenceError):community_checked(sample())


@pytest.mark.parametrize('damage',['request-cloud','missing-lane','unavailable-lane','other-lane-cheaper',
                                  'nonfinite','unavailable-stock','serverless','missing-cuda'])
def test_community_cannot_borrow_secure_price_or_other_availability(damage):
    o=community()
    if damage=='request-cloud':o['request']['cloud']='SECURE'
    elif damage=='missing-lane':del o['response']['community']
    elif damage=='unavailable-lane':o['response']['community']=False
    elif damage=='other-lane-cheaper':o['response']['price']={'secure':'.01','community':'.70'}
    elif damage=='nonfinite':o['response']['price']['community']='NaN'
    elif damage=='unavailable-stock':o['response']['availability']='NONE'
    elif damage=='serverless':o['request']['product']=['SERVERLESS']
    else:o['response']['cudaVersions']=[]
    with pytest.raises(EvidenceError):community_checked(o)


@pytest.mark.parametrize('cloud',[None,'SPOT','community',False])
def test_unsupported_cloud_fails_before_catalog_selection(cloud):
    with pytest.raises(EvidenceError):check_cloud(community(),now=1001,gpu_id='test-gpu',
        cuda_versions=['13.0','13.2'],maximum_hourly_usd='.69',cloud=cloud)
