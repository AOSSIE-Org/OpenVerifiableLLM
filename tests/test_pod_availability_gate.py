import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from pod_availability_gate import check
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
