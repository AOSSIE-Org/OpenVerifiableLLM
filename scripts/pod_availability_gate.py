"""Check a retained POD-scoped catalog read before a future creation intent.

Availability is an observation, never a reservation or evidence of noncreation.
This gate does not contact RunPod, release fences, or alter active rentals.
"""
from decimal import Decimal,InvalidOperation
from ovl_pipeline.canonical import EvidenceError


def check(observation,*,now,gpu_id,cuda_versions,maximum_hourly_usd):
    expected={'id':gpu_id,'include':['AVAILABILITY'],'product':['POD'],
              'cloud':'SECURE','count':1,'cudaVersions':cuda_versions}
    if observation.get('request')!=expected or type(observation['request'].get('count')) is not int:
        raise EvidenceError('exact single secure POD and CUDA-scoped request required')
    epoch=observation.get('observed_epoch')
    if type(epoch) is not int or type(now) is not int or not 0<=now-epoch<=600:
        raise EvidenceError('stale or invalid POD availability observation')
    gpu=observation.get('response')
    if type(gpu) is not dict or gpu.get('id')!=gpu_id or gpu.get('secure') is not True:
        raise EvidenceError('POD GPU identity mismatch')
    if gpu.get('availability') not in ('LOW','MEDIUM','HIGH'):
        raise EvidenceError('selected secure POD GPU unavailable')
    versions=gpu.get('cudaVersions')
    if type(versions) is not list or not versions or len(cuda_versions)!=len(set(cuda_versions)):
        raise EvidenceError('invalid CUDA availability inventory')
    seen=set();available=[]
    for v in versions:
        if type(v) is not dict or v.get('version') not in cuda_versions or v['version'] in seen or type(v.get('available')) is not bool:
            raise EvidenceError('mismatched CUDA availability entry')
        seen.add(v['version'])
        if v['available']:available.append(v['version'])
    if seen!=set(cuda_versions) or not available:
        raise EvidenceError('selected CUDA versions unavailable or missing')
    try:
        rate=Decimal(str(gpu['price']['secure']));cap=Decimal(maximum_hourly_usd)
        if not rate.is_finite() or not cap.is_finite() or not 0<rate<=cap:raise ValueError()
    except (KeyError,TypeError,ValueError,InvalidOperation):
        raise EvidenceError('invalid or over-budget secure POD quote') from None
    return {'gpu_id':gpu_id,'secure_hourly_usd':format(rate,'f'),
            'available_cuda_versions':available,'observed_epoch':epoch,
            'scope':'POD-scoped availability observation; not a reservation or creation authorization'}
