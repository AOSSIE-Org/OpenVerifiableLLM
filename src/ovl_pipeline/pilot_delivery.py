"""Bounded pilot checkpoint delivery; acknowledgements are operator receipts.

This protocol gates development updates on actual off-pod retention. It confers
no public anchoring, training truth, or independent-verification credit.
"""
from pathlib import Path
import time

from .canonical import EvidenceError, canonical, confined, digest, read_json, require_digest, write_json
from .schema import fields, integer, control as validate_control


def policy(value):
    fields(value,'schema session mode phase deadline_epoch copy_timeout_seconds maximum_checkpoint_bytes','pilot delivery policy')
    if (value['schema']!='ovl.pilot-delivery-policy.v1' or value['mode'] not in ('record','replay')
        or value['phase'] not in ('wikipedia','conversation')):raise EvidenceError('invalid pilot delivery policy')
    require_digest(value['session']);integer(value['deadline_epoch'],1,2**53-1,'pilot deadline')
    integer(value['copy_timeout_seconds'],30,420,'checkpoint copy timeout')
    integer(value['maximum_checkpoint_bytes'],1,2**40,'checkpoint byte bound')
    return value


def origin(binding,record_sha256=None):
    if record_sha256 is not None:require_digest(record_sha256)
    return digest({'binding':binding,'record_sha256':record_sha256})


def validate_request(value,selected,expected_origin,index,previous):
    fields(value,'schema policy origin_sha256 index previous path checkpoint control copy_deadline_epoch','pilot delivery request')
    if (value['schema']!='ovl.pilot-delivery-request.v1' or value['policy']!=selected
        or value['origin_sha256']!=expected_origin or value['index']!=index or value['previous']!=previous):
        raise EvidenceError('pilot delivery session, origin or parent differs')
    integer(value['index'],0,4095,'delivery sequence')
    prefix='boundary' if selected['mode']=='record' else 'verifier-boundary'
    if value['path']!=f'{prefix}-{index:05d}':raise EvidenceError('pilot delivery path differs')
    integer(value['copy_deadline_epoch'],1,selected['deadline_epoch'],'fixed checkpoint deadline')
    checkpoint=value['checkpoint'];fields(checkpoint,'schema state_root files','delivered checkpoint')
    if checkpoint['schema']!='ovl.checkpoint.v1':raise EvidenceError('unknown delivered checkpoint')
    require_digest(checkpoint['state_root'])
    if type(checkpoint['files']) is not list or len(checkpoint['files'])!=2:raise EvidenceError('complete checkpoint inventory required')
    for f in checkpoint['files']:
        fields(f,'path bytes sha256','delivered safe file');require_digest(f['sha256'])
        integer(f['bytes'],1,selected['maximum_checkpoint_bytes'],'checkpoint file bytes')
    if [f['path'] for f in checkpoint['files']]!=['state.json','state.safetensors']:
        raise EvidenceError('closed safe checkpoint names required')
    if sum(f['bytes'] for f in checkpoint['files'])+len(canonical(checkpoint))>selected['maximum_checkpoint_bytes']:
        raise EvidenceError('checkpoint exceeds bound before transfer')
    c=dict(value['control']);cycle=c.pop('pilot_cycle',None);validate_control(c);integer(cycle,0,1_000_000,'pilot cycle')
    if c['phase']!=selected['phase'] or (index==0 and (c['global_step'] or c['phase_step'] or c['cursor'] or cycle)):
        raise EvidenceError('pilot delivery control differs')
    if index and (c['global_step']<index or c['phase_step']!=c['global_step']):raise EvidenceError('pilot delivery step differs')
    return value


def acknowledgement(request,receipt_sha256):
    require_digest(receipt_sha256)
    return {'schema':'ovl.pilot-delivery-ack.v1','session':request['policy']['session'],
            'request_sha256':digest(request),'index':request['index'],
            'state_root':request['checkpoint']['state_root'],'receipt_sha256':receipt_sha256,
            'scope':'operator-verified-offpod-safe-state-only'}


def check_ack(value,request):
    fields(value,'schema session request_sha256 index state_root receipt_sha256 scope','pilot delivery acknowledgement')
    integer(value['index'],0,4095,'acknowledgement index')
    if value!=acknowledgement(request,value['receipt_sha256']):raise EvidenceError('wrong pilot delivery acknowledgement')
    return value


class Delivery:
    def __init__(self,output,selected,expected_origin,*,wall=time.time,sleep=time.sleep):
        self.output=Path(output);self.policy=policy(selected);require_digest(expected_origin);self.origin=expected_origin
        self.wall=wall;self.sleep=sleep;self.index=0;self.previous=digest(selected)
        self.directory=self.output/'delivery';self.directory.mkdir(mode=0o700,exist_ok=False)

    def checkpoint(self,path,checkpoint,control):
        request={'schema':'ovl.pilot-delivery-request.v1','policy':self.policy,'origin_sha256':self.origin,
                 'index':self.index,'previous':self.previous,'path':path,'checkpoint':checkpoint,'control':dict(control),
                 'copy_deadline_epoch':min(self.policy['deadline_epoch'],int(self.wall())+self.policy['copy_timeout_seconds'])}
        validate_request(request,self.policy,self.origin,self.index,self.previous)
        write_json(self.directory/f'request-{self.index:05d}.json',request)
        write_json(self.directory/'request.json',request)
        ack=self.directory/f'ack-{self.index:05d}.json'
        while True:
            if self.wall()>=request['copy_deadline_epoch']:raise EvidenceError('fixed pilot delivery deadline expired')
            if ack.exists():
                check_ack(read_json(confined(self.directory,ack.name)),request);break
            self.sleep(1)
        self.previous=digest(request);self.index+=1


def settings_delivery(settings):
    version=settings.get('schema')
    if version=='ovl.gpu-pilot-settings.v1':
        if 'delivery' in settings:raise EvidenceError('legacy pilot cannot hide delivery policy')
        return None
    if version!='ovl.gpu-pilot-settings.v2':raise EvidenceError('unsupported pilot settings')
    selected=policy(settings['delivery'])
    if selected['mode']!='record' or selected['phase']!=settings['stream']['phase']:
        raise EvidenceError('record delivery policy differs')
    return selected


def binding(settings):
    return {'schema':'ovl.pilot-record-parent-binding.v1',
            **{k+'_sha256':digest(settings[k]) for k in ('recipe','kernel','stream')},'code_root':settings['code_root']}


def verify_tree(directory,selected,expected_origin,boundaries):
    """Check every bound request and acknowledgement against actual selected states.

    An ACK is an operator assertion; the caller separately reads safe-state bytes.
    """
    policy(selected);directory=Path(directory);previous=digest(selected);paths=set()
    for index,b in enumerate(boundaries):
        name=f'delivery/request-{index:05d}.json';request=read_json(confined(directory,name));paths.add(name)
        validate_request(request,selected,expected_origin,index,previous)
        if request['checkpoint']!=b['checkpoint'] or request['control']!=b['control']:
            raise EvidenceError('retained delivery differs from numerical boundary')
        name=f'delivery/ack-{index:05d}.json';check_ack(read_json(confined(directory,name)),request);paths.add(name)
        previous=digest(request)
    if not boundaries:raise EvidenceError('empty delivery chain')
    if read_json(confined(directory,'delivery/request.json'))!=request:raise EvidenceError('last delivery request differs')
    paths.add('delivery/request.json')
    actual={p.relative_to(directory).as_posix() for p in (directory/'delivery').rglob('*') if p.is_file()}
    if actual!=paths:raise EvidenceError('unselected delivery files')
    return paths
