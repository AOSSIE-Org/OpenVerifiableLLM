#!/usr/bin/env python3
"""Authenticated checkpoint export and ordered acknowledgement over bounded SSH.

Never creates compute. Registration identity is supplied by an independently
verified packet; remote telemetry and saved PASS fields cannot establish it.
All snapshots are fresh and retained, including failures. An exported primary
proves artifact preservation only, never execution or sequential replay.
"""
from dataclasses import asdict
from pathlib import Path
import os
import time

from ovl_pipeline.canonical import EvidenceError,canonical,confined,digest,inventory,read_json,sha256,write_json
from ovl_pipeline.production_anchoring import object_at,verify_packet
from ovl_pipeline.production_chain import verify_chain
from ovl_pipeline.progress_anchoring import verify_prefix
from ovl_pipeline.schema import fields
from ovl_pipeline.state import read_state,unpack


def observe(transport,name,destination,maximum,deadline,*,optional=False):
    """Fresh peer bytes, not an authenticated inventory or progress heartbeat."""
    data=transport.read_live(name,maximum,deadline)
    if data is None:
        if optional:return None
        raise EvidenceError('required remote observation is absent')
    fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    return read_json(destination)


def selected(registration,root,directory):
    chain=read_json(confined(directory,'chain.json'))
    fields(chain,'schema complete boundaries','handoff chain')
    if chain['schema']!='ovl.production-chain.v1' or type(chain['complete']) is not bool:
        raise EvidenceError('unsupported handoff chain')
    envelopes=chain['boundaries']
    checked=verify_chain(registration,root,envelopes,complete=chain['complete'])
    body=envelopes[-1]['body'];index=len(envelopes)-1
    waiting={'schema':'ovl.awaiting-public-progress.v1','registration_sha256':root,'index':index,
             'boundary_sha256':digest(envelopes[-1]),'checkpoint_path':body['checkpoint_path'],'checkpoint':body['checkpoint']}
    if read_json(confined(directory,'awaiting-anchor.json'))!=waiting:
        raise EvidenceError('remote waiting marker differs from authenticated chain')
    return chain,body,waiting,checked


def state_check(registration,root,directory):
    chain,body,waiting,checked=selected(registration,root,directory)
    checkpoint=confined(directory,body['checkpoint_path'])
    if ({p.name for p in checkpoint.iterdir()}!={'checkpoint.json','state.json','state.safetensors'} or
        any(p.is_symlink() or not p.is_file() for p in checkpoint.iterdir())):
        raise EvidenceError('closed safe checkpoint files required')
    if read_json(confined(checkpoint,'checkpoint.json'))!=body['checkpoint']:
        raise EvidenceError('transferred checkpoint marker differs')
    metadata,tensors=read_state(checkpoint,body['checkpoint'])
    if unpack(metadata['tree'],tensors)['control']!=body['control']:
        raise EvidenceError('transferred checkpoint control differs')
    return chain,body,waiting,checked


def snapshot(transport,registration,root,output,deadline,*,progress=None):
    """Caller must authenticate registration first; copies a whole paused primary."""
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    observe(transport,'chain.json',output/'chain.json',16*1024**2,deadline)
    observe(transport,'awaiting-anchor.json',output/'awaiting-anchor.json',1024**2,deadline)
    chain,body,waiting,checked=selected(registration,root,output)
    checkpoint=output/body['checkpoint_path'];checkpoint.mkdir(mode=0o700)
    marker=canonical(body['checkpoint'])
    files=[{'path':'checkpoint.json','bytes':len(marker),'sha256':sha256(marker)},*body['checkpoint']['files']]
    transfers=[]
    from pod_transfer import staged_get
    for item in files:
        name=body['checkpoint_path']+'/'+item['path']
        transfers.append(staged_get(transport,name,checkpoint/item['path'],{**item,'path':name},deadline,
                                    output/'transfers'/digest(item),progress=progress))
    state_check(registration,root,output)
    after=output/'after';after.mkdir(mode=0o700)
    for name,expected,maximum in [('chain.json',chain,16*1024**2),('awaiting-anchor.json',waiting,1024**2)]:
        if observe(transport,name,after/name,maximum,deadline)!=expected:
            raise EvidenceError('remote checkpoint snapshot changed during transfer; copies preserved')
    if time.time()>=deadline:raise EvidenceError('checkpoint export deadline expired')
    receipt={'schema':'ovl.offpod-checkpoint-export.v1','result':'PASS','registration_sha256':root,
             'boundary_sha256':waiting['boundary_sha256'],'index':waiting['index'],
             'checkpoint_path':body['checkpoint_path'],'checkpoint':body['checkpoint'],
             'files':inventory(checkpoint,[f['path'] for f in files]),'chain_check':checked,'transfers':transfers,
             'completed_epoch':int(time.time()),'scope':'actual complete safe state bytes preserved off pod',
             'public_boundary_anchor':'NOT_RUN','training_replay':'NOT_RUN','workload_complete':False}
    write_json(output/'export.json',receipt)
    return receipt


def snapshot_registered(transport,packet,bundle,production_policy,source_policy,source_checkout,output,deadline,*,progress=None):
    registration=authenticated(packet,bundle,production_policy,source_policy,source_checkout)
    return snapshot(transport,registration,digest(registration),output,deadline,progress=progress)


def authenticated(packet,bundle,production_policy,source_policy,source_checkout):
    checked=verify_packet(packet,bundle,production_policy,source_policy,source_checkout=source_checkout)
    registration=object_at(packet,'registration.json')
    if digest(registration)!=checked['registration_sha256'] or digest(registration)!=production_policy.statement_sha256:
        raise EvidenceError('registration changed after publisher verification')
    return registration


def deliver(transport,registration,root,snapshot_directory,ack,policies,output,deadline):
    """Reverify local state and public prefix; install both files before policies.

    Policies come from the coordinator's prior selection, never from ack. Saved
    acknowledgement result strings supply no verification credit. This verifies
    retained public anchor bytes again; fresh public downloads belong to the
    publisher receipt, not to this SSH delivery operation.
    """
    chain,body,waiting,_=state_check(registration,root,Path(snapshot_directory))
    index=waiting['index'];envelopes=chain['boundaries']
    if (ack.get('schema')!='ovl.verified-public-progress-ack.v1' or ack.get('registration_sha256')!=root
        or ack.get('index')!=index or ack.get('boundary_sha256')!=waiting['boundary_sha256']
        or len(policies)!=len(envelopes) or ack.get('policy')!=asdict(policies[-1])):
        raise EvidenceError('acknowledgement differs from independently selected boundary/policy')
    anchors=Path(ack['anchor_directory'])
    checked=verify_prefix(registration,root,envelopes,anchors,policies,complete=False)
    current=confined(anchors,f'progress-{index:05d}')
    value=read_json(current/'statement.json')
    if ack.get('checkpoint_archive')!=value['archive']:
        raise EvidenceError('acknowledged public checkpoint archive differs from verified statement')
    output=Path(output);output.mkdir(mode=0o700,parents=True,exist_ok=False)
    if observe(transport,'awaiting-anchor.json',output/'before-waiting.json',1024**2,deadline)!=waiting:
        raise EvidenceError('pod is no longer waiting at selected checkpoint')
    if observe(transport,'chain.json',output/'before-chain.json',16*1024**2,deadline)!=chain:
        raise EvidenceError('pod chain changed before acknowledgement')
    previous=observe(transport,'external-progress-policies.json',output/'before-policies.json',16*1024**2,deadline,optional=True)
    values=[asdict(p) for p in policies]
    allowed=[values[:-1],values]
    if index==0:allowed.append(None)
    if previous not in allowed:raise EvidenceError('remote policy prefix would be rolled back or replaced')
    transfers=[]
    # One bounded remote inventory avoids two SSH handshakes for every old
    # boundary on every delivery. This is a peer observation only: the recorder
    # still verifies the complete public prefix before advancing numerically.
    from pod_job_client import tree
    remote=tree(transport,'anchors',deadline,allow_missing=True)
    expected=inventory(anchors,[f'progress-{i:05d}/{name}' for i in range(index+1)
                               for name in ('statement.json','statement.sigstore.json')])
    selected={f['path']:f for f in expected}
    for f in remote:
        if selected.get(f['path'])!=f:raise EvidenceError('existing remote anchor bytes differ; preserve rather than overwrite')
    present={f['path'] for f in remote}
    for f in expected:
        if f['path'] not in present:
            transfers.append(transport.put('anchors/'+f['path'],confined(anchors,f['path']),deadline))
    if observe(transport,'awaiting-anchor.json',output/'final-waiting.json',1024**2,deadline)!=waiting:
        raise EvidenceError('pod advanced during anchor delivery')
    policy_file=output/'external-progress-policies.json';write_json(policy_file,values)
    transfers.append(transport.put('external-progress-policies.json',policy_file,deadline,replace=True))
    receipt={'schema':'ovl.progress-ack-delivery.v1','result':'PASS','registration_sha256':root,'index':index,
             'boundary_sha256':waiting['boundary_sha256'],'policies_sha256':digest(values),'prefix_check':checked,
             'transfers':transfers,'scope':'verified public anchor bytes delivered before atomic policy handoff',
             'remote_recorder_verification':'REQUIRED_BY_RECORDER','fresh_public_download_this_command':'NOT_RUN',
             'training_replay':'NOT_RUN','workload_complete':False}
    write_json(output/'delivery.json',receipt)
    return receipt


def main():
    import argparse
    from pod_transfer import Transport
    from ovl_pipeline.anchoring import PublisherPolicy
    from ovl_pipeline.production_identity import ProductionPublisherPolicy
    from ovl_pipeline.progress_anchoring import ProgressPublisherPolicy
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['snapshot','deliver'])
    for name in ('profile','key','known-hosts','packet','registration-bundle','production-policy','source-policy','source-checkout','output'):
        p.add_argument('--'+name,required=True,type=Path)
    for name in ('snapshot-directory','ack','progress-policies'):p.add_argument('--'+name,type=Path)
    p.add_argument('--deadline',required=True,type=int);a=p.parse_args()
    try:
        r=authenticated(a.packet,a.registration_bundle,ProductionPublisherPolicy(**read_json(a.production_policy)),
                        PublisherPolicy(**read_json(a.source_policy)),a.source_checkout)
        transport=Transport(read_json(a.profile),a.key,a.known_hosts)
        if a.operation=='snapshot':result=snapshot(transport,r,digest(r),a.output,a.deadline)
        else:
            if any(v is None for v in (a.snapshot_directory,a.ack,a.progress_policies)):
                raise EvidenceError('delivery requires snapshot, acknowledgement and separate policies')
            policies=[ProgressPublisherPolicy(**v) for v in read_json(a.progress_policies)]
            result=deliver(transport,r,digest(r),a.snapshot_directory,read_json(a.ack),policies,a.output,a.deadline)
        print(result['schema']+' '+digest(result))
    except Exception as error:p.exit(1,'checkpoint handoff refused: '+type(error).__name__+'; preserve partial outputs\n')


if __name__=='__main__':main()
