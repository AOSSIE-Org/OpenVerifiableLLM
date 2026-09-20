"""Read-only recovery after an uncertain atomic policy handoff.

The independently selected policies remain external to the acknowledgement.
A peer's policy file alone never proves public signatures or numerical progress.
No remote writes, training starts, receipt fabrication or deadline renewal.
"""
from dataclasses import asdict
from pathlib import Path

from ovl_pipeline.canonical import EvidenceError,confined,digest,inventory,read_json,write_json
from ovl_pipeline.progress_anchoring import verify_prefix
from ovl_pipeline.production_chain import verify_chain
from pod_checkpoint_handoff import observe,state_check
from pod_job_client import tree
from pod_versioned_export import regular_directory


def reconcile(transport,registration,root,snapshot_directory,ack,policies,output,deadline):
    chain,body,waiting,_=state_check(registration,root,Path(snapshot_directory))
    index=waiting['index'];envelopes=chain['boundaries']
    if (ack.get('schema')!='ovl.verified-public-progress-ack.v1' or ack.get('registration_sha256')!=root
        or ack.get('index')!=index or ack.get('boundary_sha256')!=waiting['boundary_sha256']
        or len(policies)!=len(envelopes) or ack.get('policy')!=asdict(policies[-1])):
        raise EvidenceError('uncertain handoff differs from selected boundary/policy')
    anchors=Path(ack['anchor_directory'])
    verify_prefix(registration,root,envelopes,anchors,policies,complete=False)
    statement=read_json(confined(anchors,f'progress-{index:05d}/statement.json'))
    if ack.get('checkpoint_archive')!=statement['archive']:raise EvidenceError('uncertain handoff archive differs')
    output=regular_directory(output,fresh=True)
    peer=observe(transport,'chain.json',output/'chain.json',16*1024**2,deadline)
    from ovl_pipeline.schema import fields
    fields(peer,'schema complete boundaries','observed record chain')
    if peer['schema']!='ovl.production-chain.v1' or type(peer['complete']) is not bool:
        raise EvidenceError('unsupported observed chain')
    verify_chain(registration,root,peer['boundaries'],complete=peer['complete'])
    if peer['boundaries'][:len(envelopes)]!=envelopes or not len(envelopes)<=len(peer['boundaries'])<=len(envelopes)+1:
        raise EvidenceError('peer chain does not extend exactly the selected acknowledged prefix')
    values=[asdict(p) for p in policies]
    observed=observe(transport,'external-progress-policies.json',output/'policies.json',16*1024**2,deadline,optional=True)
    if observed!=values:
        if observed==values[:-1] or index==0 and observed is None:
            if peer['boundaries']!=envelopes:raise EvidenceError('peer advanced without selected public policy')
            result={'schema':'ovl.uncertain-progress-handoff.v1','result':'NOT_DELIVERED',
                    'registration_sha256':root,'boundary_sha256':waiting['boundary_sha256'],
                    'scope':'prior policy remains; caller may only use separately guarded current-boundary handoff',
                    'remote_writes':False,'training_replay':'NOT_RUN'}
            write_json(output/'reconciliation.json',result);return result
        raise EvidenceError('peer policy prefix was replaced or advanced by another controller')
    expected=inventory(anchors,[f'progress-{i:05d}/{name}' for i in range(index+1)
                                for name in ('statement.json','statement.sigstore.json')])
    if tree(transport,'anchors',deadline)!=expected:raise EvidenceError('peer anchor inventory differs from selected public prefix')
    downloaded=output/'anchors';downloaded.mkdir()
    transfers=[]
    for item in expected:
        target=confined(downloaded,item['path']);target.parent.mkdir(parents=True,exist_ok=True)
        transfers.append(transport.get('anchors/'+item['path'],target,{**item,'path':'anchors/'+item['path']},deadline))
    checked=verify_prefix(registration,root,envelopes,downloaded,policies,complete=False)
    # A read-only check must not bless a moving or concurrently replaced prefix.
    after=observe(transport,'external-progress-policies.json',output/'policies-after.json',16*1024**2,deadline)
    if after!=values:raise EvidenceError('peer policies changed during handoff reconciliation')
    result={'schema':'ovl.uncertain-progress-handoff.v1','result':'DELIVERED',
            'registration_sha256':root,'boundary_sha256':waiting['boundary_sha256'],
            'policies_sha256':digest(values),'peer_chain_sha256':digest(peer),
            'public_prefix_check':checked,'transfers':transfers,'remote_writes':False,
            'scope':'exact selected public policy present; complete actual peer anchor bytes downloaded and verified; recorder consumption not asserted',
            'training_replay':'NOT_RUN'}
    write_json(output/'reconciliation.json',result);return result
