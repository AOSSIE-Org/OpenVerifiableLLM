"""Full sequential numerical replay from regenerated production initialization.

Every registered update and boundary is recomputed. Prover checkpoints are only
read for comparison and never restored. This checks numerical trajectory, not raw
transformation reconstruction, download provenance, or independent observation.
"""
import argparse
from pathlib import Path
import time

from . import gpu,initialization
from .production_trajectory import walk
from .anchoring import PublisherPolicy
from .canonical import EvidenceError,canonical,confined,digest,read_json,write_json
from .production_anchoring import object_at,verify_packet
from .production_chain import verify_artifacts
from .production_identity import ProductionPublisherPolicy
from .progress_anchoring import ProgressPublisherPolicy,verify_prefix
from .state import capture,read_state,save_state,state_root,tensor_digest
from .training import code_root


def authenticate(packet,registration_bundle,production_policy,source_policy,source_checkout,
                 chain_directory,progress_directory,progress_policies):
    endorsement=verify_packet(packet,registration_bundle,production_policy,source_policy,source_checkout=source_checkout)
    r=object_at(packet,'registration.json');root=digest(r)
    log=read_json(confined(chain_directory,'chain.json'))
    from .schema import fields
    fields(log,'schema complete boundaries','production chain')
    if log['schema']!='ovl.production-chain.v1' or log['complete'] is not True:
        raise EvidenceError('complete production chain required')
    anchors=verify_prefix(r,root,log['boundaries'],progress_directory,progress_policies,complete=True)
    return r,log['boundaries'],{'registration':endorsement,'progress':anchors}


def replay(packet,registration_bundle,production_policy,source_policy,source_checkout,
           chain_directory,progress_directory,progress_policies,stream_directories,output):
    if output.exists():raise EvidenceError('replay output must be fresh; preserve earlier attempts')
    started=time.monotonic_ns()
    r,envelopes,endorsements=authenticate(packet,registration_bundle,production_policy,source_policy,source_checkout,
                                         chain_directory,progress_directory,progress_policies)
    root=digest(r)
    if r['code_root']!=code_root():raise EvidenceError('executing replay source differs from registered code')
    artifacts=verify_artifacts(r,root,envelopes,chain_directory,stream_directories,complete=True)
    trajectory=walk(r,stream_directories);event=next(trajectory)
    if event.kind!='initial':raise EvidenceError('trajectory omitted regenerated initialization')
    model,opt,control,environment=event.model,event.optimizer,event.control,event.environment
    position=0;comparisons=[];base_root=None;recoveries=[]
    output.mkdir(parents=True,exist_ok=False)
    session={'schema':'ovl.production-replay-session.v1','registration_sha256':root,
             'chain_sha256':digest(envelopes),'code_root':code_root(),'process_observation':initialization.process_identity(),
             'environment':environment,'scope':'fresh-start-continuous-all-updates-numerical-replay',
             'prover_state_restored':False,'resume_supported':False}
    write_json(output/'session.json',session)
    def compare(kind):
        nonlocal position,base_root
        if position>=len(envelopes):raise EvidenceError('extra numerical boundary')
        b=envelopes[position]['body']
        if b['kind']!=kind or b['control']!=control:raise EvidenceError('recomputed boundary schedule/control differs')
        metadata,tensors=read_state(confined(chain_directory,b['checkpoint_path']),b['checkpoint'])
        actual=state_root(*capture(model,opt,control))
        if actual!=state_root(metadata,tensors):raise EvidenceError(f'continuous replay state mismatch at boundary {position}')
        # Preserve verifier-generated state separately, only AFTER this comparison.
        # This release deliberately has no resume reader and never treats it as a
        # substitute for the fresh trajectory. Later continuation must authenticate
        # the verifier's own completed prefix, never select a prover checkpoint.
        own=save_state(output/f'verifier-boundary-{position:05d}',model,opt,control)
        if own['state_root']!=actual:raise EvidenceError('verifier checkpoint changed during capture')
        record={'index':position,'boundary_sha256':digest(envelopes[position]),'state_root':actual,
                'control':control.copy(),'verifier_checkpoint':own,'result':'PASS'}
        comparisons.append(record)
        write_json(output/'progress.json',{'session_sha256':digest(session),'complete':False,'comparisons':comparisons})
        if kind=='base':base_root=tensor_digest(dict(model.state_dict()))
        position+=1
    compare('initial')
    gpu.torch.cuda.synchronize();setup_ms=(time.monotonic_ns()-started+999999)//1000000
    numerical_started=time.monotonic_ns();updates={};targets={}
    for event in trajectory:
        model,opt,control=event.model,event.optimizer,event.control
        if event.kind=='recovery':
            own=save_state(output/f"verifier-recovery-{control['global_step']:09d}",model,opt,control)
            recoveries.append({'global_step':control['global_step'],'state_root':own['state_root']})
            write_json(output/'recoveries.json',{'session_sha256':digest(session),'recoveries':recoveries})
        else:compare(event.kind)
        if event.kind in ('base','final'):
            phase=control['phase'];updates[phase]=control['phase_step'];targets[phase]=control['cursor']
    if position!=len(envelopes) or base_root is None:raise EvidenceError('missing final comparisons/base state')
    gpu.torch.cuda.synchronize()
    report={'schema':'ovl.production-numerical-replay.v1','result':'PASS',
            'scope':'fresh-regenerated-initialization-continuous-two-phase-all-update-state-comparison',
            'registration_sha256':root,'session_sha256':digest(session),'chain_sha256':digest(envelopes),
            'endorsements':endorsements,'artifact_check':artifacts,'comparisons':comparisons,'recovery_checkpoints':recoveries,
            'updates_recomputed':updates,'targets_recomputed':targets,'initial_state_regenerated':True,
            'prover_checkpoints_restored':False,'base_model_root':base_root,
            'chat_model_root':tensor_digest(dict(model.state_dict())),
            'setup_ms':setup_ms,'numerical_replay_ms':(time.monotonic_ns()-numerical_started+999999)//1000000,
            'performed_by':'local-execution-operator','independent_third_party':False,
            'raw_transformation_reconstruction':'NOT_RUN','public_download_verification':'NOT_RUN',
            'cost_guard_admission':'NOT_RUN','end_to_end_release_verification':'NOT_RUN'}
    write_json(output/'verification.json',report)
    write_json(output/'progress.json',{'session_sha256':digest(session),'complete':True,'comparisons':comparisons})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','registration-bundle','production-policy','source-policy','source-checkout','chain-directory',
                 'progress-directory','progress-policies','wikipedia-stream','conversation-stream','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:
        policies=read_json(a.progress_policies)
        if type(policies) is not list:raise EvidenceError('external progress policy list required')
        result=replay(a.packet,a.registration_bundle,ProductionPublisherPolicy(**read_json(a.production_policy)),
                      PublisherPolicy(**read_json(a.source_policy)),a.source_checkout,a.chain_directory,a.progress_directory,
                      [ProgressPublisherPolicy(**v) for v in policies],
                      {'wikipedia':a.wikipedia_stream,'conversation':a.conversation_stream},a.output)
    except Exception as e:print(canonical({'result':'FAIL','reason':str(e)}).decode());return 1
    print(canonical({'result':result['result'],'report_sha256':digest(result),'output':str(a.output)}).decode());return 0

if __name__=='__main__':raise SystemExit(main())
