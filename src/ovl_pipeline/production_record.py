"""Record the shared numerical trajectory, pausing for each public endorsement.

This is the recording driver, not a provider provisioner. Its operating controller
must already enforce the authorized cost guard and protected export reserve.
No production update can follow a primary boundary until fresh signature/log
verification of that exact prefix succeeds. Missing anchors preserve state and
stop at the caller's pre-budgeted checkpoint deadline.
"""
from pathlib import Path
import time

from .anchoring import bounded_bytes
from .canonical import EvidenceError,confined,digest,parse_json,read_json,write_json
from .production_observation import schedule_counts
from .initialization import process_identity
from .production_anchoring import object_at,verify_packet
from .production_chain import verify_artifacts,verify_chain
from .production_trajectory import resume_record,walk
from .progress_anchoring import ProgressPublisherPolicy,verify_prefix
from .schema import integer
from .state import capture,read_state,save_state,state_root
from .supervision import Journal
from .training import signed


def await_anchor(registration,root,envelopes,anchor_directory,policy_file,checkpoint_deadline,*,wall=time.time,sleep=time.sleep,stop_file=None):
    integer(checkpoint_deadline,1,2**53-1,'checkpoint stop deadline')
    # A numeric deadline alone is not a provider guard. The enclosing operating
    # controller supplies this earlier time from its authenticated bounded plan.
    monotonic_limit=time.monotonic()+max(0,checkpoint_deadline-wall())
    while wall()<checkpoint_deadline and time.monotonic()<monotonic_limit:
        if stop_file is not None and stop_file.exists():raise EvidenceError('external controller requested stop; preserved checkpoint')
        if policy_file.exists():
            values=parse_json(bounded_bytes(policy_file,16*1024*1024),canonical_required=True)
            if type(values) is not list:raise EvidenceError('separate external progress policies required')
            if len(values)>len(envelopes):raise EvidenceError('progress policy prefix ahead of recorded trajectory')
            current=confined(anchor_directory,f'progress-{len(envelopes)-1:05d}')
            if len(values)==len(envelopes) and current.is_dir() and (current/'statement.sigstore.json').is_file():
                return verify_prefix(registration,root,envelopes,anchor_directory,
                                     [ProgressPublisherPolicy(**v) for v in values],complete=False)
        sleep(min(5,max(0,checkpoint_deadline-wall())))
    raise EvidenceError('checkpoint deadline reached while waiting for verified public anchor; preserved state')


def save_or_adopt(path,event,output):
    """Adopt only exact recomputed state; preserve incomplete uncommitted bytes."""
    if path.exists():
        marker=confined(path,'checkpoint.json')
        if marker.is_file():
            if ({p.name for p in path.iterdir()}!={'checkpoint.json','state.json','state.safetensors'} or
                    any(p.is_symlink() or not p.is_file() for p in path.iterdir())):
                raise EvidenceError('unexpected uncommitted checkpoint files')
            checkpoint=read_json(marker);read_state(path,checkpoint)
            if checkpoint['state_root']!=state_root(*capture(event.model,event.optimizer,event.control)):
                raise EvidenceError('uncommitted checkpoint differs from recomputed trajectory')
            return checkpoint
        import os,uuid
        recovery=output.parent/(output.name+'-preserved-partials')
        recovery.mkdir(exist_ok=True)
        if recovery.is_symlink():raise EvidenceError('partial preservation directory must not be a symlink')
        destination=recovery/(path.name+'-'+uuid.uuid4().hex)
        os.rename(path,destination)  # Same filesystem; never delete the sole copy.
        for directory in (recovery,output,output.parent):
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:os.fsync(fd)
            finally:os.close(fd)
        write_json(recovery/(destination.name+'.json'),{'original_name':path.name,'preserved_name':destination.name,
                   'scope':'uncommitted incomplete checkpoint, not verification evidence'})
    return save_state(path,event.model,event.optimizer,event.control)


def record(packet,registration_bundle,production_policy,source_policy,source_checkout,stream_directories,
           output,signing_key,anchor_directory,policy_file,checkpoint_deadline,*,resume=False):
    integer(checkpoint_deadline,1,2**53-1,'checkpoint stop deadline')
    if type(resume) is not bool:raise EvidenceError('explicit resume mode required')
    if output.exists() and not resume:raise EvidenceError('fresh record output required; recovery must be explicitly verified')
    if resume and (not output.is_dir() or output.is_symlink()):raise EvidenceError('recording recovery requires existing regular directory')
    if time.time()>=checkpoint_deadline:raise EvidenceError('operating checkpoint deadline already expired')
    endorsement=verify_packet(packet,registration_bundle,production_policy,source_policy,source_checkout=source_checkout)
    r=object_at(packet,'registration.json');root=digest(r)
    if bytes(signing_key.verify_key).hex()!=r['run_public_key']:raise EvidenceError('wrong registered run signing key')
    if type(stream_directories) is not dict or set(stream_directories)!={'wikipedia','conversation'}:
        raise EvidenceError('both full training streams required')
    for phase,path in stream_directories.items():
        if schedule_counts(path,r['recipe'])!=r['coverage'][phase]:raise EvidenceError('full record input census differs')
    if not resume:output.mkdir(parents=True,exist_ok=False)
    envelopes=[];recoveries=[];anchor_receipts=[]
    session={'schema':'ovl.production-record-session.v1','registration_sha256':root,
             'registration_endorsement':endorsement,'process_observation':process_identity(),
             'checkpoint_stop_epoch':checkpoint_deadline,'provider_admission':'enclosing-controller-responsibility'}
    with Journal(output/'journal').lease() as journal:
        if resume:
            prior=read_json(confined(output,'session.json'))
            if prior['registration_sha256']!=root:raise EvidenceError('recording session registration differs')
            if (output/'record.json').exists():raise EvidenceError('completed recording cannot be resumed')
            log=read_json(confined(output,'chain.json'))
            from .schema import fields
            fields(log,'schema complete boundaries','recording recovery chain')
            if log['schema']!='ovl.production-chain.v1' or type(log['complete']) is not bool:
                raise EvidenceError('invalid recovery chain')
            envelopes=log['boundaries']
            verify_artifacts(r,root,envelopes,output,stream_directories,complete=log['complete'])
            receipt=await_anchor(r,root,envelopes,anchor_directory,policy_file,checkpoint_deadline,stop_file=output/'request-stop')
            anchor_receipts=[receipt]
            last=envelopes[-1]['body']
            # Preserve prior mutable navigation before regenerating its unfinished
            # suffix. Full checkpoints and append-only journal remain untouched.
            for name in ('recoveries.json','anchor-receipts.json'):
                path=output/name
                if path.exists():
                    old=read_json(path);history=output/'history';history.mkdir(exist_ok=True)
                    archived=history/(digest(old)+'.json')
                    if not archived.exists():write_json(archived,old)
                    if name=='recoveries.json':
                        if old['registration_sha256']!=root:raise EvidenceError('recovery navigation registration differs')
                        recoveries=[v for v in old['checkpoints'] if v['control']['global_step']<=last['control']['global_step']]
            write_json(output/('resume-'+digest(session)+'.json'),session)
            trajectory=resume_record(r,stream_directories,confined(output,last['checkpoint_path']),last)
        else:
            write_json(output/'session.json',session)
            trajectory=walk(r,stream_directories)
        for event in trajectory:
            control=event.control
            if event.kind=='recovery':
                name=f"recovery-{control['global_step']:09d}"
                checkpoint=save_or_adopt(confined(output,name),event,output)
                item={'path':name,'control':control,'checkpoint':checkpoint,
                      'last_primary_boundary_sha256':digest(envelopes[-1]['body'])}
                recoveries.append(item);journal.append('checkpoint',item)
                write_json(output/'recoveries.json',{'registration_sha256':root,'checkpoints':recoveries})
            else:
                index=len(envelopes);name=f'boundary-{index:05d}'
                checkpoint=save_or_adopt(confined(output,name),event,output)
                body={'schema':'ovl.production-boundary.v1','index':index,'registration':root,
                      'previous':digest(envelopes[-1]['body']) if envelopes else root,
                      'kind':event.kind,'control':control,'checkpoint_path':name,'checkpoint':checkpoint}
                envelopes.append(signed(body,signing_key))
                verify_chain(r,root,envelopes,complete=event.kind=='final')
                write_json(output/'chain.json',{'schema':'ovl.production-chain.v1',
                           'complete':event.kind=='final','boundaries':envelopes})
                journal.append('checkpoint',{'boundary_sha256':digest(envelopes[-1]),'index':index})
                write_json(output/'awaiting-anchor.json',{'schema':'ovl.awaiting-public-progress.v1',
                           'registration_sha256':root,'index':index,'boundary_sha256':digest(envelopes[-1]),
                           'checkpoint_path':name,'checkpoint':checkpoint})
                receipt=await_anchor(r,root,envelopes,anchor_directory,policy_file,checkpoint_deadline,stop_file=output/'request-stop')
                anchor_receipts.append(receipt)
                write_json(output/'anchor-receipts.json',{'registration_sha256':root,'verified_prefixes':anchor_receipts})
                journal.append('decision',{'action':'ADVANCE_AFTER_VERIFIED_PUBLIC_ANCHOR',
                                          'index':index,'receipt_sha256':digest(receipt)})
            # The iterator has cleared gradients and saved state at this boundary.
            # A controller may ask to stop earlier; only the lifetime watchdog can
            # enforce teardown when the numerical process itself is stalled.
            if time.time()>=checkpoint_deadline or (output/'request-stop').exists():
                write_json(output/'stopped.json',{'schema':'ovl.record-graceful-stop.v1','registration_sha256':root,
                           'control':control,'state_preserved':True,'production_complete':False})
                return {'result':'STOPPED','complete':False,'control':control}
        verification=verify_chain(r,root,envelopes,complete=True)
        report={'schema':'ovl.production-record.v1','result':'RECORDED_NOT_REPLAYED','registration_sha256':root,
                'chain_sha256':digest(envelopes),'boundaries':len(envelopes),'recovery_checkpoints':len(recoveries),
                'public_prefixes_verified_this_process':len(anchor_receipts),'chain_check':verification,
                'recording_resumed':resume,
                'training_replay':'NOT_RUN','independent_third_party':False}
        write_json(output/'record.json',report)
        return report


def main():
    import argparse
    from .anchoring import PublisherPolicy
    from .canonical import canonical
    from .production_identity import ProductionPublisherPolicy
    from .run_key import load
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','registration-bundle','production-policy','source-policy','source-checkout',
                 'wikipedia-stream','conversation-stream','output','key-directory','anchor-directory','progress-policies'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--registration-sha256',required=True)
    p.add_argument('--checkpoint-deadline',required=True,type=int)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:
        r=object_at(a.packet,'registration.json')
        if digest(r)!=a.registration_sha256:raise EvidenceError('registration differs from external selection')
        key=load(a.key_directory,run_id=r['run_id'],expected_public_key=r['run_public_key'])
        result=record(a.packet,a.registration_bundle,ProductionPublisherPolicy(**read_json(a.production_policy)),
                      PublisherPolicy(**read_json(a.source_policy)),a.source_checkout,
                      {'wikipedia':a.wikipedia_stream,'conversation':a.conversation_stream},a.output,key,
                      a.anchor_directory,a.progress_policies,a.checkpoint_deadline,resume=a.resume)
        print(canonical(result).decode());return 0
    except Exception as error:
        # Key loader failures are deliberately fixed messages; secret bytes are
        # never formatted, serialized or supplied in command-line arguments.
        print(canonical({'result':'FAIL','reason':str(error)}).decode());return 1


if __name__=='__main__':raise SystemExit(main())
