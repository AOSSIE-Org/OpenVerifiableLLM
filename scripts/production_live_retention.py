"""Retain an immutable complete production checkpoint while its job runs.

The caller selects registration and recording/replay parents independently.
These checks preserve actual state bytes; they never establish numerical replay
or authorize a public boundary acknowledgement. Final whole-tree retention is
still mandatory, including unfinished files and failed attempts.
"""
from pathlib import Path

from ovl_pipeline import schema
from ovl_pipeline.canonical import EvidenceError,canonical,digest,read_json,require_digest,sha256,verify_inventory
from ovl_pipeline.production_chain import checkpoint_manifest,verify_chain,schedule
from ovl_pipeline.state import read_state,unpack
from pod_job_client import save_once
from pod_versioned_export import export,regular_directory


def record_selection(registration,root,chain):
    schema.fields(chain,'schema complete boundaries','live recording chain')
    if chain['schema']!='ovl.production-chain.v1' or type(chain['complete']) is not bool:
        raise EvidenceError('unsupported recording chain')
    verify_chain(registration,root,chain['boundaries'],complete=chain['complete'])
    body=chain['boundaries'][-1]['body']
    return {'schema':'ovl.live-checkpoint-selection.v1','registration_sha256':root,'kind':'record-primary',
            'path':body['checkpoint_path'],'checkpoint':body['checkpoint'],'control':body['control'],
            'parent_sha256':digest(chain['boundaries'][-1])}


def replay_selection(registration,root,envelopes,session,progress):
    verify_chain(registration,root,envelopes,complete=True)
    schema.fields(session,'schema registration_sha256 chain_sha256 code_root process_observation environment scope prover_state_restored resume_supported','live replay session')
    if (session['schema']!='ovl.production-replay-session.v1' or session['registration_sha256']!=root
        or session['chain_sha256']!=digest(envelopes) or session['code_root']!=registration['code_root']
        or session['scope']!='fresh-start-continuous-all-updates-numerical-replay'
        or session['prover_state_restored'] is not False or session['resume_supported'] is not False):
        raise EvidenceError('replay session differs from selected parents or supported scope')
    if digest(session['environment']['compatible'])!=registration['runtime']['compatible_environment_sha256']:
        raise EvidenceError('replay session compatible environment differs')
    schema.fields(progress,'session_sha256 complete comparisons','live replay progress')
    if progress['session_sha256']!=digest(session) or type(progress['complete']) is not bool:
        raise EvidenceError('replay progress session differs')
    comparisons=progress['comparisons']
    if type(comparisons) is not list or not 1<=len(comparisons)<=len(envelopes):
        raise EvidenceError('bounded nonempty replay comparison prefix required')
    if progress['complete'] and len(comparisons)!=len(envelopes):raise EvidenceError('incomplete replay claims completion')
    for i,(item,envelope) in enumerate(zip(comparisons,envelopes)):
        schema.fields(item,'index boundary_sha256 state_root control verifier_checkpoint result','live replay comparison')
        schema.integer(item['index'],0,len(envelopes)-1,'replay comparison index')
        checkpoint_manifest(item['verifier_checkpoint'])
        body=envelope['body']
        if (item['index']!=i or item['boundary_sha256']!=digest(envelope) or item['result']!='PASS'
            or item['control']!=body['control'] or item['state_root']!=body['checkpoint']['state_root']
            or item['verifier_checkpoint']['state_root']!=item['state_root']):
            raise EvidenceError('replay comparison prefix differs from selected chain')
    last=comparisons[-1]
    return {'schema':'ovl.live-checkpoint-selection.v1','registration_sha256':root,'kind':'replay-primary',
            'path':f'verifier-boundary-{last["index"]:05d}','checkpoint':last['verifier_checkpoint'],
            'control':last['control'],'parent_sha256':digest(session)}


def recovery_control(registration,step):
    """Registered recovery step/phase only; exact target cursor is not claimed."""
    wiki=registration['coverage']['wikipedia']['updates'];chat=registration['coverage']['conversation']['updates']
    schema.integer(step,1,wiki+chat,'recovery global step')
    phase='wikipedia' if step<=wiki else 'conversation';local=step if phase=='wikipedia' else step-wiki
    if (local==registration['coverage'][phase]['updates'] or local%registration['recovery_every']
        or local%registration['recipe']['boundary_every']==0):
        raise EvidenceError('recovery step is not a registered nonprimary checkpoint')
    return {'phase':phase,'phase_step':local,'global_step':step}


def record_recovery_selection(registration,root,chain,recoveries):
    primary=record_selection(registration,root,chain)
    schema.fields(recoveries,'registration_sha256 checkpoints','live recording recoveries')
    if recoveries['registration_sha256']!=root:raise EvidenceError('recovery registration differs')
    items=recoveries['checkpoints'];expected=schedule(registration)
    if type(items) is not list or not 1<=len(items)<=65536:raise EvidenceError('bounded nonempty recovery list required')
    parents={digest(e['body']):e['body'] for e in chain['boundaries']};previous=0
    for item in items:
        schema.fields(item,'path control checkpoint last_primary_boundary_sha256','live recording recovery')
        c=item['control'];schema.control(c);checkpoint_manifest(item['checkpoint'])
        selected=recovery_control(registration,c['global_step'])
        parent=parents.get(item['last_primary_boundary_sha256'])
        if (parent is None or any(c[k]!=v for k,v in selected.items()) or c['global_step']<=previous
            or item['path']!=f'recovery-{c["global_step"]:09d}' or parent['control']['phase']!=c['phase']
            or not parent['control']['phase_step']<c['phase_step']<expected[parent['index']+1]['phase_step']
            or not parent['control']['cursor']<c['cursor']<registration['coverage'][c['phase']]['targets']):
            raise EvidenceError('record recovery ancestry, schedule or control differs')
        previous=c['global_step']
    last=items[-1]
    if last['control']['global_step']<=primary['control']['global_step']:return primary
    return {'schema':'ovl.live-checkpoint-selection.v1','registration_sha256':root,'kind':'record-recovery',
            'path':last['path'],'checkpoint':last['checkpoint'],'control':last['control'],
            'parent_sha256':last['last_primary_boundary_sha256']}


def replay_recovery_selection(registration,root,envelopes,session,progress,recoveries,checkpoint):
    primary=replay_selection(registration,root,envelopes,session,progress)
    schema.fields(recoveries,'session_sha256 recoveries','live verifier recoveries')
    if recoveries['session_sha256']!=digest(session):raise EvidenceError('verifier recovery session differs')
    items=recoveries['recoveries']
    if type(items) is not list or not 1<=len(items)<=65536:raise EvidenceError('bounded nonempty verifier recoveries required')
    previous=0
    for item in items:
        schema.fields(item,'global_step state_root','live verifier recovery')
        recovery_control(registration,item['global_step']);require_digest(item['state_root'])
        if item['global_step']<=previous:raise EvidenceError('verifier recovery steps regressed')
        previous=item['global_step']
    last=items[-1]
    if last['global_step']<=primary['control']['global_step']:return primary
    checkpoint_manifest(checkpoint)
    if checkpoint['state_root']!=last['state_root']:raise EvidenceError('verifier recovery marker differs')
    return {'schema':'ovl.live-checkpoint-selection.v1','registration_sha256':root,'kind':'replay-recovery',
            'path':f'verifier-recovery-{last["global_step"]:09d}','checkpoint':checkpoint,
            'control':recovery_control(registration,last['global_step']),'parent_sha256':digest(session)}


def retain(transport,selection,expected_selection,job,health,health_file,store,output,deadline,maximum_bytes):
    """Once-selected snapshot, with adoption after complete transfer interruption.

    The selection digest must come from the caller's authenticated parent checks.
    It remains fixed across retries. Export credit requires all actual state bytes.
    """
    require_digest(expected_selection)
    schema.fields(selection,'schema registration_sha256 kind path checkpoint control parent_sha256','live checkpoint selection')
    if digest(selection)!=expected_selection or selection['schema']!='ovl.live-checkpoint-selection.v1':
        raise EvidenceError('live checkpoint selection changed')
    require_digest(selection['registration_sha256']);require_digest(selection['parent_sha256'])
    import re
    prefix={'record-primary':'boundary','replay-primary':'verifier-boundary',
            'record-recovery':'recovery','replay-recovery':'verifier-recovery'}.get(selection['kind'])
    digits=9 if selection['kind'].endswith('recovery') else 5
    if prefix is None or not re.fullmatch(prefix+r'-[0-9]{'+str(digits)+'}',selection['path']):
        raise EvidenceError('unsupported live checkpoint kind/path')
    checkpoint_manifest(selection['checkpoint'])
    if selection['kind']=='replay-recovery':
        schema.fields(selection['control'],'phase phase_step global_step','selected verifier recovery control')
        if selection['control']!=recovery_control(health.registration,selection['control']['global_step']):
            raise EvidenceError('verifier recovery selection changed schedule')
    else:schema.control(selection['control'])
    contract=health.contract(job)
    wanted='production-record' if selection['kind'].startswith('record-') else 'full-replay'
    if (contract['kind']!=wanted or contract['registration_sha256']!=selection['registration_sha256']
        or not any(x['root']==transport.profile['remote_root'] and x['profile_sha256']==digest(transport.profile)
                   for x in contract['outputs'])):
        raise EvidenceError('live checkpoint does not belong to selected production output')
    schema.integer(deadline,1,health.plan['external_terminate_epoch'],'fixed live export deadline')
    schema.integer(maximum_bytes,1,2**40,'selected live checkpoint bound')
    marker=canonical(selection['checkpoint'])
    files=[{'path':'checkpoint.json','bytes':len(marker),'sha256':sha256(marker)},*selection['checkpoint']['files']]
    if sum(f['bytes'] for f in files)>maximum_bytes:raise EvidenceError('live checkpoint exceeds selected export budget')
    output=regular_directory(output)
    save_once(output/'selection.json',{'job_sha256':job,'selection':selection,'profile_sha256':digest(transport.profile),
              'deadline_epoch':deadline,'maximum_bytes':maximum_bytes})
    # Two bounded transfer attempts, preserving every failed copy and the same
    # original deadline. No retry if a completed receipt is malformed.
    target=output/'snapshot-000'
    if target.exists() and not(target/'export.json').exists():target=output/'snapshot-001'
    if target.exists():
        if not(target/'export.json').exists():raise EvidenceError('live checkpoint transfer attempts exhausted')
        receipt=read_json(target/'export.json')
    else:
        def progress(operation,counts,total):
            health.bytes(operation,counts,total=total);health.write(health_file)
        receipt=export(transport,selection['path'],store,target,deadline,progress=progress,expected_files=files)
    if (receipt['schema']!='ovl.offpod-versioned-tree-export.v1' or receipt['result']!='PASS'
        or receipt['pod_id']!=health.pod or receipt['profile_sha256']!=digest(transport.profile)
        or receipt['root']!=selection['path'] or receipt['files']!=files or receipt['numerical_verification']!='NOT_RUN'):
        raise EvidenceError('retained live checkpoint receipt differs')
    directory=Path(receipt['files_directory'])
    if (directory.absolute()!=(target/'files').absolute()
        or any(p.is_symlink() for p in [directory,*directory.absolute().parents])):
        raise EvidenceError('retained live files escaped the selected snapshot')
    verify_inventory(directory,files)
    if read_json(directory/'checkpoint.json')!=selection['checkpoint']:raise EvidenceError('retained live marker differs')
    metadata,tensors=read_state(directory,selection['checkpoint'])
    control=unpack(metadata['tree'],tensors)['control'];schema.control(control)
    if selection['kind']=='replay-recovery':
        if (any(control[k]!=v for k,v in selection['control'].items())
            or not 0<control['cursor']<health.registration['coverage'][control['phase']]['targets']):
            raise EvidenceError('retained verifier recovery control differs')
    elif control!=selection['control']:raise EvidenceError('retained live control differs')
    result={'schema':'ovl.live-checkpoint-retention.v1','result':'PASS','selection_sha256':expected_selection,
            'receipt_sha256':digest(receipt),'receipt_path':str((target/'export.json').resolve()),
            'scope':'complete selected safe checkpoint copied and checked off pod; final whole-output retention still required',
            'numerical_replay':'NOT_RUN','public_anchor':'NOT_RUN','workload_complete':False}
    save_once(output/'retention.json',result)
    health.exported_files(job,directory,files);health.write(health_file)
    return result
