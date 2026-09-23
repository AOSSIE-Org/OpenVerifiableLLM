"""Bind one audited production descriptor and retained result to its registration.

All complete replay/state checks still run in the numerical driver and final
release verifier. These orchestration checks grant no independent verification.
"""
from pathlib import Path
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json
from ovl_pipeline.production_anchoring import PACKET_FILES
from ovl_pipeline.production_chain import verify_chain
from ovl_pipeline.state import read_state,unpack
from production_retention import verify


def command(job,r,packet,bundle,production_policy,source_policy,numerical,kind):
    argv=job['argv'];module='ovl_pipeline.'+('production_record' if kind=='production-record' else 'production_replay')
    if (argv.count('--module')!=1 or argv[argv.index('--module')+1]!=module or argv.count('--')!=1
        or job['kind']!=kind):raise EvidenceError('exact audited production module required')
    args=argv[argv.index('--')+1:]
    if len(args)%2 or len(set(args[::2]))!=len(args)//2:raise EvidenceError('unambiguous complete production arguments required')
    selected=dict(zip(args[::2],args[1::2]))
    flags={'--packet','--registration-bundle','--production-policy','--source-policy','--source-checkout',
           '--wikipedia-stream','--conversation-stream','--output','--progress-policies'}
    flags|=({'--registration-sha256','--checkpoint-deadline','--key-directory','--anchor-directory'} if kind=='production-record'
            else {'--chain-directory','--progress-directory'})
    if set(selected)!=flags or selected['--output']!=numerical.profile['remote_root'] or selected['--output'] not in job['export_roots']:
        raise EvidenceError('production output or complete argument selection differs')
    if kind=='production-record':
        if selected['--registration-sha256']!=digest(r) or not 0<int(selected['--checkpoint-deadline'])<=job['deadline_epoch']:
            raise EvidenceError('registered production identity/deadline differs')
        secret=selected['--key-directory']
        if any(secret==p or secret.startswith(p+'/') or p.startswith(secret+'/') for p in job['export_roots']):
            raise EvidenceError('private run key overlaps a public retained root')
        if selected['--anchor-directory']!=selected['--output']+'/anchors' or selected['--progress-policies']!=selected['--output']+'/external-progress-policies.json':
            raise EvidenceError('record public handoff paths differ from selected numerical output')
    elif (selected['--progress-directory']!=selected['--chain-directory']+'/anchors'
          or selected['--progress-policies']!=selected['--chain-directory']+'/external-progress-policies.json'):
        raise EvidenceError('replay public handoff paths differ from selected recorded input')
    expected={selected['--packet']+'/'+name:file_hash(packet/name) for name in PACKET_FILES}
    expected.update({selected['--registration-bundle']:file_hash(bundle),selected['--production-policy']:digest(production_policy),
                     selected['--source-policy']:digest(source_policy)})
    expected.update({selected['--'+phase+'-stream']+'/stream.json':r['coverage'][phase]['stream_sha256'] for phase in ('wikipedia','conversation')})
    inputs={f['path']:f for f in job['required_files']}
    if len(inputs)!=len(job['required_files']) or any(p not in inputs or inputs[p]['sha256']!=root for p,root in expected.items()):
        raise EvidenceError('production required bytes omit or alter selected public parents')
    return selected


def result(registration,stage,control,transports,job_file,job_root,worker_root,remote_root,kind):
    proof=read_json(Path(stage['retention_path']))
    if digest(proof)!=stage['retention_sha256']:raise EvidenceError('stage retention digest differs')
    receipts=verify(proof,control,transports,job_file,job_root,worker_root)
    roots=[receipt for ref,receipt in zip(proof['roots'],receipts) if ref['declared_root']==remote_root]
    if len(roots)!=1:raise EvidenceError('numerical result not fully retained')
    directory=Path(roots[0]['files_directory']);r=registration;root=digest(r)
    if kind=='production-record':
        report=read_json(directory/'record.json');chain=read_json(directory/'chain.json')
        if (report.get('schema')!='ovl.production-record.v1' or report.get('result')!='RECORDED_NOT_REPLAYED'
            or report['registration_sha256']!=root or chain.get('complete') is not True
            or report['chain_sha256']!=digest(chain['boundaries'])):
            raise EvidenceError('successful process did not finish complete recording')
        verify_chain(r,root,chain['boundaries'],complete=True)
    else:
        report=read_json(directory/'verification.json');session=read_json(directory/'session.json')
        if (report.get('schema')!='ovl.production-numerical-replay.v1' or report.get('result')!='PASS'
            or report['registration_sha256']!=root or report['initial_state_regenerated'] is not True
            or report['prover_checkpoints_restored'] is not False or report['session_sha256']!=digest(session)
            or session['prover_state_restored'] is not False or session['resume_supported'] is not False
            or any(report['updates_recomputed'][p]!=c['updates'] or report['targets_recomputed'][p]!=c['targets'] for p,c in r['coverage'].items())):
            raise EvidenceError('successful process did not finish full fresh replay')
    return {'directory':str(directory),'files':roots[0]['files'],'report':report}


def downloaded_registration(publication,saved):
    """Recheck previously completed actual-download bytes on controller recovery."""
    selected={}
    for name,archive in [('packet-publication',saved['packet']),('anchor-publication',saved['registration_anchor'])]:
        base=publication/name;plan=read_json(base/'plan.json')
        if (plan['repo']!=archive['repo'] or plan['prefix']!=archive['prefix'] or plan['files']!=archive['inventory']):
            raise EvidenceError('public registration archive differs from original publication plan')
        matches=[]
        for output in sorted(base.glob('download-*')):
            if not(output/'verification.json').exists():continue
            proof=read_json(output/'verification.json');intent=read_json(output/'intent.json')
            if (proof.get('schema')!='ovl.evidence-download.v1' or proof.get('result')!='PASS'
                or proof.get('repo')!=archive['repo'] or proof.get('revision')!=archive['revision']
                or proof.get('files')!=archive['inventory']):continue
            if (proof['intent_sha256']!=digest(intent) or proof['plan_sha256']!=digest(plan)
                or intent['plan_sha256']!=digest(plan) or intent['force_download'] is not True
                or intent['authentication']!='anonymous-token-False' or intent['revision']!=archive['revision']):
                raise EvidenceError('retained public download selection differs')
            from ovl_pipeline.canonical import verify_inventory
            verify_inventory(output/'downloaded',archive['inventory']);matches.append(output/'downloaded')
        if not matches:raise EvidenceError('public registration lacks completed actual-download bytes')
        selected[name]=matches[0]
    return selected


def complete_replay(r,stage,control,transports,job_file,job_root,worker_root,remote_output,chain,envelopes,numerical):
    from assemble_computation_evidence import replay_states,process
    states=replay_states(r,envelopes,chain,Path(numerical['directory']),numerical['report'])
    proof=read_json(Path(stage['retention_path']));job=read_json(job_file)
    receipts=verify(proof,control,transports,job_file,job_root,worker_root)
    args=job['argv'][:job['argv'].index('--')]
    if args.count('--output')!=1:raise EvidenceError('unique audited process output required')
    audit=args[args.index('--output')+1]
    matches=[(ref,receipt) for ref,receipt in zip(proof['roots'],receipts) if audit.startswith(ref['declared_root']+'/')]
    if len(matches)!=1:raise EvidenceError('audited replay process was not fully retained')
    ref,receipt=matches[0]
    process(r,Path(receipt['files_directory'])/audit.removeprefix(ref['declared_root']+'/'),job,stage['terminal'],remote_output)
    return {'safe_states_checked':states,'scope':'retained complete operator replay states and selected process; no independent verification'}
