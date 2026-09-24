#!/usr/bin/env python3
"""Run already-admitted pinned inputs on exactly one already-guarded pod.

Provider creation remains in the existing separately supervised controller. This
entrypoint adopts original selections/fences after interruption. It finishes
paid computation before CPU release assembly and makes no final-release claim.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import time
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,inventory,read_json,verify_inventory
from ovl_pipeline.schema import fields
from ovl_pipeline.anchoring import PublisherPolicy
from ovl_pipeline.run_key import load as load_key
from pod_transfer import Transport
from pod_job_client import save_once
from production_run_coordinator import Run,restore_phase_bindings
from production_run_inputs import registration,production_job,qualified_runtime,validate_run_key,qualified_volume


def transport(profile,key,known):return Transport(profile,key,known)


def registration_templates(spec):
    """Read only explicitly inventoried, hash-pinned candidate templates."""
    root=Path(spec['input_inventory']['directory']).absolute()
    locations=spec['registration_template'] if spec['schema']=='ovl.production-lifetime-invocation.v2' else {'baseline':spec['registration_template']}
    entries={e['path']:e for e in spec['input_inventory']['files']}
    selected={}
    for variant,location in locations.items():
        try:name=Path(location).absolute().relative_to(root).as_posix()
        except ValueError as exc:raise EvidenceError('registration template outside pinned inventory') from exc
        if name not in entries:raise EvidenceError('registration template missing from pinned inventory')
        verify_inventory(root,[entries[name]])
        value=read_json(root/name)
        if digest(value)!=entries[name]['sha256']:raise EvidenceError('registration template changed while reading')
        selected[variant]=value
    return selected


def selected_phases(spec,bindings,downloads):
    result={};previous=[]
    optimized=spec.get('schema')=='ovl.production-lifetime-invocation.v2'
    names=('qualification','optimization','initialization-baseline','initialization-candidate') if optimized else ('qualification','initialization')
    for name in names:
        item=spec['initialization'][name.removeprefix('initialization-')] if name.startswith('initialization-') else spec[name]
        fields(item,'plan inputs output','lifetime phase locations')
        plan=read_json(Path(item['plan']));expected=spec['selection']['phases'][name]
        if digest(plan)!=expected or plan['prior_jobs']!=[]:raise EvidenceError('selected original phase template differs')
        resolved={**plan,'prior_jobs':sorted(previous)}
        out=Path(item['output'])
        if (out/'selected-plan.json').exists() and read_json(out/'selected-plan.json')!=resolved:
            raise EvidenceError('original resolved phase parent selection differs')
        restore_phase_bindings(resolved,digest(resolved),out,bindings,downloads)
        if (out/'final/result.json').exists():previous+=sorted(v['job_sha256'] for v in read_json(out/'final/result.json')['stages'])
        result[name]=(plan,expected,Path(item['inputs']),out)
    return result


def inputs(run,spec,a):
    """Upload exact public parents plus the private run key outside exports."""
    base=run.control.profile['remote_root'];root=run.output/'input-delivery';root.mkdir(exist_ok=True)
    files=[]
    from ovl_pipeline.production_anchoring import PACKET_FILES
    for name in sorted(PACKET_FILES):files.append(('production-inputs/packet/'+name,a['packet']/name))
    for name,value in [('production-policy.json',asdict(a['production_policy'])),('source-policy.json',asdict(a['source_policy']))]:
        path=run.output/name;save_once(path,value);files.append(('production-inputs/'+name,path))
    files.append(('production-inputs/registration.sigstore.json',a['bundle']))
    key=Path(spec['run_key']);load_key(key,run_id=a['registration']['run_id'],expected_public_key=a['registration']['run_public_key'])
    files.extend(('private/run-key/'+name,key/name) for name in ('public.json','seed.key'))
    for name,path in files:
        # Secret bytes and private-key hashes never enter the public input or
        # output inventories. Local transfer metadata stays in the private key
        # directory; only its public identity is used for cost-liveness binding.
        private=name.startswith('private/')
        output=(key/'operator-transfer-receipts'/run.control.profile['pod_id'] if private else root)
        output.mkdir(mode=0o700,parents=True,exist_ok=True)
        identity={'profile_sha256':digest(run.control.profile),'remote_path':name,
                  'bytes':path.stat().st_size,'sha256':file_hash(path)}
        receipt=output/(digest(identity)+'.json');window=output/(digest(identity)+'-intent.json')
        if receipt.exists():
            if read_json(receipt)['selection']!=identity:raise EvidenceError('original upload parent identity changed')
            continue
        if not window.exists():
            now=run.health.now();save_once(window,{'selection':identity,'started_epoch':now,
                       'deadline_epoch':min(run.plan['request_checkpoint_epoch'],now+180)})
        original=read_json(window)
        if original['selection']!=identity or original['deadline_epoch']!=min(run.plan['request_checkpoint_epoch'],original['started_epoch']+180):
            raise EvidenceError('original input delivery deadline changed')
        operation=digest({'registration':digest(a['registration']),'destination':name})
        def progress(counts):run.health.bytes(operation,counts,total=identity['bytes']);run.health.write(run.health_file)
        sent=run.control.put(name,path,original['deadline_epoch'],progress=progress)
        save_once(receipt,{'selection':identity,'receipt':sent})


def run(spec,expected):
    optimized=spec.get('schema')=='ovl.production-lifetime-invocation.v2'
    fields(spec,'schema selection rental controller watchdog profile key known_hosts worker output health qualification initialization registration_template source_statement source_bundle source_policy preparation source_checkout run_key static_files input_inventory'+(' optimization' if optimized else ''),'production lifetime invocation')
    if spec['schema'] not in ('ovl.production-lifetime-invocation.v1','ovl.production-lifetime-invocation.v2') or digest(spec)!=expected:raise EvidenceError('selected lifetime invocation differs')
    if optimized:
        fields(spec['initialization'],'baseline candidate','selected initialization variants')
        fields(spec['registration_template'],'baseline candidate','selected registration variants')
        if spec['selection']['schema']!='ovl.production-run-selection.v2':raise EvidenceError('optimization invocation requires bounded selection')
    elif spec['selection']['schema']!='ovl.production-run-selection.v1':raise EvidenceError('legacy invocation cannot omit optimization')
    # Every path in this caller-owned input inventory is explicit and confined;
    # no filesystem orientation, credential discovery or private-project search.
    fields(spec['input_inventory'],'directory files','pinned lifetime inputs')
    verify_inventory(Path(spec['input_inventory']['directory']),spec['input_inventory']['files'])
    templates=registration_templates(spec)
    run_identity=validate_run_key(Path(spec['run_key']),list(templates.values()))
    key=Path(spec['key']);known=Path(spec['known_hosts']);profile=read_json(Path(spec['profile']))
    control=transport(profile,key,known);output=Path(spec['output']);bindings={};downloads={}
    phases=selected_phases(spec,bindings,downloads)
    runtime=qualified_runtime(phases['qualification'][0],phases['qualification'][2])
    volume_environment=qualified_volume(phases['qualification'][0],phases['qualification'][2])
    for plan,_,directory,_ in phases.values():
        if qualified_volume(plan,directory)!=volume_environment:raise EvidenceError('selected phases use different volume quota guards')
        if qualified_runtime(plan,directory)!=runtime:
            raise EvidenceError('selected phases use different runtimes')
    profiles={kind:{**profile,'remote_root':profile['remote_root']+'-'+name} for kind,name in
              [('production-record','production-record'),('full-replay','production-replay')]}
    transports={kind:transport(value,key,known) for kind,value in profiles.items()}
    for kind,t in transports.items():
        job=output/kind/'job.json'
        if job.exists():bindings[digest(read_json(job))]={'job_file':job,'worker_sha256':spec['selection']['worker_sha256'],
                                                       'control':control,'transports':[t]}
    with Run(spec['selection'],digest(spec['selection']),read_json(Path(spec['rental'])),control,Path(spec['worker']),
             Path(spec['controller']),Path(spec['watchdog']),output,Path(spec['health']),bindings,downloads) as owner:
        if owner.health.complete:
            owner.health.write(owner.health_file)
            final=output/'final/result.json'
            if not final.exists():raise EvidenceError('earlier failed paid work closed; no replacement launch')
            return read_json(final)
        q=owner.phase('qualification',*phases['qualification'])
        variant='baseline'
        if optimized:variant,q=owner.optimize(phases['optimization'])
        name='initialization-'+variant if optimized else 'initialization'
        initial=owner.phase(name,*phases[name])
        decision=output/'registration-selection.json'
        if not decision.exists():save_once(decision,{'invocation_sha256':expected,'selected_epoch':owner.health.now(),
                    'qualification_sha256':digest(q),'initialization_sha256':digest(initial)})
        selected=read_json(decision)
        if (selected['invocation_sha256']!=expected or selected['qualification_sha256']!=digest(q)
            or selected['initialization_sha256']!=digest(initial)):
            raise EvidenceError('original registration parents changed')
        if registration_templates(spec)!=templates:raise EvidenceError('admitted registration templates changed')
        r,basis=registration(templates[variant],q,initial,owner.rental,selected['selected_epoch'],
                             construction_seconds=spec['selection']['timing']['registration_seconds'])
        save_once(output/'forecast-basis.json',basis)
        if validate_run_key(Path(spec['run_key']),[r])!=run_identity:
            raise EvidenceError('registration run key differs from admitted identity')
        a=owner.register(r,read_json(Path(spec['source_statement'])),Path(spec['source_bundle']),
             PublisherPolicy(**read_json(Path(spec['source_policy']))),read_json(Path(spec['preparation'])),Path(spec['source_checkout']))
        inputs(owner,spec,a)
        static=read_json(Path(spec['static_files']))
        offline=Path(spec['qualification']['inputs'])/'offline-config.json'
        record_output=output/'production-record'
        template=production_job('production-record',profile,offline,static,a['packet'],a['bundle'],
                                 output/'production-policy.json',output/'source-policy.json',runtime_root=runtime,volume_environment=volume_environment)
        job,job_root=owner.select_job(template,record_output)
        owner.stage('production-record',job,job_root,[transports['production-record']],transports['production-record'],record_output,owner.store)
        record=read_json(record_output/'checked-numerical-result.json');chain=Path(record['directory'])
        envelopes=read_json(chain/'chain.json')['boundaries']
        progress,policies=owner.record_publisher.previous(len(envelopes),envelopes)
        if read_json(chain/'external-progress-policies.json')!=[asdict(p) for p in policies]:
            raise EvidenceError('recorded policies differ from independently retained operator policy history')
        replay_output=output/'full-replay'
        template=production_job('full-replay',profile,offline,static,a['packet'],a['bundle'],output/'production-policy.json',
                                 output/'source-policy.json',record_files=record['files'],runtime_root=runtime,volume_environment=volume_environment)
        job,job_root=owner.select_job(template,replay_output)
        owner.stage('full-replay',job,job_root,[transports['full-replay']],transports['full-replay'],replay_output,owner.store,
                    chain=chain,progress_directory=progress,progress_policies=policies)
        return owner.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True)
    a=p.parse_args()
    try:
        value=run(read_json(a.spec),a.spec_sha256)
        print('Paid work retained; release acceptance remains separate. Completion '+digest(value))
    except Exception as error:p.exit(1,'production lifetime refused: '+type(error).__name__+'; preserve guards and outputs\n')

if __name__=='__main__':main()
