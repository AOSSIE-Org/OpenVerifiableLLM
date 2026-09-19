"""Actual tiny reconstructed data/states; explicitly synthetic provider/audit identities."""
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import assemble_computation_evidence as m
from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
from test_production_release import setup
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json
from ovl_pipeline import production_replay,production_trajectory,source_commitment
from ovl_pipeline.state import save_state
from types import SimpleNamespace


def configured(inputs,prepared,tmp_path,monkeypatch):
    r,reports,_,_=setup(inputs,prepared,tmp_path,monkeypatch)
    old=read_json(reports/'verification.json');full=tmp_path/'full';chain=tmp_path/'checkpoints'
    _,envelopes,_=production_replay.authenticate()
    # The shared record fixture saves only primary states. Execute its same tiny
    # CPU trajectory to supply actual recovery states for this all-state check.
    streams={phase:full/'reconstructed'/phase for phase in ('wikipedia','conversation')}
    for event in production_trajectory.walk(r,streams):
        if event.kind=='recovery':save_state(chain/f"recovery-{event.control['global_step']:09d}",event.model,event.optimizer,event.control)
    obs_path=full.parent/'observation.json'
    observation={'schema':'ovl.preparation-execution.v1','status':'PASS','source_commitment_sha256':r['source_statement_sha256'],
                 'resume_requested':False,'stages_executed_this_run':m.STAGES,'stages_adopted_from_local_cache':[],
                 'scope':'explicit process identity substitute; full tiny reconstruction actually ran in setup'}
    rec=old['reconstruction'];rec['execution_observation_sha256']=digest(observation)
    cp={'schema':'ovl.complete-clean-reconstruction-checkpoint.v2','result':'PASS','independent_third_party':False,
        'whole_job_elapsed_ms':1,'report':rec}
    prior=tmp_path/'prior.json';write_json(prior,cp);write_json(obs_path,observation)
    base={'pod_id':'synthetic','host':'127.0.0.1','port':22,'user':'test','endpoint_observation_sha256':'a'*64,
          'known_hosts_sha256':'b'*64,'host_key_trust':'explicit-test-double'}
    control=SimpleNamespace(profile={**base,'remote_root':'/control'})
    transports=[SimpleNamespace(profile={**base,'remote_root':name}) for name in ('/replay','/audit')]
    selected={n:'/selected/'+n for n in ('packet','registration-bundle','production-policy','source-policy','source-checkout',
        'chain-directory','progress-directory','progress-policies','wikipedia-stream','conversation-stream')}
    selected['output']='/replay';arguments=[item for k,v in selected.items() for item in ('--'+k,v)]
    argv=['/python','setup.py','launch','--module','ovl_pipeline.production_replay','--',*arguments]
    required={selected['packet']+'/registration.json':digest(r),selected['packet']+'/source-statement.json':r['source_statement_sha256'],
        selected['packet']+'/source-statement.sigstore.json':r['source_bundle_sha256'],selected['packet']+'/preparation.json':r['preparation_sha256']}
    required.update({selected[p+'-stream']+'/stream.json':r['coverage'][p]['stream_sha256'] for p in ('wikipedia','conversation')})
    job={'kind':'full-replay','export_roots':['/replay','/audit'],'argv':argv,
         'required_files':[{'path':k,'sha256':v,'bytes':1} for k,v in required.items()]}
    job_path=tmp_path/'job.json';write_json(job_path,job);job_root=digest(job);worker='c'*64
    terminal={'schema':'ovl.workload-job-exit.v1','job_sha256':job_root,'state':'EXITED','exit_code':0}
    control_dir=tmp_path/'control-export';control_dir.mkdir();write_json(control_dir/'exit.json',terminal)
    audit_parent=tmp_path/'audit-export';audit=audit_parent/'audit';audit.mkdir(parents=True)
    manifest={'dependency_lock_sha256':r['runtime']['dependency_lock_sha256']}
    installed={'result':'PASS','wheel_manifest_sha256':digest(manifest)}
    payloads={'archive_sha256':'d'*64};python={'result':'PASS','archive_manifest_sha256':digest(payloads),'archive_sha256':'d'*64}
    launch={'schema':'ovl.audited-runtime-launch.v1','module':'ovl_pipeline.production_replay','arguments':arguments,'source':selected['source-checkout']+'/src',
        'dependency_lock_sha256':r['runtime']['dependency_lock_sha256'],'wheel_manifest_sha256':digest(manifest),
        'installed_audit_sha256':digest(installed),'interpreter_origin':{'archive_sha256':'d'*64,
        'manifest_sha256':digest(payloads),'audit_sha256':digest(python)}}
    process={'schema':'ovl.audited-runtime-process.v1','launch_sha256':digest(launch),'exit_code':0,
             'scope':'external package audit and constrained target-process launch; not model verification by itself'}
    for name,value in [('wheel-payloads',manifest),('installed-audit',installed),('python-payloads',payloads),
                        ('python-audit',python),('launch',launch),('process',process)]:write_json(audit/(name+'.json'),value)
    roots=[];replay_dir=full/'numerical-replay'
    for index,(directory,transport,name,absolute) in enumerate([
        (control_dir,control,'jobs/'+job_root,'/control/jobs/'+job_root),
        (replay_dir,transports[0],'.','/replay'),(audit_parent,transports[1],'.','/audit')]):
        receipt={'schema':'ovl.offpod-versioned-tree-export.v1','pod_id':'synthetic','profile_sha256':digest(transport.profile),
                 'root':name,'result':'PASS','numerical_verification':'NOT_RUN','files_directory':str(directory),
                 'files':inventory(directory,sorted(p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()))}
        receipt_path=tmp_path/f'receipt-{index}.json';write_json(receipt_path,receipt)
        roots.append({'declared_root':absolute,'profile_sha256':digest(transport.profile),'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt)})
    proof={'schema':'ovl.production-terminal-retention.v1','job_sha256':job_root,'worker_sha256':worker,'pod_id':'synthetic',
           'terminal':terminal,'roots':roots,'scope':'explicit provider identity substitute with actual complete CPU artifact bytes',
           'training_replay':'NOT_RUN','production_acceptance':'NOT_RUN'}
    proof_path=tmp_path/'retention.json';write_json(proof_path,proof)
    replay=read_json(replay_dir/'verification.json')
    record_files=inventory(chain,sorted(p.relative_to(chain).as_posix() for p in chain.rglob('*') if p.is_file()))
    published={'schema':'ovl.retained-full-replay-publication.v1','registration_sha256':digest(r),
        'job_sha256':job_root,'worker_sha256':worker,'retention_sha256':digest(proof),'launch_sha256':digest(launch),
        'process_sha256':digest(process),'terminal_sha256':digest(terminal),'report_sha256':digest(replay),
        'record_inventory_sha256':digest(record_files),'replay_inventory_sha256':digest(read_json(tmp_path/'receipt-1.json')['files'])}
    prefix='https://raw.githubusercontent.com/AOSSIE-Org/OpenVerifiableLLM/'+'1'*40+'/project/evidence/synthetic/'
    from ovl_pipeline.canonical import canonical
    public={prefix+'prior.json':canonical(cp),prefix+'replay.json':canonical(published)}
    monkeypatch.setattr(source_commitment,'fetch_metadata',lambda url:public[url])
    args=dict(packet=tmp_path/'packet',bundle=None,production_policy=None,source_policy=None,source_checkout=tmp_path,
        chain=chain,progress=tmp_path/'progress',progress_policies=[],raw=tmp_path/'raw',prepared=full/'reconstructed',
        exports=tmp_path/'export',context=read_json(reports/'context.json'),prior_checkpoint=prior,prior_observation=obs_path,
        prior_public={'url':prefix+'prior.json','sha256':digest(cp)},replay_public={'url':prefix+'replay.json','sha256':digest(published)},
        control=control,transports=transports,job_file=job_path,job_sha256=job_root,worker_sha256=worker,
        retention=proof_path,retention_sha256=digest(proof),replay_root='/replay',audit_root='/audit',audit_relative='audit',
        output=tmp_path/'assembled')
    return args,r,public


def test_join_actual_complete_tiny_reconstruction_replay_and_evaluation(cpu_runtime,inputs,prepared,tmp_path,monkeypatch):
    args,r,_=configured(inputs,prepared,tmp_path,monkeypatch)
    value=m.assemble(**args)
    assert value['schema']=='ovl.complete-computation-verification.v2' and value['locally_recomputed'] is False
    assert value['execution']['assembly']['numerical_updates_executed']==0
    assert value['execution']['assembly']['safe_states_compared']>len(read_json(tmp_path/'full/numerical-replay/verification.json')['comparisons'])
    with pytest.raises(EvidenceError,match='fresh output'):m.assemble(**args)


@pytest.mark.parametrize('damage',['public-byte','observation','prepared-byte','state-byte','failed-exit','missing-audit','changed-launch'])
def test_missing_or_altered_execution_evidence_never_assembles_pass(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    args,r,public=configured(inputs,prepared,tmp_path,monkeypatch)
    if damage=='public-byte':public[args['prior_public']['url']]+=b' '
    elif damage=='observation':write_json(args['prior_observation'],{'result':'PASS'})
    elif damage=='prepared-byte':
        manifest=read_json(args['prepared']/'preparation.json')
        (args['prepared']/'wikipedia'/manifest['streams']['wikipedia']['files'][0]['path']).write_bytes(b'changed')
    elif damage=='state-byte':(tmp_path/'full/numerical-replay/verifier-boundary-00000/state.safetensors').write_bytes(b'changed')
    elif damage=='failed-exit':
        v=read_json(args['retention']);v['terminal']['exit_code']=1;write_json(args['retention'],v)
    elif damage=='missing-audit':(tmp_path/'audit-export/audit/python-audit.json').unlink()
    else:
        v=read_json(tmp_path/'audit-export/audit/launch.json');v['arguments']=['--output','/other'];write_json(tmp_path/'audit-export/audit/launch.json',v)
    with pytest.raises((EvidenceError,OSError)):m.assemble(**args)
    assert not(args['output']/'verification.json').exists()
