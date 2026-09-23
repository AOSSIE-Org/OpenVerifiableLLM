"""Closed descriptor binding; synthetic paths do not claim a numerical execution."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from types import SimpleNamespace
from dataclasses import asdict
import pytest
import production_run_job as m
from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,write_json
from ovl_pipeline.production_anchoring import PACKET_FILES
from test_production_commitment import request


def selected(tmp_path,kind):
    packet,req=request(tmp_path);r=read_json(packet/'registration.json')
    bundle=tmp_path/'bundle';bundle.write_bytes(b'explicit-test-signature-double')
    policy={'explicit-test-policy':True};source=req['source_policy'];out='/selected/record' if kind=='production-record' else '/selected/replay'
    args={'--packet':'/selected/packet','--registration-bundle':'/selected/bundle',
          '--production-policy':'/selected/policy','--source-policy':'/selected/source-policy','--source-checkout':'/selected/source',
          '--wikipedia-stream':'/selected/wiki','--conversation-stream':'/selected/conversation',
          '--output':out,'--progress-policies':'/selected/record/external-progress-policies.json'}
    if kind=='production-record':args.update({'--registration-sha256':digest(r),'--checkpoint-deadline':'1000',
        '--key-directory':'/selected/private','--anchor-directory':out+'/anchors'})
    else:args.update({'--chain-directory':'/selected/record','--progress-directory':'/selected/record/anchors'})
    expected={args['--packet']+'/'+name:file_hash(packet/name) for name in PACKET_FILES}
    expected.update({args['--registration-bundle']:file_hash(bundle),args['--production-policy']:digest(policy),args['--source-policy']:digest(source)})
    expected.update({args['--'+phase+'-stream']+'/stream.json':r['coverage'][phase]['stream_sha256'] for phase in ('wikipedia','conversation')})
    job={'kind':kind,'deadline_epoch':1000,'export_roots':[out],
         'argv':['/selected/python','-I','-S','/selected/pod_runtime_setup.py','launch','--module',
                 'ovl_pipeline.'+('production_record' if kind=='production-record' else 'production_replay'),'--',*[x for pair in args.items() for x in pair]],
         'required_files':[{'path':p,'sha256':h,'bytes':1} for p,h in expected.items()]}
    t=SimpleNamespace(profile={'remote_root':out})
    return job,r,packet,bundle,policy,source,t,kind


@pytest.mark.parametrize('kind',['production-record','full-replay'])
def test_exact_audited_invocation_binds_all_required_public_parents(tmp_path,kind):
    args=selected(tmp_path,kind);result=m.command(*args)
    assert result['--output']==args[6].profile['remote_root']


@pytest.mark.parametrize('option',['--progress-directory','--progress-policies'])
def test_replay_cannot_read_a_different_public_handoff_tree(tmp_path,option):
    args=selected(tmp_path,'full-replay');argv=args[0]['argv']
    argv[argv.index(option)+1]='/selected/unrelated'
    with pytest.raises(EvidenceError,match='replay public handoff'):m.command(*args)


@pytest.mark.parametrize('damage',['module','duplicate','missing-parent','wrong-parent','output','registration','secret-export','ack-path','deadline'])
def test_misbound_invocation_is_rejected_before_worker_launch(tmp_path,damage):
    args=selected(tmp_path,'production-record');job=args[0];argv=job['argv']
    if damage=='module':argv[argv.index('--module')+1]='ovl_pipeline.gpu_pilot'
    elif damage=='duplicate':argv+=['--output','/selected/record']
    elif damage=='missing-parent':job['required_files'].pop()
    elif damage=='wrong-parent':job['required_files'][0]['sha256']='0'*64
    elif damage=='output':argv[argv.index('--output')+1]='/selected/another'
    elif damage=='registration':argv[argv.index('--registration-sha256')+1]='0'*64
    elif damage=='secret-export':job['export_roots'].append('/selected/private')
    elif damage=='ack-path':argv[argv.index('--anchor-directory')+1]='/selected/other-anchors'
    else:argv[argv.index('--checkpoint-deadline')+1]='1001'
    with pytest.raises(EvidenceError):m.command(*args)


def retained_replay_fixture(inputs,prepared,tmp_path,monkeypatch):
    from test_computation_assembly import configured
    from ovl_pipeline.canonical import inventory
    args,r,_=configured(inputs,prepared,tmp_path,monkeypatch)
    job=read_json(args['job_file']);i=job['argv'].index('--')
    job['argv'][i:i]=['--output','/audit/audit'];write_json(args['job_file'],job);root=digest(job)
    # The fixture explicitly substitutes process/provider identity. Update those
    # synthetic identities coherently; the reconstructed/replayed CPU states are
    # actual and remain unchanged, including all recovery states.
    proof=read_json(args['retention']);proof['job_sha256']=root;proof['terminal']['job_sha256']=root
    ref=proof['roots'][0];receipt=read_json(Path(ref['receipt_path']))
    receipt['root']='jobs/'+root;ref['declared_root']='/control/jobs/'+root
    directory=Path(receipt['files_directory']);write_json(directory/'exit.json',proof['terminal'])
    receipt['files']=inventory(directory,['exit.json']);write_json(Path(ref['receipt_path']),receipt);ref['receipt_sha256']=digest(receipt)
    write_json(args['retention'],proof)
    stage={'retention_path':str(args['retention']),'retention_sha256':digest(proof),'terminal':proof['terminal']}
    from ovl_pipeline.production_replay import authenticate
    _,envelopes,_=authenticate()  # Existing explicit identity fixture supplies the actual tiny chain.
    return args,r,root,stage,envelopes


@pytest.mark.parametrize('damage',[None,'sampled-report','corrupt-state','wrong-audit-module'])
def test_complete_actual_cpu_replay_states_and_process_link_before_paid_work_completion(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    args,r,root,stage,envelopes=retained_replay_fixture(inputs,prepared,tmp_path,monkeypatch)
    checked=m.result(r,stage,args['control'],args['transports'],args['job_file'],root,args['worker_sha256'],'/replay','full-replay')
    if damage=='sampled-report':checked['report']['updates_recomputed']['wikipedia']-=1
    elif damage=='corrupt-state':(Path(checked['directory'])/'verifier-boundary-00000/state.safetensors').write_bytes(b'changed')
    elif damage=='wrong-audit-module':
        p=tmp_path/'audit-export/audit/launch.json';v=read_json(p);v['module']='ovl_pipeline.gpu_pilot';write_json(p,v)
    def verify():return m.complete_replay(r,stage,args['control'],args['transports'],args['job_file'],root,args['worker_sha256'],'/replay',args['chain'],envelopes,checked)
    if damage is None:assert verify()['safe_states_checked']>len(envelopes)
    else:
        with pytest.raises(EvidenceError):verify()

from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
