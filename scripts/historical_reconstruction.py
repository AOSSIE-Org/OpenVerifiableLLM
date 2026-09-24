"""Pinned historical preparation execution for the external complete verifier.

The clean retained revision supplies bytes only; original signer identity remains
unchanged. Isolated Python and trusted installed dependencies are assumptions,
not sandboxing or hardware attestation. No credentials enter the child environment.
"""
from dataclasses import asdict
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

from ovl_pipeline.canonical import EvidenceError,canonical,digest,file_hash,inventory,read_json,verify_inventory,write_json
from ovl_pipeline.production_preparation import PREPARATION_PATHS,PREPARATION_SNAPSHOTS,_materialize

DRIVER_FILES=['historical_reconstruction.py','verify_complete.py','verify_release_complete.py']


def driver_identity(source_checkout,*,entrypoint=None):
    from ovl_pipeline import training
    import ovl_pipeline
    source=Path(source_checkout).resolve(strict=True)/'src'
    if Path(training.__file__).resolve().parent!=source/'ovl_pipeline':
        raise EvidenceError('driver must import the selected frozen source checkout')
    for name,module in tuple(sys.modules.items()):
        if name=='ovl_pipeline' or name.startswith('ovl_pipeline.'):
            if not Path(module.__file__).resolve().is_relative_to(source/'ovl_pipeline'):
                raise EvidenceError('driver imported mixed pipeline source trees')
    directory=Path(__file__).resolve().parent
    if entrypoint is not None and Path(entrypoint).resolve() not in [directory/n for n in DRIVER_FILES]:
        raise EvidenceError('executing driver differs from selected helper directory')
    for name in ('historical_reconstruction','verify_complete','verify_release_complete','__main__'):
        module=sys.modules.get(name)
        origin=getattr(module,'__file__',None)
        if origin is not None and Path(origin).name in DRIVER_FILES and Path(origin).resolve()!=directory/Path(origin).name:
            raise EvidenceError('mixed verification driver origins')
    return {'schema':'ovl.complete-verifier-driver.v1','files':inventory(directory,DRIVER_FILES),
            'frozen_code_root':training.code_root(),'scope':'separately selected driver bytes; not publisher authentication'}


def admit_runtime_source(source_checkout,runtime):
    selected=Path(runtime['source'])
    expected=Path(source_checkout).resolve(strict=True)/'src'
    if selected.is_symlink() or selected.resolve(strict=True)!=expected:
        raise EvidenceError('replay child source differs from authenticated frozen checkout')


_WORKER=r'''
import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
request_path=root/'request.json'
assert hashlib.sha256(request_path.read_bytes()).hexdigest()==sys.argv[2]
request=json.loads(request_path.read_bytes())
sys.path.insert(0,str(root/'kernel/src'))
from ovl_pipeline.canonical import digest,inventory,read_json,verify_inventory,write_json
from ovl_pipeline.anchoring import PublisherPolicy
from ovl_pipeline.preparation import prepare_committed
verify_inventory(root/'kernel',request['code'])
for name,expected in request['inputs'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
statement=read_json(root/'statement.json')
assert digest(statement)==request['statement_sha256']
report=prepare_committed(root/'statement.json',root/'bundle.json',PublisherPolicy(**read_json(root/'policy.json')),
    pathlib.Path(request['raw'])/'wikipedia',pathlib.Path(request['raw'])/'conversation',pathlib.Path(request['output']),
    expected_preparation=read_json(root/'expected.json'),resume=False)
for name,module in tuple(sys.modules.items()):
    if name=='ovl_pipeline' or name.startswith('ovl_pipeline.'):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root/'kernel/src/ovl_pipeline'),name
assert inventory(root/'kernel',[e['path'] for e in request['code']])==request['code']
for name,expected in request['inputs'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256(request_path.read_bytes()).hexdigest()==sys.argv[2]
write_json(root/'result.json',{'report':report,'nonce':request['nonce'],'pid':os.getpid(),
    'request_sha256':sys.argv[2],'scope':'fresh historical preparation child; not independent attestation'})
'''


def reconstruct(source_checkout,statement_path,bundle_path,policy,raw,output,expected,execution):
    """Always authenticate again, then execute all stages in a fresh child."""
    from ovl_pipeline.anchoring import verify_anchor
    statement=read_json(statement_path)
    admission=verify_anchor(statement_path,bundle_path,policy)
    root=digest(statement)
    if root!=policy.statement_sha256 or admission['statement_sha256']!=root:
        raise EvidenceError('historical reconstruction source admission differs')
    revision=PREPARATION_SNAPSHOTS.get(root)
    if revision is None:raise EvidenceError('no reviewed historical preparation snapshot')
    if expected['source_commitment_sha256']!=root:raise EvidenceError('historical preparation parent differs')
    if output.exists() or execution.exists():raise EvidenceError('historical reconstruction requires fresh output and execution')
    execution.mkdir(parents=True,exist_ok=False);execution=execution.resolve()
    _materialize(source_checkout,revision,statement['code'],execution/'kernel')
    inputs={'statement.json':statement,'policy.json':asdict(policy),'expected.json':expected}
    for name,value in inputs.items():write_json(execution/name,value)
    # Sigstore bundles are authenticated exact bytes, not canonical OVL JSON.
    (execution/'bundle.json').write_bytes(bundle_path.read_bytes())
    input_names=[*inputs,'bundle.json']
    request={'statement_sha256':root,'code':statement['code'],'raw':str(raw.resolve(strict=True)),
             'output':str(output.resolve()),'inputs':{name:file_hash(execution/name) for name in input_names},'nonce':secrets.token_hex(32)}
    write_json(execution/'request.json',request);request_hash=file_hash(execution/'request.json')
    environment={'LC_ALL':'C.UTF-8'}
    if 'TOKENIZERS_PARALLELISM' in os.environ:environment['TOKENIZERS_PARALLELISM']=os.environ['TOKENIZERS_PARALLELISM']
    started=time.monotonic_ns()
    # No short internal timeout: complete raw reconstruction is potentially long.
    # Caller-owned resource/cost guards continue to govern any paid execution.
    with (execution/'stdout.log').open('xb') as stdout,(execution/'stderr.log').open('xb') as stderr:
        child=subprocess.Popen([sys.executable,'-I','-B','-c',_WORKER,str(execution),request_hash],
                               cwd=execution,env=environment,stdout=stdout,stderr=stderr)
        try:code=child.wait()
        except BaseException:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    process={'schema':'ovl.historical-reconstruction-process.v1','request_sha256':request_hash,
             'worker_sha256':digest(_WORKER),'pid':child.pid,'exit_code':code,'retained_source_revision':revision,
             'original_source_revision':statement['source_revision'],'elapsed_ms':(time.monotonic_ns()-started+999999)//1000000}
    write_json(execution/'process.json',process)
    if code!=0:raise EvidenceError('historical reconstruction child failed; preserve execution logs')
    if file_hash(execution/'request.json')!=request_hash or inventory(execution/'kernel',PREPARATION_PATHS)!=statement['code']:
        raise EvidenceError('historical execution inputs changed')
    for name,expected_hash in request['inputs'].items():
        if file_hash(execution/name)!=expected_hash:raise EvidenceError('historical execution input changed')
    result=read_json(execution/'result.json')
    if (set(result)!={'report','nonce','pid','request_sha256','scope'} or result['nonce']!=request['nonce']
        or result['pid']!=child.pid or result['request_sha256']!=request_hash):
        raise EvidenceError('historical result disconnected from fresh child')
    return result['report']


def check_replay_launch(output,process,r,envelopes,runtime,arguments,replay):
    """Bind the fresh invocation, audited runtime and complete replay session."""
    launch=read_json(output/'gpu-launch/launch.json')
    session=read_json(output/'numerical-replay/session.json')
    progress=read_json(output/'numerical-replay/progress.json')
    if (process!=read_json(output/'gpu-launch/process.json') or process['exit_code']!=0
        or process['launch_sha256']!=digest(launch) or launch['module']!='ovl_pipeline.production_export'
        or launch['arguments']!=['replay-check',*arguments] or launch['source']!=str(runtime['source'].resolve())
        or launch['dependency_lock_sha256']!=file_hash(runtime['lock'])
        or launch['interpreter_origin']['archive_sha256']!=runtime['interpreter_sha256']):
        raise EvidenceError('replay launch differs from selected fresh invocation')
    if (session['registration_sha256']!=digest(r) or session['code_root']!=r['code_root']
        or session['chain_sha256']!=digest(envelopes) or session['prover_state_restored'] is not False
        or session['resume_supported'] is not False or replay['session_sha256']!=digest(session)
        or replay['chain_sha256']!=digest(envelopes) or progress['session_sha256']!=digest(session)
        or progress['complete'] is not True or progress['comparisons']!=replay['comparisons']):
        raise EvidenceError('replay session is disconnected or incomplete')
    audit=session['environment']['compatible']['installed_wheel_audit']
    if (audit['dependency_lock_sha256']!=launch['dependency_lock_sha256']
        or audit['wheel_payloads_sha256']!=launch['wheel_manifest_sha256']
        or audit['interpreter_origin']['archive_sha256']!=runtime['interpreter_sha256']):
        raise EvidenceError('replay session runtime audit differs from launch')
