"""External full package audit followed by a constrained fresh target process.

Run this parent in the operator's separately trusted verifier environment. It never
provisions resources or authorizes production updates. The selected module retains
its own required source, publisher, data, initialization and trajectory gates.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

from .canonical import EvidenceError,digest,file_hash,read_json,write_json
from .runtime_audit import wheel_manifest,verify_installed

MODULES={'ovl_pipeline','ovl_pipeline.gpu_pilot','ovl_pipeline.initialization',
         'ovl_pipeline.production_replay','ovl_pipeline.production_record','ovl_pipeline.production_export',
         'ovl_pipeline.runtime_audit','ovl_pipeline.runtime_launch'}
DETERMINISTIC_ENV={'CUBLAS_WORKSPACE_CONFIG':':4096:8','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'0',
                   'CUBLASLT_WORKSPACE_SIZE':'32768','TORCH_CUBLASLT_UNIFIED_WORKSPACE':'1',
                   'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONHASHSEED':'0',
                   'USE_PYTORCH_KERNEL_CACHE':'0'}


def current_launch():
    """Check selected parent evidence/startup flags; no hardware attestation."""
    selected=os.environ.get('OVL_AUDITED_RUNTIME_LAUNCH')
    if not selected:raise EvidenceError('GPU runtime requires the external audited launcher')
    path=Path(selected);record=read_json(path)
    manifest=read_json(path.parent/'wheel-payloads.json');audit=read_json(path.parent/'installed-audit.json')
    if (record.get('schema')!='ovl.audited-runtime-launch.v1' or record['wheel_manifest_sha256']!=digest(manifest)
        or record['installed_audit_sha256']!=digest(audit) or audit.get('result')!='PASS'
        or audit['wheel_manifest_sha256']!=digest(manifest) or record['dependency_lock_sha256']!=manifest['dependency_lock_sha256']):
        raise EvidenceError('audited launch parent mismatch')
    base=Path(sysconfig.get_path('stdlib')).resolve(strict=True)
    expected=[record['source'],str(base),str(base/'lib-dynload'),record['site']]
    if sys.path!=expected or not(sys.flags.no_site and sys.flags.no_user_site and sys.flags.safe_path) or sys.flags.hash_randomization:
        raise EvidenceError('audited target import/startup settings changed')
    if sys.pycache_prefix!=record['pycache_prefix'] or not Path(sys.pycache_prefix).is_dir():
        raise EvidenceError('audited target bytecode cache changed')
    if file_hash(Path('/proc/self/exe'))!=record['python_executable_sha256']:
        raise EvidenceError('target interpreter differs from audited launch')
    origin=None
    if record.get('interpreter_origin') is not None:
        selected=record['interpreter_origin'];payloads=read_json(path.parent/'python-payloads.json');checked=read_json(path.parent/'python-audit.json')
        if (selected['manifest_sha256']!=digest(payloads) or selected['audit_sha256']!=digest(checked)
            or checked['archive_manifest_sha256']!=digest(payloads) or checked['result']!='PASS'
            or checked['archive_sha256']!=selected['archive_sha256'] or payloads['archive_sha256']!=selected['archive_sha256']
            or base!=Path(selected['root'])/'python/lib/python3.12'
            or Path(sys.base_prefix).resolve()!=Path(selected['root'])/'python'):
            raise EvidenceError('audited interpreter/standard-library origin differs')
        origin={'archive_sha256':selected['archive_sha256'],'payloads_sha256':digest(payloads)}
    return {'wheel_payloads_sha256':digest(manifest),'dependency_lock_sha256':manifest['dependency_lock_sha256'],
            'allowed_generated':record['allowed_generated'],'startup':'no-site-no-user-site-safe-path-fresh-bytecode-v1',
            'interpreter_origin':origin,
            'scope':'operator-observed external complete package audit and constrained process; not runtime attestation'}


def launch(lock,wheels,venv,source,output,module,arguments,*,allowed_generated=None,execute=subprocess.run,
           interpreter_archive=None,interpreter_sha256=None,interpreter_root=None,progress=None,bytecode_root=None,
           lease_fd=None,lease_path=None):
    if module not in MODULES:raise EvidenceError('unsupported audited target module')
    if output.exists():raise EvidenceError('launch requires a fresh record and bytecode cache')
    if venv.is_symlink() or source.is_symlink():raise EvidenceError('target environment/source roots must be regular directories')
    venv=venv.resolve(strict=True);source=source.resolve(strict=True)
    python=venv/'bin/python';site=venv/'lib/python3.12/site-packages'
    if not python.is_file() or not site.is_dir():raise EvidenceError('declared Python3.12 environment required')
    bootstrap=source/'ovl_pipeline/runtime_bootstrap.py'
    if file_hash(bootstrap)!=file_hash(Path(__file__).with_name('runtime_bootstrap.py')):
        raise EvidenceError('selected bootstrap differs from trusted verifier source')
    origin=None;python_payloads=python_audit=None
    if any(v is not None for v in (interpreter_archive,interpreter_sha256,interpreter_root)):
        if any(v is None for v in (interpreter_archive,interpreter_sha256,interpreter_root)):
            raise EvidenceError('complete external interpreter origin selection required')
        from .python_origin import manifest as origin_manifest,audit as origin_audit
        python_payloads=origin_manifest(interpreter_archive,interpreter_sha256)
        python_audit=origin_audit(python_payloads,interpreter_root)
        interpreter_root=interpreter_root.resolve(strict=True)
        if python.resolve(strict=True)!=interpreter_root/'python/bin/python3.12':
            raise EvidenceError('target interpreter is outside audited public distribution')
        origin={'archive_sha256':interpreter_sha256,'manifest_sha256':digest(python_payloads),
                'audit_sha256':digest(python_audit),'root':str(interpreter_root)}
    if progress is not None:progress('launch-interpreter')
    # Complete archive and installed-file hashing occurs in the trusted parent,
    # before any code from the selected environment is imported by the child.
    manifest=wheel_manifest(lock,wheels)
    if progress is not None:progress('launch-wheels')
    audit=verify_installed(manifest,{'site':site,'prefix':venv,'scripts':venv/'bin','headers':venv/'include/python3.12'},
                           allowed_generated=allowed_generated)
    if progress is not None:progress('launch-installed')
    output.mkdir(parents=True,exist_ok=False);output=output.resolve()
    if bytecode_root is None:
        cache=output/'pycache';cache.mkdir(mode=0o700)
    else:
        bytecode_root=Path(bytecode_root).absolute()
        if any(p.is_symlink() for p in [bytecode_root,*bytecode_root.parents]):raise EvidenceError('bytecode root symlink')
        bytecode_root.mkdir(mode=0o700,parents=True,exist_ok=True)
        cache=Path(tempfile.mkdtemp(prefix='launch-',dir=bytecode_root))
    # A fresh local prefix excludes old caches. -B prevents per-launch cache
    # growth and remote bytecode writes without admitting pre-existing code.

    write_json(output/'wheel-payloads.json',manifest);write_json(output/'installed-audit.json',audit)
    if origin is not None:
        write_json(output/'python-payloads.json',python_payloads);write_json(output/'python-audit.json',python_audit)
    record={'schema':'ovl.audited-runtime-launch.v1','wheel_manifest_sha256':digest(manifest),'installed_audit_sha256':digest(audit),
            'dependency_lock_sha256':file_hash(lock),'python_executable_sha256':file_hash(python),'bootstrap_sha256':file_hash(bootstrap),
            'source':str(source),'site':str(site),'pycache_prefix':str(cache),'module':module,'arguments':arguments,
            'allowed_generated':allowed_generated or {},'interpreter_origin':origin,
            'performed_by':'project-operator','production_admission':'NOT_RUN'}
    if lease_fd is not None:
        from .process_safety import identity
        record['lifecycle_parent']=identity(os.getpid())
    write_json(output/'launch.json',record)
    # Do not pass the operator's credentials, numerical overrides or executable
    # search path to the numerical child. Only the selected liveness path crosses
    # this boundary; it does not authorize updates or numerical acceptance.
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','HOME':str(output)}
    if 'OVL_ACTIVITY_FILE' in os.environ:env['OVL_ACTIVITY_FILE']=os.environ['OVL_ACTIVITY_FILE']
    env.update(DETERMINISTIC_ENV)
    command=[str(python),'-B','-s','-S','-P','-X','pycache_prefix='+str(cache),str(bootstrap),
             '--source',str(source),'--site',str(site),'--launch-record',str(output/'launch.json'),'--module',module,'--',*arguments]
    # REMAINDER retains the '--'; remove it only in the bootstrap argument parser
    # and record the exact user arguments independently of shell interpolation.
    execution={}
    if lease_fd is not None or lease_path is not None:
        # Preserve the operating owner's lease through this extra process layer.
        # It grants no scientific admission and passes no operator environment.
        import fcntl
        if type(lease_fd) is not int or lease_fd<0 or lease_path is None:
            raise EvidenceError('complete workload lease selection required')
        selected=os.fstat(lease_fd);path=Path(lease_path)
        expected=path.stat()
        if path.is_symlink() or (selected.st_dev,selected.st_ino)!=(expected.st_dev,expected.st_ino):
            raise EvidenceError('workload lease descriptor differs')
        fcntl.flock(lease_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        execution['pass_fds']=(lease_fd,)
    result=execute(command,env=env,check=False,**execution)
    receipt={'schema':'ovl.audited-runtime-process.v1','launch_sha256':digest(record),'exit_code':result.returncode,
             'scope':'external package audit and constrained target-process launch; not model verification by itself'}
    write_json(output/'process.json',receipt)
    if result.returncode:raise EvidenceError('audited target process failed; preserve launch/output evidence')
    return receipt


def main():
    if sys.argv[1:]==['--inspect-current']:
        from .canonical import canonical
        print(canonical(current_launch()).decode());return
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('lock','wheels','venv','source','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--module',required=True,choices=sorted(MODULES));p.add_argument('--allowed-generated',type=Path)
    p.add_argument('--interpreter-archive',type=Path);p.add_argument('--interpreter-sha256');p.add_argument('--interpreter-root',type=Path)
    p.add_argument('--lifecycle-lease-fd',type=int);p.add_argument('--lifecycle-lease-path',type=Path)
    p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args();args=a.arguments
    if args[:1]==['--']:args=args[1:]
    result=launch(a.lock,a.wheels,a.venv,a.source,a.output,a.module,args,
                  allowed_generated=read_json(a.allowed_generated) if a.allowed_generated else None,
                  interpreter_archive=a.interpreter_archive,interpreter_sha256=a.interpreter_sha256,interpreter_root=a.interpreter_root,
                  lease_fd=a.lifecycle_lease_fd,lease_path=a.lifecycle_lease_path)
    print('Audited target exited successfully; launch receipt '+digest(result))

if __name__=='__main__':main()
