"""Standard-library-only target bootstrap after an external package audit.

Invoke through runtime_launch, never as evidence that an unaudited runtime is safe.
No .pth hooks, user site or pre-existing bytecode cache is admitted. The parent is
part of the operator's trusted verifier environment, not the target being audited.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import sysconfig


def bootstrap():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--site',required=True,type=Path)
    p.add_argument('--launch-record',required=True,type=Path);p.add_argument('--module',required=True)
    p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.arguments[:1]==['--']:a.arguments=a.arguments[1:]
    if not(sys.flags.no_site and sys.flags.no_user_site and sys.flags.safe_path) or sys.flags.hash_randomization:
        raise RuntimeError('target must start with -s -S -P and PYTHONHASHSEED=0')
    if os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME') or os.environ.get('PYTHONSTARTUP'):
        raise RuntimeError('unexpected Python startup path configuration')
    if sys.pycache_prefix is None:raise RuntimeError('fresh bytecode prefix required')
    cache=Path(sys.pycache_prefix)
    # The interpreter may already have generated its own stdlib caches before
    # reaching this file. Freshness is established by the parent mkdir, not by
    # pretending this directory is still empty after interpreter initialization.
    if not cache.is_dir() or cache.is_symlink():raise RuntimeError('invalid bytecode cache directory')
    source=a.source.resolve(strict=True);site=a.site.resolve(strict=True)
    base=Path(sysconfig.get_path('stdlib')).resolve(strict=True)
    paths=[str(source),str(base),str(base/'lib-dynload'),str(site)]
    sys.path[:]=paths
    record=json.loads(a.launch_record.read_text())
    if (record['schema']!='ovl.audited-runtime-launch.v1' or record['source']!=str(source)
        or record['site']!=str(site) or record['pycache_prefix']!=str(cache)
        or record['module']!=a.module or record['arguments']!=a.arguments):
        raise RuntimeError('launch selection differs from audited parent record')
    os.environ['OVL_AUDITED_RUNTIME_LAUNCH']=str(a.launch_record.resolve())
    sys.argv=[a.module]+a.arguments
    runpy.run_module(a.module,run_name='__main__',alter_sys=True)


if __name__=='__main__':bootstrap()
