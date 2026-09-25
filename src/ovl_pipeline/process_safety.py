"""Linux parent-death binding for bounded numerical process trees.

This is process containment, not scientific verification or provider shutdown.
The caller supplies an exact live parent identity; no other process fd is read.
"""
import ctypes
import os
from pathlib import Path
import signal


def identity(pid):
    stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':stat[19],
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def bind_parent(expected):
    if (type(expected) is not dict or set(expected)!={'pid','start_ticks','boot_id'}
            or type(expected['pid']) is not int or expected['pid']<=1):
        raise RuntimeError('invalid numerical parent identity')
    libc=ctypes.CDLL(None,use_errno=True)
    # SIGKILL is intentional: a lost enforcing parent cannot leave an unbounded
    # numerical descendant. Check again after prctl to close the startup race.
    if libc.prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'cannot bind numerical parent death')
    if os.getppid()!=expected['pid'] or identity(os.getppid())!=expected:
        raise RuntimeError('numerical enforcing parent changed before startup')
