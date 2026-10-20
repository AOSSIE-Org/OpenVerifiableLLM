"""Linux parent-death binding for bounded numerical process trees.

This is process containment, not scientific verification or provider shutdown.
The caller supplies an exact live parent identity; no other process fd is read.
"""
import ctypes
import os
from pathlib import Path
import signal


import re
import struct
import sys


def identity(pid):
    proc_stat = Path(f'/proc/{pid}/stat')
    if proc_stat.is_file():
        stat = proc_stat.read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': stat[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    if sys.platform == 'darwin':
        try:
            from .canonical import host_boot_id, host_process_stat
            _, ticks, _ = host_process_stat(pid)
            return {'pid': pid, 'start_ticks': ticks, 'boot_id': host_boot_id()}
        except Exception:
            pass
        libc = ctypes.CDLL(None, use_errno=True)
        buf = ctypes.create_string_buffer(64)
        size = ctypes.c_size_t(64)
        boot_id = '0' * 8 + '-0000-0000-0000-' + '0' * 12
        if libc.sysctlbyname(b'kern.bootsessionuuid', buf, ctypes.byref(size), None, 0) == 0:
            raw = buf.value.decode('ascii').strip().lower()
            if re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', raw):
                boot_id = raw
        mib = (ctypes.c_int * 4)(1, 14, 1, pid)
        kbuf = ctypes.create_string_buffer(648)
        ksize = ctypes.c_size_t(648)
        if libc.sysctl(mib, 4, kbuf, ctypes.byref(ksize), None, 0) == 0 and ksize.value >= 648:
            tv_sec, tv_usec = struct.unpack_from('qq', kbuf.raw, 0)
            ticks = str(tv_sec * 100 + tv_usec // 10000)
            return {'pid': pid, 'start_ticks': ticks, 'boot_id': boot_id}
    raise FileNotFoundError(f'/proc/{pid}/stat')


def bind_parent(expected):
    if (type(expected) is not dict or set(expected)!={'pid','start_ticks','boot_id'}
            or type(expected['pid']) is not int or expected['pid']<=1):
        raise RuntimeError('invalid numerical parent identity')
    libc=ctypes.CDLL(None,use_errno=True)
    # SIGKILL is intentional: a lost enforcing parent cannot leave an unbounded
    # numerical descendant. Check again after prctl to close the startup race.
    if hasattr(libc, 'prctl') and libc.prctl(1,signal.SIGKILL,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'cannot bind numerical parent death')
    if not hasattr(libc, 'prctl') and sys.platform == 'darwin':
        import threading, time
        def _watch_parent():
            while True:
                time.sleep(0.02)
                try:
                    if os.getppid() != expected['pid'] or identity(os.getppid()) != expected:
                        os.kill(os.getpid(), signal.SIGKILL)
                except Exception:
                    os.kill(os.getpid(), signal.SIGKILL)
        threading.Thread(target=_watch_parent, daemon=True).start()
    if os.getppid()!=expected['pid'] or identity(os.getppid())!=expected:
        raise RuntimeError('numerical enforcing parent changed before startup')
