"""Bounded local-only failure evidence; never an export or retry authority.

Callers select a private directory outside declared remote/public export roots.
Raw diagnostic bytes are never added to exception messages or printed here.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import traceback
import uuid

PREFIX_BYTES = 65536
MAX_RECORD_BYTES = 256 * 1024
MAX_RECORDS = 128
MAX_TOTAL_BYTES = 32 * 1024 * 1024


def bounded_bytes(data):
    data=bytes(data)
    return {'received_bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'prefix_b64':base64.b64encode(data[:PREFIX_BYTES]).decode(),
            'truncated':len(data)>PREFIX_BYTES}


def retain(error,directory,context):
    """Best effort only: the exact primary error and cleanup remain authoritative."""
    if directory is None or getattr(error,'private_diagnostic_status',{}).get('result')=='RETAINED':return
    try:
        directory=Path(directory)
        if any(p.is_symlink() for p in [directory,*directory.parents]):raise ValueError('symlink diagnostic path')
        directory.mkdir(mode=0o700,exist_ok=True)
        info=directory.stat()
        if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:raise ValueError('private diagnostic permissions')
        entries=list(directory.iterdir())
        if len(entries)>=MAX_RECORDS:raise ValueError('diagnostic record cap')
        total=0
        for entry in entries:
            item=entry.lstat()
            if not stat.S_ISREG(item.st_mode):raise ValueError('diagnostic entry type')
            total+=item.st_size
        frames=[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}
                for f in traceback.extract_tb(error.__traceback__)[-16:]]
        value={'schema':'ovl.private-read-failure.v1','scope':'LOCAL_ONLY; no retry, progress or verification credit',
               'context':context,'exception_class':type(error).__name__,'message':str(error)[:4096],
               'frames':frames,'transport':getattr(error,'transport_diagnostic',None),
               'metadata_response':getattr(error,'metadata_response_diagnostic',None),
               'cleanup':getattr(error,'transport_cleanup_diagnostic',None)}
        data=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
        if len(data)>MAX_RECORD_BYTES or total+len(data)>MAX_TOTAL_BYTES:raise ValueError('diagnostic byte cap')
        path=directory/(uuid.uuid4().hex+'.json')
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        error.private_diagnostic_status={'result':'RETAINED','path':str(path)}
    except Exception as failure:
        # Preserve the primary exception, its classification, and process cleanup.
        error.private_diagnostic_status={'result':'UNAVAILABLE','category':type(failure).__name__}


def capture(error,directory,context):
    """Enclosing dispatch paths call this before reducing an error to its class."""
    try:retain(error,directory,context)
    except Exception:pass
