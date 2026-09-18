"""Optional bounded operator telemetry for the external cost supervisor.

This observation is never training/replay evidence or a durable checkpoint receipt.
It records an actually completed update at most once every 30 seconds. The off-pod
supervisor assigns its own clock when it observes a new sequence; remote timestamps
cannot renew a rental deadline. No RNG or numerical state is changed.
"""
import os
from pathlib import Path
import time
import uuid
from .canonical import EvidenceError,write_json

_process=None
_sequence=0
_last=None


def update(control,*,clock=time.monotonic):
    return _emit({'schema':'ovl.runtime-activity.v1','kind':'completed-numerical-update','control':control},clock=clock)


def stream_validation(stream_sha256,documents,completed_documents,complete,*,clock=time.monotonic):
    """A checked row prefix, not training progress or a verification receipt."""
    return _emit({'schema':'ovl.runtime-stream-validation.v1','stream_sha256':stream_sha256,
                  'documents':documents,'completed_documents':completed_documents,'complete':complete},
                 clock=clock,force=complete)


def _emit(value,*,clock,force=False):
    global _process,_sequence,_last
    selected=os.environ.get('OVL_ACTIVITY_FILE')
    if not selected:return
    path=Path(selected)
    if (not path.is_absolute() or path.name!='activity.json' or any(p.is_symlink() for p in [path,*path.parents])
        or not path.parent.is_dir()):
        raise EvidenceError('activity output must be an explicit regular activity.json in an existing directory')
    now=clock()
    if not force and _last is not None and 0<=now-_last<30:return
    if _process is None:_process=uuid.uuid4().hex
    _sequence+=1
    write_json(path,{**value,'process_instance':_process,'pid':os.getpid(),'sequence':_sequence,
                     'scope':'operator-supervision-only-not-training-verification'})
    _last=now
