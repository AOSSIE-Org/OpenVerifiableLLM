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
    global _process,_sequence,_last
    selected=os.environ.get('OVL_ACTIVITY_FILE')
    if not selected:return
    path=Path(selected)
    if not path.is_absolute() or path.name!='activity.json' or path.is_symlink() or not path.parent.is_dir():
        raise EvidenceError('activity output must be an explicit regular activity.json in an existing directory')
    now=clock()
    if _last is not None and 0<=now-_last<30:return
    if _process is None:_process=uuid.uuid4().hex
    _sequence+=1
    write_json(path,{'schema':'ovl.runtime-activity.v1','process_instance':_process,'pid':os.getpid(),
                     'sequence':_sequence,'kind':'completed-numerical-update','control':control,
                     'scope':'operator-supervision-only-not-training-verification'})
    _last=now
