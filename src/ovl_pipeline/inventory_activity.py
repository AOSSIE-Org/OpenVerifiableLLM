"""Actual hash input bytes, never hash success or training evidence.

The selected manifest binds the finite byte count. All original file/hash checks
still execute. Small local inventories emit nothing. Slow reads emit only after
a completed hash update; a stalled read cannot manufacture timer progress.
"""
import os
import time
from . import runtime_activity,schema
from .canonical import EvidenceError,digest

MAX_PASSES=32
_pass_index=0


def observe(manifest):
    global _pass_index
    if not os.environ.get('OVL_ACTIVITY_FILE'):return None
    schema.stream(manifest)
    sizes=[f['bytes'] for f in manifest['files']]
    total=sum(sizes)
    if total<1024**2:return None
    if _pass_index>=MAX_PASSES:raise EvidenceError('inventory observation pass limit exceeded')
    _pass_index+=1;index=_pass_index
    root=digest(manifest);completed=0;current=0;previous=0;last=None
    def progress(file_index,count,verified):
        nonlocal completed,current,previous,last
        if (file_index!=current or type(count) is not int or not previous<=count<=sizes[current]
                or type(verified) is not bool or verified and count!=sizes[current]):
            raise EvidenceError('inventory read progress changed')
        previous=count;consumed=completed+count
        now=time.monotonic()
        # Throttle before _emit's path checks and durable write: the activity
        # directory may itself live on network storage. No per-chunk stat walk.
        if last is None or now-last>=30:
            runtime_activity._emit({'schema':'ovl.runtime-inventory-read.v1','pass_index':index,
                'manifest':manifest,'stream_sha256':root,'read_bytes':consumed},clock=time.monotonic)
            last=now
        if verified:completed+=count;current+=1;previous=0
    return progress
