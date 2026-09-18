"""Observe completed work in the unchanged complete stream validator.

The function body and all its checks remain data.validate_stream's exact code.
A private globals mapping replaces only its rows iterator. No data module globals
are patched. The selected pilot telemetry permits one observed validator at a time;
ordinary preparation without the activity environment still calls the original.
Generator continuation runs only after the caller has checked the yielded row.
Even a complete row prefix is not final validation success: final roots/counts
still have to pass before the distinct completion observation is emitted.
"""
import os
import time
import threading
from pathlib import Path
from types import FunctionType

from . import data, runtime_activity
from .canonical import EvidenceError,digest

_observing=threading.Lock()


def validate_stream(directory,manifest):
    original=data.validate_stream
    if not os.environ.get('OVL_ACTIVITY_FILE'):
        return original(directory,manifest)
    if not _observing.acquire(blocking=False):
        raise EvidenceError('only one observed stream validator may run per process')
    try:return _observed(directory,manifest,original)
    finally:_observing.release()


def _observed(directory,manifest,original):
    selected=Path(directory)/'documents.jsonl'
    stream_root=digest(manifest)
    original_rows=data.rows
    count=0
    entered=False

    def observed_rows(path):
        nonlocal count,entered
        if entered or Path(path)!=selected:
            raise EvidenceError('unexpected stream validation iterator')
        entered=True
        # validate_stream verifies every selected file before entering rows.
        runtime_activity.stream_validation(stream_root,manifest['documents'],0,False)
        last=time.monotonic()
        for row in original_rows(path):
            yield row
            count+=1
            now=time.monotonic()
            if now-last>=30:
                runtime_activity.stream_validation(stream_root,manifest['documents'],count,False)
                last=now

    observed=FunctionType(original.__code__,{**original.__globals__,'rows':observed_rows},
                          original.__name__,original.__defaults__,original.__closure__)
    observed.__kwdefaults__=original.__kwdefaults__
    result=observed(directory,manifest)
    if not entered or count!=manifest['documents']:
        raise EvidenceError('validation iterator observation incomplete')
    runtime_activity.stream_validation(stream_root,manifest['documents'],count,True)
    return result
