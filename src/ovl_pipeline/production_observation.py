"""Observe actual complete production scans without replacing their checks.

Execute each original function's exact code with a private globals mapping.
Only row iteration and its nested complete validator are observed. A row counts
after its consumer resumes; final success emits only after all checks return.
No source module globals, numerical state or preparation implementation changes.
"""
import os
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
import threading
import time
from types import FunctionType

from . import data,runtime_activity,schema
from .canonical import EvidenceError,digest,read_json,verify_inventory

_lock=threading.RLock()
_pass_index=0
_iterating=False
MAX_PASSES=16
_checked_streams=ContextVar('production_checked_streams',default=None)


def checked_stream_scope(function):
    """Reuse semantic checks only for freshly rehashed identical bytes in one call.

    No persisted/prover proof can populate this process-local cache. Every use
    still checks the complete manifest schema and every file's size/path/hash.
    Coverage and cursor enumerations remain separate, complete computations.
    The scope is discarded on success or failure, including recording resumes.
    This has the same immutable-input/host trust assumption as uncached memmaps;
    it does not protect against a hostile host changing bytes during a scan.
    """
    @wraps(function)
    def wrapped(*args,**kwargs):
        if _checked_streams.get() is not None:raise EvidenceError('nested checked stream scope refused')
        token=_checked_streams.set({})
        try:return function(*args,**kwargs)
        finally:_checked_streams.reset(token)
    return wrapped


def _scan(original,kind,directory,manifest,args):
    global _iterating
    if not os.environ.get('OVL_ACTIVITY_FILE'):
        if kind=='stream-validation' or _checked_streams.get() is None:return original(*args)
        checked=FunctionType(original.__code__,{**original.__globals__,'validate_stream':validate_stream},
                             original.__name__,original.__defaults__,original.__closure__)
        checked.__kwdefaults__=original.__kwdefaults__
        return checked(*args)
    if not _lock.acquire(blocking=False):raise EvidenceError('concurrent production scans refused')
    entered=False
    try:
        if _iterating:raise EvidenceError('nested production row scans refused')
        selected=Path(directory)/'documents.jsonl';count=0;index=None
        root=digest(manifest);original_rows=original.__globals__['rows']
        def emit(complete):
            runtime_activity._emit({'schema':'ovl.runtime-production-scan.v1','pass_index':index,
                'operation':kind,'stream_sha256':root,'documents':manifest['documents'],
                'completed_documents':count,'complete':complete},clock=time.monotonic,force=complete)
        def rows(path):
            nonlocal entered,count,index
            global _pass_index,_iterating
            if entered or Path(path)!=selected:raise EvidenceError('unexpected production scan iterator')
            entered=True;_iterating=True
            # Assign only when iteration actually starts: nested validation runs
            # before the outer census, so observations remain in execution order.
            if _pass_index>=MAX_PASSES:raise EvidenceError('production preflight pass limit exceeded')
            _pass_index+=1;index=_pass_index;emit(False);last=time.monotonic()
            for row in original_rows(path):
                yield row
                count+=1;now=time.monotonic()
                if now-last>=30:emit(False);last=now
        globals_copy={**original.__globals__,'rows':rows}
        if kind!='stream-validation':globals_copy['validate_stream']=validate_stream
        observed=FunctionType(original.__code__,globals_copy,original.__name__,original.__defaults__,original.__closure__)
        observed.__kwdefaults__=original.__kwdefaults__
        result=observed(*args)
        if not entered or count!=manifest['documents']:raise EvidenceError('production scan observation incomplete')
        emit(True)
        return result
    finally:
        if entered:_iterating=False
        _lock.release()


def validate_stream(directory,manifest):
    cache=_checked_streams.get()
    key=(str(Path(directory).absolute()),digest(manifest))
    if cache is not None and key in cache:
        schema.stream(manifest)
        # Never rely on mtime, inode, a prior hash receipt or caller assertion.
        verify_inventory(directory,manifest['files'])
        return cache[key]
    result=_scan(data.validate_stream,'stream-validation',directory,manifest,(directory,manifest))
    if cache is not None:
        if len(cache)>=2:raise EvidenceError('production validation scope exceeds two streams')
        # Bind completed semantic work to bytes still matching the same root.
        verify_inventory(directory,manifest['files'])
        cache[key]=result
    return result


def schedule_counts(directory,recipe):
    from .coverage import schedule_counts as original
    if not os.environ.get('OVL_ACTIVITY_FILE') and _checked_streams.get() is None:return original(directory,recipe)
    manifest=read_json(Path(directory)/'stream.json')
    return _scan(original,'coverage-census',directory,manifest,(directory,recipe))


def boundary_cursors(directory,recipe,expected_stream_sha256,steps):
    from .production_cursors import boundary_cursors as original
    if not os.environ.get('OVL_ACTIVITY_FILE') and _checked_streams.get() is None:return original(directory,recipe,expected_stream_sha256,steps)
    manifest=read_json(Path(directory)/'stream.json')
    return _scan(original,'boundary-cursor-census',directory,manifest,(directory,recipe,expected_stream_sha256,steps))
