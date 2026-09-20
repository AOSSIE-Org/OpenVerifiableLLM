"""Bounded ordered pure-text workers; execution parallelism cannot reorder pages."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import mwparserfromhell
import sys
from .canonical import EvidenceError


def configure_worker(use_c, recursion_limit):
    if mwparserfromhell.parser.use_c != use_c:
        raise EvidenceError('spawned wikitext parser backend differs from parent')
    sys.setrecursionlimit(recursion_limit)


def strip_text(raw):
    # References are citations rather than article prose; no network expansion.
    code=mwparserfromhell.parse(raw)
    for tag in list(code.filter_tags(recursive=True)):
        if str(tag.tag).lower() in {'ref','references','gallery'}:
            try:code.remove(tag)
            except ValueError:pass
    return code.strip_code(normalize=True,collapse=False)


class OrderedExtraction:
    """At most32 pages /64MiB raw input pending (or one oversized page).

    Total process memory is larger: parser trees, IPC copies and results are not
    included. Tiny inputs stay serial. Only parent writes output.

    Source identity, duplicates, ordering, inclusion and byte roots stay in the
    parent. Workers receive raw strings only, never source paths or output handles.
    No completion order, worker identity or runtime enters deterministic artifacts.
    """
    def __init__(self,emit,*,workers=8,pending_limit=32,pending_bytes_limit=64*1024*1024):
        if type(workers) is not int or not 1<=workers<=8:raise EvidenceError('invalid extraction worker count')
        if type(pending_limit) is not int or not 1<=pending_limit<=32:raise EvidenceError('invalid pending-page bound')
        if type(pending_bytes_limit) is not int or not 1<=pending_bytes_limit<=64*1024*1024:raise EvidenceError('invalid pending byte bound')
        self.pending_bytes_limit=pending_bytes_limit;self.pending_bytes=0
        self.emit=emit;self.workers=workers;self.limit=pending_limit;self.pending=deque();self.pool=None
        self.active=None

    def observation(self):
        def identity(record):return {k:record[k] for k in ('source_order','ordinal','page_id','revision_id') if k in record}
        return {'workers':self.workers,'pending_limit':self.limit,'pending_bytes_limit':self.pending_bytes_limit,
                'pool_created':self.pool is not None,'pending_raw_bytes':self.pending_bytes,
                'pending_records':[identity(r) for r,_,_,_ in self.pending],
                'active_record':identity(self.active) if self.active is not None else None}

    def __enter__(self):return self

    def submit(self,record,reason,raw):
        size=len(raw.encode('utf-8')) if reason is None else 0
        while self.pending and self.pending_bytes+size>self.pending_bytes_limit:self._one()
        # One oversized source page may still be transformed; never silently drop
        # corpus content to satisfy an execution-memory optimization.
        self.pending_bytes+=size
        if self.pool is None:
            self.pending.append((record,reason,raw,size))
            if len(self.pending)<self.limit:return
            if self.workers==1:
                self._one()
                return
            self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=get_context('spawn'),
                initializer=configure_worker,initargs=(mwparserfromhell.parser.use_c,sys.getrecursionlimit()))
            self.pending=deque((r,why,self.pool.submit(strip_text,text) if why is None else None,size)
                               for r,why,text,size in self.pending)
        else:self.pending.append((record,reason,self.pool.submit(strip_text,raw) if reason is None else None,size))
        if len(self.pending)>=self.limit:self._one()

    def _one(self):
        r,why,work,size=self.pending.popleft()
        self.active=r
        self.pending_bytes-=size
        text=(work.result() if self.pool is not None else strip_text(work)) if why is None else None
        self.emit(r,why,text)
        self.active=None

    def __exit__(self,kind,value,traceback):
        try:
            if kind is None:
                while self.pending:self._one()
        finally:
            if self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=True)
