"""Opt-in operator timings, never numerical or verification evidence.

Wall categories are exclusive within a phase. CUDA event spans overlap their
wall category and include device idle time awaiting host dispatch; they are NOT
kernel-busy counters. No cross-machine clock subtraction is supported.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import time

from .canonical import EvidenceError, digest
from .schema import fields, integer

_active = ContextVar('ovl_phase_timing', default=None)


class Collector:
    def __init__(self, *, clock=time.monotonic_ns):
        self.clock=clock;self.phase='setup';self.entries={};self.busy=False

    @contextmanager
    def activate(self):
        if _active.get() is not None:raise EvidenceError('nested profiling collector')
        token=_active.set(self)
        try:yield self
        finally:_active.reset(token)

    @contextmanager
    def measure(self, name, *, cuda=False):
        if self.busy:raise EvidenceError('overlapping wall timing categories')
        self.busy=True;start_event=end_event=None;started=self.clock()
        try:
            if cuda:
                import torch
                start_event=torch.cuda.Event(enable_timing=True)
                end_event=torch.cuda.Event(enable_timing=True)
                start_event.record()
            yield
            device_ns=0
            if cuda:
                end_event.record();end_event.synchronize()
                device_ns=round(start_event.elapsed_time(end_event)*1_000_000)
            elapsed=self.clock()-started
            if elapsed<0 or device_ns<0:raise EvidenceError('profiling clock regressed')
            entry=self.entries.setdefault((self.phase,name),{'calls':0,'wall_ns':0,'cuda_stream_span_ns':0,'cuda_calls':0})
            entry['calls']+=1;entry['wall_ns']+=elapsed
            entry['cuda_stream_span_ns']+=device_ns;entry['cuda_calls']+=int(cuda)
        finally:self.busy=False

    def report(self, parent, *, scope):
        return {'schema':'ovl.phase-timing.v1','parent_sha256':digest(parent),'scope':scope,
            'measurements':[{'phase':phase,'operation':name,**entry} for (phase,name),entry in sorted(self.entries.items())],
            'wall_categories':'exclusive; uninstrumented work remains in enclosing report time',
            'cuda_interpretation':'overlaps wall time; device stream interval includes host-dispatch idle time, not kernel-busy time',
            'cross_host_clock_comparison':False,'timing_overhead_in_enclosing_measurement':True,
            'verification_credit':False}

    def add_wall(self, name, elapsed):
        if type(elapsed) is not int or elapsed<0:raise EvidenceError('invalid measured wall interval')
        entry=self.entries.setdefault((self.phase,name),{'calls':0,'wall_ns':0,'cuda_stream_span_ns':0,'cuda_calls':0})
        entry['calls']+=1;entry['wall_ns']+=elapsed


def observe(name, *, cuda=False):
    current=_active.get()
    return nullcontext() if current is None else current.measure(name,cuda=cuda)


def phase(name):
    current=_active.get()
    if current is not None:
        if current.busy:raise EvidenceError('phase changed inside measurement')
        current.phase=name


def validate(value, parent, *, scope):
    fields(value,'schema parent_sha256 scope measurements wall_categories cuda_interpretation cross_host_clock_comparison timing_overhead_in_enclosing_measurement verification_credit','operator phase timing')
    empty=Collector().report(parent,scope=scope)
    if any(value[k]!=v for k,v in empty.items() if k!='measurements'):raise EvidenceError('timing parent or scope differs')
    if type(value['measurements']) is not list or len(value['measurements'])>128:raise EvidenceError('bounded timing categories required')
    seen=set()
    for item in value['measurements']:
        fields(item,'phase operation calls wall_ns cuda_stream_span_ns cuda_calls','timing category')
        for k in ('phase','operation'):
            if type(item[k]) is not str or not item[k] or len(item[k])>80:raise EvidenceError('invalid timing category')
        key=(item['phase'],item['operation'])
        if key in seen:raise EvidenceError('duplicate timing category')
        seen.add(key)
        integer(item['calls'],1,1_000_001,'timing calls')
        integer(item['cuda_calls'],0,item['calls'],'CUDA timing calls')
        for k in ('wall_ns','cuda_stream_span_ns'):integer(item[k],0,2**53-1,'timing duration')
        if not item['cuda_calls'] and item['cuda_stream_span_ns']:raise EvidenceError('CUDA timing without events')
    return value
