"""Full-stream checkpoint cursor census independent of the trainer's batches.

Every document/window mask is scanned; requested boundary positions only bound
output size. This is input/schedule verification, not performed training coverage.
"""
from pathlib import Path
import numpy as np
from . import schema
from .canonical import EvidenceError,digest,read_json,require_digest
from .data import rows,validate_stream


def boundary_cursors(directory:Path,recipe,expected_stream_sha256,steps):
    schema.recipe(recipe,gpu=True);require_digest(expected_stream_sha256)
    if type(steps) is not list or not 2<=len(steps)<=4096:raise EvidenceError('nonempty bounded opening/closing cursor schedule required')
    for step in steps:schema.integer(step,0,2**53-1,'requested boundary step')
    if steps!=sorted(set(steps)) or steps[0]!=0:raise EvidenceError('cursor schedule must start zero and increase')
    stream=read_json(directory/'stream.json')
    if digest(stream)!=expected_stream_sha256:raise EvidenceError('stream differs from external selection')
    validate_stream(directory,stream)
    masks=np.memmap(directory/'mask.u8',dtype=np.uint8,mode='r')
    context=recipe['context'];batch=recipe['batch_size'];position=1
    windows=targets=documents=0;result=[{'step':0,'windows':0,'targets':0}]
    for doc in rows(directory/'documents.jsonl'):
        documents+=1;mask=masks[doc['offset']:doc['offset']+doc['tokens']]
        full,tail=divmod(len(mask),context)
        counts=mask[:full*context].reshape(full,context).sum(axis=1) if full else np.empty(0,dtype=np.uint64)
        if tail:counts=np.append(counts,np.uint64(mask[full*context:].sum()))
        positive=counts[counts>0];prefix=positive.cumsum(dtype=np.uint64)
        closing_windows=windows+len(positive)
        while position<len(steps) and steps[position]*batch<=closing_windows:
            needed=steps[position]*batch
            result.append({'step':steps[position],'windows':needed,'targets':targets+int(prefix[needed-windows-1])})
            position+=1
        windows=closing_windows;targets+=int(counts.sum());
    updates=(windows+batch-1)//batch
    if not windows or targets!=stream['targets'] or documents!=stream['documents']:
        raise EvidenceError('full cursor census disagrees with stream totals')
    if steps[-1]!=updates:raise EvidenceError('cursor schedule must include exact full-stream final update')
    if position<len(steps):
        if position!=len(steps)-1 or updates*batch<=windows:raise EvidenceError('unresolved checkpoint cursor')
        result.append({'step':updates,'windows':windows,'targets':targets});position+=1
    if len(result)!=len(steps) or result[-1]['targets']!=targets:raise EvidenceError('incomplete checkpoint cursor census')
    return {'schema':'ovl.boundary-cursor-map.v1','scope':'complete-stream-target-prefix-census',
            'stream_sha256':expected_stream_sha256,'recipe_sha256':digest(recipe),'phase':stream['phase'],
            'documents':documents,'targets':targets,'target_bearing_windows':windows,'updates':updates,
            'boundaries':result,'training_coverage':'NOT_RUN'}
