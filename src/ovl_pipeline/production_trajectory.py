"""Shared numerical trajectory primitive for record and continuous replay.

No publication or paid-execution authority is granted here. The enclosing driver
must authenticate registration/inputs first and, when recording, verify each
public primary-boundary anchor before advancing this iterator again. Recovery
boundaries never change the registered primary schedule or authorize new work.
"""
from dataclasses import dataclass

from . import gpu,initialization
from .canonical import EvidenceError,digest
from .data import batches
from .state import capture,state_root
from .training import _transition,code_root

@dataclass
class NumericalBoundary:
    kind:str
    model:object
    optimizer:object
    control:dict
    environment:dict


def _initial(registration,stream_directories):
    r=registration;recipe=r['recipe'];kernel=r['kernel']
    if r['code_root']!=code_root():raise EvidenceError('executing trajectory source differs from registered code')
    model,opt,control,environment,stream=initialization.fresh(stream_directories['wikipedia'],recipe,kernel,r['initialization']['warmup_updates'])
    if digest(environment['compatible'])!=r['runtime']['compatible_environment_sha256']:
        raise EvidenceError('trajectory compatible runtime differs from registration')
    if digest(stream)!=r['coverage']['wikipedia']['stream_sha256']:
        raise EvidenceError('regenerated initialization used wrong input stream')
    if state_root(*capture(model,opt,control))!=r['initialization']['state_sha256']:
        raise EvidenceError('regenerated full initial state differs from registration')
    return model,opt,control,environment,gpu.flags()


def _remaining(r,stream_directories,model,opt,control,environment,flags):
    recipe=r['recipe'];kernel=r['kernel']
    def event(kind):return NumericalBoundary(kind,model,opt,control.copy(),environment)
    for phase in ('wikipedia','conversation'):
        if phase=='wikipedia' and control['phase']=='conversation':continue
        if phase=='conversation' and control['phase']!='conversation':
            opt,control=_transition(model,recipe,control);yield event('transition')
        count=0;total=r['coverage'][phase]['targets'];opening=control['phase_step']
        for count,batch in enumerate(batches(stream_directories[phase],recipe['context'],recipe['batch_size']),1):
            if count<=opening:continue
            control=gpu.update(model,opt,batch,control,total,kernel,expected_flags=flags)
            final=control['cursor']==total
            if final or control['phase_step']%recipe['boundary_every']==0:
                yield event('final' if final and phase=='conversation' else 'base' if final else 'progress')
            elif control['phase_step']%r['recovery_every']==0:
                yield event('recovery')
        if count!=r['coverage'][phase]['updates'] or control['phase_step']!=count or control['cursor']!=total:
            raise EvidenceError('full trajectory phase coverage differs from registration')


def walk(registration,stream_directories):
    model,opt,control,environment,flags=_initial(registration,stream_directories)
    yield NumericalBoundary('initial',model,opt,control.copy(),environment)
    yield from _remaining(registration,stream_directories,model,opt,control,environment,flags)


def resume_record(registration,stream_directories,checkpoint_directory,boundary):
    """Recording-only continuation from an already authenticated public boundary.

    Caller must verify the full signed/anchored prefix, actual checkpoint bytes and
    full stream cursor census first. Continuous replay never calls this function.
    Fresh initialization/runtime is still regenerated/checked before restoration.
    """
    from .state import read_state,restore
    from .training import new_optimizer
    model,opt,control,environment,flags=_initial(registration,stream_directories)
    md,tensors=read_state(checkpoint_directory,boundary['checkpoint'])
    if boundary['control']['phase']=='conversation':opt=new_optimizer(model,registration['recipe'])
    control=restore(model,opt,md,tensors)
    if control!=boundary['control']:raise EvidenceError('restored recording control differs from authenticated boundary')
    yield from _remaining(registration,stream_directories,model,opt,control,environment,flags)
