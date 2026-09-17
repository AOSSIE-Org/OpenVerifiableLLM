"""GPU development probes: fresh initialization, exact replay, resume and timing.

Pilot cycles may repeat data solely for measurement. They are not a production
pass, and their weights must never initialize the production run.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import time

import torch

from . import gpu, schema
from .canonical import EvidenceError, canonical, confined, digest, read_json, require_digest, write_json
from .data import batches, validate_stream
from .state import capture, read_state, restore, save_state, state_root
from .training import code_root


def cycling_batches(directory, recipe):
    cycle = 0
    while True:
        seen = False
        for batch in batches(directory, recipe["context"], recipe["batch_size"]):
            seen = True
            yield cycle, batch
        if not seen:raise EvidenceError("empty pilot stream")
        cycle += 1


def advance(model, optimizer, control, cycle, batch, total, config, expected_flags, metrics=None):
    if cycle != control["pilot_cycle"]:
        if cycle != control["pilot_cycle"]+1 or control["cursor"] != total:
            raise EvidenceError("pilot cycle changed before complete target coverage")
        control = {**control,"pilot_cycle":cycle,"cursor":0}
    return gpu.update(model,optimizer,batch,control,total,config,expected_flags=expected_flags,metrics=metrics)


def initialize(directory, recipe, config, stream, warmup_updates):
    """Warm up on discarded weights, then regenerate all numerical/control/RNG state."""
    schema.integer(warmup_updates,1,1000,"warmup updates")
    model,opt,control=gpu.initialize(recipe,config)
    expected_flags=gpu.flags()
    control.update(phase=stream["phase"],pilot_cycle=0)
    it=cycling_batches(directory,recipe)
    for _ in range(warmup_updates):
        cycle,batch=next(it)
        control=advance(model,opt,control,cycle,batch,stream["targets"],config,expected_flags)
    torch.cuda.synchronize()
    # Observe libraries after forward/backward/AdamW loaded their numerical code.
    environment=gpu.environment(config)
    del model,opt,control,it,batch
    gc.collect()
    torch.cuda.empty_cache()
    model,opt,control=gpu.initialize(recipe,config)
    control.update(phase=stream["phase"],pilot_cycle=0)
    return model,opt,control,expected_flags,environment


def record(directory, recipe, config, output, *, updates=None, seconds=None, warmup_updates=4, checkpoint_every=128):
    if (updates is None)==(seconds is None):raise EvidenceError("select fixed updates or timed measurement")
    if updates is not None:schema.integer(updates,1,1_000_000,"pilot updates")
    if seconds is not None:schema.integer(seconds,600,3600,"pilot measurement seconds")
    schema.integer(checkpoint_every,1,1_000_000,"pilot checkpoint interval")
    if updates is not None and (updates+checkpoint_every-1)//checkpoint_every+1>4096:
        raise EvidenceError("pilot checkpoint schedule exceeds bounded format")
    schema.recipe(recipe,gpu=True);gpu.validate_config(config)
    if output.exists():raise EvidenceError("pilot output must be fresh; preserve failed attempts")
    setup_started=time.monotonic_ns()
    stream=read_json(confined(directory,"stream.json"));validate_stream(directory,stream)
    model,opt,control,expected_flags,environment=initialize(directory,recipe,config,stream,warmup_updates)
    output.mkdir(parents=True,exist_ok=False)
    settings={"schema":"ovl.gpu-pilot-settings.v1","scope":"development-gpu-pilot-only",
              "recipe":recipe,"kernel":config,"stream":stream,"code_root":code_root(),
              "environment":environment,"warmup_updates":warmup_updates,
              "requested_updates":updates,"requested_seconds":seconds,"checkpoint_every":checkpoint_every}
    write_json(output/"settings.json",settings)
    boundaries=[]
    def boundary():
        if len(boundaries)>=4096:raise EvidenceError("pilot checkpoint schedule exhausted; incomplete run preserved")
        path=f"boundary-{len(boundaries):05d}"
        checkpoint=save_state(output/path,model,opt,control)
        b={"index":len(boundaries),"step":control["global_step"],"control":control.copy(),
           "path":path,"checkpoint":checkpoint,"previous":digest(boundaries[-1]) if boundaries else digest(settings)}
        boundaries.append(b)
        write_json(output/"progress.json",{"settings_sha256":digest(settings),"boundaries":boundaries,"complete":False})
    boundary()
    torch.cuda.synchronize();setup_ns=time.monotonic_ns()-setup_started
    started=time.monotonic_ns();measured_targets=0;count=0;full_batches=0
    it=cycling_batches(directory,recipe)
    with (output/"updates.jsonl").open("wb") as log:
        while True:
            cycle,batch=next(it);metrics={}
            control=advance(model,opt,control,cycle,batch,stream["targets"],config,expected_flags,metrics)
            count+=1;measured_targets+=metrics["targets"]
            full_batches+=int(batch["inputs"].shape[0]==recipe["batch_size"])
            if count>1_000_000:raise EvidenceError("pilot update ceiling exceeded; incomplete run preserved")
            log.write(canonical({"step":count,"cycle":cycle,"cursor":control["cursor"],
                                 "transcript":control["transcript"],**metrics})+b"\n")
            if count%checkpoint_every==0:
                boundary();log.flush()
            if (updates is not None and count>=updates) or (seconds is not None and time.monotonic_ns()-started>=seconds*10**9):
                break
        if boundaries[-1]["step"]!=count:boundary()
        log.flush()
        os.fsync(log.fileno())
    torch.cuda.synchronize();elapsed_ns=time.monotonic_ns()-started
    result={"schema":"ovl.gpu-pilot-record.v1","scope":"development-gpu-pilot-only",
            "result":"RECORDED_NOT_REPLAYED","settings":settings,"boundaries":boundaries,
            "updates":count,"measured_full_batch_updates":full_batches,
            "timed_checkpoints":len(boundaries)-1,
            "measured_targets":measured_targets,"measured_ms":(elapsed_ns+999999)//1000000,
            "setup_including_warmup_ms":(setup_ns+999999)//1000000,
            "warmup_excluded":True,"overhead_included":True,
            "eligible_duration_for_forecast":seconds is not None and elapsed_ns>=600*10**9,
            "production_training_coverage":"NOT_RUN","production_admission":"NOT_RUN"}
    write_json(output/"record.json",result)
    return result


def replay(directory, record_directory, expected_record_sha256, output, *, resume_from=None):
    setup_started=time.monotonic_ns()
    require_digest(expected_record_sha256)
    value=read_json(confined(record_directory,"record.json"))
    if digest(value)!=expected_record_sha256:raise EvidenceError("pilot record differs from caller-selected digest")
    if value.get("schema")!="ovl.gpu-pilot-record.v1" or value.get("scope")!="development-gpu-pilot-only":
        raise EvidenceError("unsupported pilot record")
    if output.exists():raise EvidenceError("pilot verifier output must be fresh")
    settings=value["settings"];recipe=settings["recipe"];config=settings["kernel"];stream=settings["stream"]
    schema.recipe(recipe,gpu=True);gpu.validate_config(config);validate_stream(directory,stream)
    if settings["code_root"]!=code_root():raise EvidenceError("pilot code differs")
    schema.integer(value["updates"],1,1_000_000,"recorded pilot updates")
    interval=settings["checkpoint_every"];schema.integer(interval,1,1_000_000,"pilot checkpoint interval")
    boundaries=value["boundaries"]
    if not boundaries or len(boundaries)>4096:raise EvidenceError("missing/oversized pilot boundaries")
    if (value["updates"]+interval-1)//interval+1!=len(boundaries):raise EvidenceError("pilot boundary count differs from schedule")
    expected_steps=list(range(0,value["updates"]+1,interval))
    if expected_steps[-1]!=value["updates"]:expected_steps.append(value["updates"])
    previous=digest(settings)
    for i,b in enumerate(boundaries):
        schema.fields(b,"index step control path checkpoint previous","pilot boundary")
        if b["index"]!=i or b["path"]!=f"boundary-{i:05d}" or b["previous"]!=previous or b["step"]!=expected_steps[i]:
            raise EvidenceError("broken pilot boundary ancestry")
        c=b["control"].copy();cycle=c.pop("pilot_cycle");schema.control(c)
        schema.integer(cycle,0,value["updates"],"pilot cycle")
        if c["global_step"]!=b["step"] or c["phase"]!=stream["phase"] or c["cursor"]>stream["targets"]:
            raise EvidenceError("pilot boundary control differs from schedule/stream")
        previous=digest(b)
    if resume_from is not None:schema.integer(resume_from,1,len(boundaries)-2,"resume probe boundary")
    model,opt,control,expected_flags,environment=initialize(directory,recipe,config,stream,settings["warmup_updates"])
    if environment["compatible"]!=settings["environment"]["compatible"]:
        raise EvidenceError("pilot compatible environment differs")
    position=0;compared=[]
    def compare():
        nonlocal position
        b=boundaries[position]
        if b["control"]!=control:raise EvidenceError("pilot control/schedule mismatch")
        md,tensors=read_state(confined(record_directory,b["path"]),b["checkpoint"])
        actual=state_root(*capture(model,opt,control))
        if actual!=state_root(md,tensors):raise EvidenceError(f"pilot state mismatch at boundary {position}")
        compared.append({"index":position,"state_root":actual});position+=1
    compare()  # Always regenerate boundary zero; never load it as initialization.
    if resume_from is not None:
        b=boundaries[resume_from]
        md,tensors=read_state(confined(record_directory,b["path"]),b["checkpoint"])
        control=restore(model,opt,md,tensors)
        if control!=b["control"]:raise EvidenceError("resume checkpoint control mismatch")
        position=resume_from;compare()
    opening=control["global_step"]
    torch.cuda.synchronize();setup_ns=time.monotonic_ns()-setup_started
    before_compared=len(compared);started=time.monotonic_ns();targets=full_batches=0
    it=cycling_batches(directory,recipe)
    for index in range(value["updates"]):
        cycle,batch=next(it)
        if index<opening:continue
        metrics={}
        control=advance(model,opt,control,cycle,batch,stream["targets"],config,expected_flags,metrics)
        targets+=metrics["targets"];full_batches+=int(batch["inputs"].shape[0]==recipe["batch_size"])
        if position<len(boundaries) and control["global_step"]==boundaries[position]["step"]:compare()
    if position!=len(boundaries) or control!=boundaries[-1]["control"]:
        raise EvidenceError("pilot replay incomplete or extra boundaries")
    torch.cuda.synchronize();elapsed_ns=time.monotonic_ns()-started
    if resume_from is None and (targets!=value["measured_targets"] or full_batches!=value["measured_full_batch_updates"]
            or len(compared)-before_compared!=value["timed_checkpoints"]):
        raise EvidenceError("recomputed pilot work differs from recorded measurement")
    report={"schema":"ovl.gpu-pilot-replay.v1","result":"PASS","record_sha256":digest(value),
            "scope":"fresh-initialization-continuous-pilot-replay" if resume_from is None else "training-resume-continuation-probe",
            "updates_recomputed":value["updates"]-opening,"initial_state_regenerated":True,
            "resume_from":resume_from,"compared":compared,"environment":environment,
            "measured_targets":targets,"measured_full_batch_updates":full_batches,
            "timed_checkpoints":len(compared)-before_compared,"measured_ms":(elapsed_ns+999999)//1000000,
            "setup_including_warmup_ms":(setup_ns+999999)//1000000,
            "warmup_excluded":True,"overhead_included":True,
            "eligible_for_forecast_comparison":resume_from is None and value["eligible_duration_for_forecast"] is True,
            "performed_by":"project-operator","independent_third_party":False,
            "production_training_coverage":"NOT_RUN","production_admission":"NOT_RUN"}
    output.mkdir(parents=True,exist_ok=False);write_json(output/"verification.json",report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest="action",required=True)
    a=subs.add_parser("record");a.add_argument("--recipe",type=Path,required=True);a.add_argument("--kernel",type=Path,required=True)
    limits=a.add_mutually_exclusive_group(required=True);limits.add_argument("--updates",type=int);limits.add_argument("--seconds",type=int)
    a.add_argument("--warmup-updates",type=int,default=4);a.add_argument("--checkpoint-every",type=int,default=128)
    b=subs.add_parser("replay");b.add_argument("--record-directory",type=Path,required=True)
    b.add_argument("--expected-record-sha256",required=True);b.add_argument("--resume-from",type=int)
    for sub in (a,b):
        sub.add_argument("--stream",type=Path,required=True);sub.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    try:
        if args.action=="record":
            result=record(args.stream,read_json(args.recipe),read_json(args.kernel),args.output,
                          updates=args.updates,seconds=args.seconds,warmup_updates=args.warmup_updates,checkpoint_every=args.checkpoint_every)
        else:result=replay(args.stream,args.record_directory,args.expected_record_sha256,args.output,resume_from=args.resume_from)
        print(canonical({"result":result["result"],"report_sha256":digest(result),"output":str(args.output)}).decode())
    except Exception as e:
        print(canonical({"result":"FAIL","reason":str(e)}).decode());return 1
    return 0


if __name__=="__main__":raise SystemExit(main())
