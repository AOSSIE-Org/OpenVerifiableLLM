#!/usr/bin/env python3
"""Bounded operational measurements only; never training or CUDA replay evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def run(source,output,expected_gpu,*,execute=subprocess.run):
    source=Path(source);output=Path(output)
    if source.is_symlink() or not source.is_file() or source.stat().st_size!=64*1024**2:
        raise ValueError('exact regular 64 MiB diagnostic input required')
    output.mkdir(parents=True,exist_ok=False)
    started=int(time.time())
    result=execute(['/usr/bin/nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id',
                    '--format=csv,noheader,nounits'],capture_output=True,timeout=30,check=True,env={'PATH':'/usr/bin:/bin','LANG':'C'})
    text=result.stdout.decode();rows=text.strip().splitlines()
    if len(rows)!=1 or rows[0].split(',')[0].strip()!=expected_gpu:raise ValueError('observed device differs from one selected GPU')
    (output/'gpu.csv').write_text(text)
    destination=output/'roundtrip.bin';t=time.monotonic_ns()
    with source.open('rb') as f,destination.open('xb') as out:
        shutil.copyfileobj(f,out,1024**2);out.flush();os.fsync(out.fileno())
    copy_ns=time.monotonic_ns()-t
    roots=[];t=time.monotonic_ns()
    for _ in range(16):
        h=hashlib.sha256()
        with destination.open('rb') as f:
            for b in iter(lambda:f.read(1024**2),b''):h.update(b)
        roots.append(h.hexdigest())
    hash_ns=time.monotonic_ns()-t
    if len(set(roots))!=1:raise ValueError('diagnostic bytes changed')
    report={'schema':'ovl.pod-operational-diagnostic.v1','result':'OBSERVED_NOT_ADMITTED','gpu_observation':text,
            'payload_bytes':destination.stat().st_size,'payload_sha256':roots[0],'copy_fsync_nanoseconds':copy_ns,
            'hash_read_bytes':16*destination.stat().st_size,'hash_nanoseconds':hash_ns,
            'free_bytes':shutil.disk_usage(output).free,'started_epoch':started,'finished_epoch':int(time.time()),
            'bootstrap_executable_sha256':hashlib.sha256(Path('/proc/self/exe').read_bytes()).hexdigest(),
            'scope':'operator-observed device and local file timings; cached hash reads, not cold disk, CUDA computation, hardware attestation or full workload timing',
            'production_admission':'NOT_RUN'}
    (output/'diagnostic.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','expected-gpu'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.source,a.output,a.expected_gpu)


if __name__=='__main__':main()
