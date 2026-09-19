#!/usr/bin/env python3
"""Development-only byte comparison of pinned historical serial and ordered workers.

Input is an explicitly identified small development XML prefix, not production
corpus evidence. No prefix is silently substituted for a complete preparation.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
import types

from ovl_pipeline.canonical import EvidenceError,file_hash,write_json,sha256
from ovl_pipeline.data import extract_wikipedia
from ovl_pipeline.preparation import preparation_code,preparation_environment

BASELINE='33f168612a71830da1443d445cc8d998b4c8f717'


def process_rss():
    # Sum RSS over coordinator + current descendants. Shared pages can be counted
    # more than once; polling can miss peaks. This is an observation, not an RSS cap.
    pending=[os.getpid()];seen=set();total=0
    while pending:
        pid=pending.pop()
        if pid in seen:continue
        seen.add(pid)
        try:
            p=Path('/proc')/str(pid)
            for line in (p/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1])*1024
            pending.extend(map(int,(p/'task'/str(pid)/'children').read_text().split()))
        except (FileNotFoundError,ProcessLookupError):pass
    return total


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',required=True,type=Path)
    p.add_argument('--expected-prefix-sha256',required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    if file_hash(a.prefix)!=a.expected_prefix_sha256:raise EvidenceError('development prefix identity differs')
    a.output.mkdir(parents=True,exist_ok=False)
    old=subprocess.check_output(['git','-C',str(root),'show',BASELINE+':src/ovl_pipeline/data.py'])
    # Explicit trusted source revision, not artifact-directed execution.
    legacy=types.ModuleType('ovl_pipeline._historical_serial_data')
    legacy.__package__='ovl_pipeline'
    exec(compile(old,BASELINE+':src/ovl_pipeline/data.py','exec'),legacy.__dict__)
    baseline_file=a.output/'historical-data.py';baseline_file.write_bytes(old)
    results=[]
    for name,fn in [('historical-serial',lambda d:legacy.extract_wikipedia([a.prefix],d)),
                    ('current-serial',lambda d:extract_wikipedia([a.prefix],d,workers=1)),
                    ('current-eight-workers',lambda d:extract_wikipedia([a.prefix],d,workers=8))]:
        stop=threading.Event();peak=[0];samples=[0]
        def monitor():
            while not stop.is_set():
                peak[0]=max(peak[0],process_rss());samples[0]+=1;stop.wait(.1)
        observer=threading.Thread(target=monitor);observer.start();start=time.monotonic_ns()
        try:manifest=fn(a.output/name)
        finally:stop.set();observer.join()
        results.append(dict(name=name,elapsed_ms=(time.monotonic_ns()-start)//1000000,
                            sampled_max_sum_rss_bytes=peak[0],rss_samples=samples[0],manifest=manifest))
    if not all(r['manifest']==results[0]['manifest'] for r in results):raise EvidenceError('manifest mismatch')
    for entry in results[0]['manifest']['files']:
        expected=(a.output/results[0]['name']/entry['path']).read_bytes()
        for r in results[1:]:
            if (a.output/r['name']/entry['path']).read_bytes()!=expected:raise EvidenceError('output bytes differ')
    value=dict(schema='ovl.extraction-comparison.v2',result='PASS',
        scope='development prefix only; no full-corpus preparation, reconstruction or production forecast credit',
        prefix_sha256=a.expected_prefix_sha256,baseline_revision=BASELINE,baseline_source_sha256=sha256(old),
        current_code=preparation_code(),environment=preparation_environment(),
        harness_sha256=file_hash(Path(__file__)),runs=results,
        worker_defaults=dict(workers=8,pending_limit=32,pending_raw_bytes_limit=64*1024*1024),
        host=dict(platform=platform.platform(),logical_cpus=os.cpu_count(),
            cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)),
        memory_scope='100ms sampled sum of parent/descendant RSS; shared pages double-counted; not guaranteed peak; includes dependencies and IPC',
        exact_file_and_manifest_equality=True)
    write_json(a.output/'comparison.json',value)
    print([(r['name'],r['elapsed_ms'],r['sampled_max_sum_rss_bytes']) for r in results])

if __name__=='__main__':main()
