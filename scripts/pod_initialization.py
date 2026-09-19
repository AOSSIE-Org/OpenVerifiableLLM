#!/usr/bin/env python3
"""One audited initialization record or regeneration process under the selected worker deadline.

The worker owns the absolute timeout. This wrapper adds no shorter numerical
timeout, retry, fallback or successor process. All runtime audits still execute.
The activity directory is separate from fresh numerical and audit outputs.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import time


def run(setup_script,setup_sha256,config,config_sha256,inputs,runtime,control,deadline,arguments):
    if not (sys.flags.isolated and sys.flags.no_site):raise ValueError('isolated no-site bootstrap required')
    paths=[Path(p).absolute() for p in (setup_script,config,inputs,runtime,control)]
    if any(p.is_symlink() for path in paths for p in [path,*path.parents]):raise ValueError('initialization bootstrap symlink')
    setup_script,config,inputs,runtime,control=paths
    if len(set(paths))!=len(paths):raise ValueError('distinct initialization bootstrap paths required')
    if type(deadline) is not int or not 0<deadline-time.time()<=1500:raise ValueError('original bounded worker deadline required')
    if control.exists():raise ValueError('fresh activity/audit control output required')
    if os.environ.get('OVL_ACTIVITY_FILE')!=str(control/'activity.json'):
        raise ValueError('explicit selected numerical activity path required')
    if not arguments or arguments[0] not in ('record','verify'):raise ValueError('one explicit initialization action required')
    # The selected setup helper independently rehashes all source and runtime
    # parents, constrains the numerical environment and checks fresh bytecode.
    if hashlib.sha256(setup_script.read_bytes()).hexdigest()!=setup_sha256:
        raise ValueError('audited setup source differs')
    control.mkdir(mode=0o700,parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('selected_initialization_audit',setup_script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.audited(config,config_sha256,inputs,runtime,control/'audit','ovl_pipeline.initialization',arguments)
    if time.time()>=deadline:raise TimeoutError('original pilot deadline expired')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('setup-script','config','inputs','runtime','control'):p.add_argument('--'+name,required=True,type=Path)
    for name in ('setup-sha256','config-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline',type=int,required=True)
    raw=sys.argv[1:];split=raw.index('--') if '--' in raw else len(raw)
    a=p.parse_args(raw[:split]);arguments=raw[split+1:]
    try:run(a.setup_script,a.setup_sha256,a.config,a.config_sha256,a.inputs,a.runtime,a.control,a.deadline,arguments)
    except Exception as error:p.exit(1,'initialization refused: '+type(error).__name__+'; preserve all partial outputs\n')


if __name__=='__main__':main()
