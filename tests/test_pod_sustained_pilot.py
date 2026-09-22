"""Real isolated bootstrap processes; explicit audit substitute, no CUDA claim."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


SCRIPT=Path(__file__).parents[1]/'scripts/pod_sustained_pilot.py'


def launch(tmp_path,*,change=None):
    helper=tmp_path/'audit.py'
    helper.write_text('''from pathlib import Path
import json,os,time
def audited(config,expected,inputs,runtime,output,module,arguments):
 time.sleep(.1)
 output.mkdir()
 (output/'called.json').write_text(json.dumps({'module':module,'arguments':arguments,'activity':os.environ.get('OVL_ACTIVITY_FILE')}))
 return {'test_only':'explicit audit substitute, no runtime or GPU acceptance'}
''')
    config=tmp_path/'config.json';config.write_text('{}')
    inputs=tmp_path/'inputs';inputs.mkdir();runtime=tmp_path/'runtime';runtime.mkdir();control=tmp_path/'control'
    sha=hashlib.sha256(helper.read_bytes()).hexdigest();deadline=int(time.time())+30
    if change=='helper':sha='f'*64
    elif change=='deadline':deadline=int(time.time())-1
    elif change=='measured-window':deadline=int(time.time())+2700
    elif change=='excess-window':deadline=int(time.time())+2702
    elif change=='existing':control.mkdir()
    args=['record','--seconds','600','--output',str(tmp_path/'numerical')]
    if change=='action':args[0]='setup'
    cmd=[sys.executable,'-I','-S',str(SCRIPT),'--setup-script',str(helper),'--setup-sha256',sha,
         '--config',str(config),'--config-sha256',hashlib.sha256(config.read_bytes()).hexdigest(),'--inputs',str(inputs),
         '--runtime',str(runtime),'--control',str(control),'--deadline',str(deadline),'--',*args]
    env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','OVL_ACTIVITY_FILE':str(control/'activity.json')}
    if change=='activity':env['OVL_ACTIVITY_FILE']=str(tmp_path/'foreign/activity.json')
    result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=20)
    return result,control,cmd,env


def test_single_selected_audit_receives_exact_action_and_activity_and_never_retries(tmp_path):
    result,control,cmd,env=launch(tmp_path)
    assert result.returncode==0,result.stderr
    called=json.loads((control/'audit/called.json').read_text())
    assert called['module']=='ovl_pipeline.gpu_pilot' and called['arguments'][:3]==['record','--seconds','600']
    assert called['activity']==env['OVL_ACTIVITY_FILE']
    again=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=20)
    assert again.returncode!=0 and json.loads((control/'audit/called.json').read_text())==called


@pytest.mark.parametrize('damage',['helper','deadline','excess-window','existing','activity','action'])
def test_foreign_expired_or_reused_job_refused_before_audit(tmp_path,damage):
    result,control,_,_=launch(tmp_path,change=damage)
    assert result.returncode!=0 and not (control/'audit/called.json').exists()


def test_measured_longer_window_keeps_one_audited_process(tmp_path):
    result,control,_,_=launch(tmp_path,change='measured-window')
    assert result.returncode==0,result.stderr
    assert (control/'audit/called.json').is_file()
