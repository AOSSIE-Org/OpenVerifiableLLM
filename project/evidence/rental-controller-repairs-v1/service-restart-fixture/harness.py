"""Real systemd restart, fake provider only. No provider requests or credentials."""
import os,sys,time,json
from pathlib import Path
from datetime import datetime,timezone
sys.path[:0]=['.ovllm-cache/rental-scale-v2/source/scripts','.ovllm-cache/rental-scale-v2/source/tests','.ovllm-cache/rental-scale-v2/source/src']
from test_external_watchdog import intent as template
from run_external_watchdog import run_guarded
from ovl_pipeline.supervision import rental_plan
from ovl_pipeline.canonical import write_json,read_json,digest
root=Path('.ovllm-cache/watchdog-service-fixture-v2');ip=root/'intent.json'
if not ip.exists():
    i=template();now=int(time.time());i['plan']['input']['now_epoch']=now;i['plan']=rental_plan(i['plan']['input'])
    i['creation_latest_epoch']=now+30;i['baseline']['observed_epoch']=now
    i['baseline']['http_clock']={k:now for k in ('server_epoch','request_started_epoch','request_completed_epoch')}
    i['payload']['terminateAfter']=datetime.fromtimestamp(i['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    write_json(ip,i)
i=read_json(ip)
def wall():
    f=root/'advance-to-deadline'
    return i['plan']['external_terminate_epoch']+time.time()-f.stat().st_mtime if f.exists() else time.time()
def pod():
    return {**{k:v for k,v in i['payload'].items() if k!='terminateAfter'},'id':'fixture-only-no-real-pod',
            'createdAt':datetime.fromtimestamp(i['plan']['input']['now_epoch'],timezone.utc).isoformat(),
            'costPerHr':'0.24','adjustedCostPerHr':'0.24'}
def account():
    now=int(wall());alive=not(root/'fake-terminated').exists()
    return {'observed_epoch':now,'pods':[pod()] if alive else [],'volume_ids':[],'autopay':False,
            'account_hourly_usd':'0.24' if alive else '0','http_clock':{k:now for k in ('server_epoch','request_started_epoch','request_completed_epoch')}}
def request(op,variables=None):
    assert op in ('terminate','identities')
    with (root/'fake-requests.jsonl').open('a') as f:f.write(json.dumps({'operation':op,'variables':variables,'pid':os.getpid(),'wall':wall(),'actual_epoch':time.time()})+'\n')
    if op=='identities':return {'myself':{'pods':account()['pods']}},'1'*64,{}
    assert variables=={'input':{'podId':'fixture-only-no-real-pod'}}
    (root/'fake-terminated').touch();return {},'2'*64,{}
with (root/'incarnations.jsonl').open('a') as f:f.write(json.dumps({'pid':os.getpid(),'actual_epoch':time.time()})+'\n')
run_guarded(root/'journal',i,digest(i),get_account=account,provider_request=request,wall=wall)
