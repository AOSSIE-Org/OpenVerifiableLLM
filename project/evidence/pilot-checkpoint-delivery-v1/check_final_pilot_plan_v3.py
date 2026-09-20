"""Purely local final-pilot admission arithmetic; no creation or quote authority."""
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import copy,runpy,shutil,sys
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest,verify_inventory,EvidenceError,file_hash
from ovl_pipeline.supervision import rental_plan
from run_rental_controller import validate as validate_rental
from run_sustained_pilot import validate
base=Path('.ovllm-cache/final-pilot-plan-dry-v3');base.mkdir(exist_ok=False)
source=Path('.ovllm-cache/final-pilot-candidate-v2')
shutil.copytree(source/'inputs',base/'inputs')
shutil.copyfile(source/'preliminary-selection.json',base/'preliminary-selection.json')
verify_inventory(base/'inputs',read_json(source/'preliminary-input-inventory.json'))
r=read_json(Path('.ovllm-cache/sustained-feasibility-v2/rental-intent.json'))
v=copy.deepcopy(r['watchdog_intent']['plan']['input'])
v.update(attempt_id='ovllm-explicit-local-final-pilot-dry-'+'0'*32,maximum_seconds=18000,checkpoint_grace_seconds=1800,
         spent_usd='2.214095',outstanding_usd='15.187334',reserved_remaining_usd='67.0',allowance_usd='5.0')
p=rental_plan(v);r['watchdog_intent']['plan']=p
for payload in (r['payload'],r['watchdog_intent']['payload']):
 payload['name']=v['attempt_id'];payload['terminateAfter']=datetime.fromtimestamp(p['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
validate_rental(r,digest(r))
profile={'schema':'ovl.pod-ssh-profile.v1','pod_id':'explicit-local-final-dry','host':'127.0.0.1','port':22,'user':'root',
         'remote_root':'/workspace/ovllm/final-pilot-v1','endpoint_observation_sha256':'a'*64,'known_hosts_sha256':'b'*64,'host_key_trust':'operator-pinned-TOFU'}
select=runpy.run_path('.ovllm-cache/select_final_pilot_v1.py')['select']
plan=select(base,profile,r,read_json(base/'inputs/prepared-plan.json'))
assert [s['name'] for s in plan['stages']]==['setup','prepared-inputs','wikipedia-record','wikipedia-replay','conversation-record','conversation-replay']
negatives=[]
for damage in ('missing-input','changed-parent','insufficient-export','insufficient-copy','renewed-deadline','too-short-rental'):
 bad=copy.deepcopy(plan);rental=copy.deepcopy(r);st=bad['stages'][-1];job=read_json(base/'inputs'/st['template_path'])
 if damage=='missing-input':job['required_files']=[]
 elif damage=='changed-parent':st['retention']['binding']={**st['retention']['binding'],'code_root':'f'*64}
 elif damage=='insufficient-export':st['export_reserve_seconds']=1500
 elif damage=='insufficient-copy':st['retention']['copy_timeout_seconds']=300;job['argv'][job['argv'].index('--delivery-timeout')+1]='300'
 elif damage=='renewed-deadline':job['argv'][job['argv'].index('--delivery-deadline')+1]='9999999999'
 else:
  inp=copy.deepcopy(v);inp['maximum_seconds']=17000;short=rental_plan(inp);rental['watchdog_intent']['plan']=short
  for payload in (rental['payload'],rental['watchdog_intent']['payload']):payload['terminateAfter']=datetime.fromtimestamp(short['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
  bad['rental_intent_sha256']=digest(rental)
 st['template_path']='negative-'+damage+'.json';write_json(base/'inputs'/st['template_path'],job);st['template_sha256']=digest(job)
 try:validate(bad,digest(bad),rental,SimpleNamespace(profile=profile),base/'inputs',Path('scripts/pod_job_worker.py'))
 except EvidenceError as e:negatives.append({'case':damage,'result':'REJECTED','reason':str(e)})
 else:raise AssertionError('accepted mutation '+damage)
minimum=sum(s['work_seconds']+s['export_reserve_seconds'] for s in plan['stages'])+(sum(u['bytes'] for u in plan['uploads'])+1024**2-1)//1024**2
result={'schema':'ovl.final-pilot-plan-check.v1','result':'PASS','scope':'Purely local actual validator/arithmetic with historical quote and explicit fake endpoint; NOT provisioning or production admission',
        'minimum_seconds':minimum,'rental_work_window_seconds':p['request_checkpoint_epoch']-v['now_epoch'],'maximum_charge_micro_usd':p['maximum_charge_micro_usd'],
        'selected_reallocation':{'remaining_mandatory_before_usd':'72','pilot_allowance_usd':'5','remaining_mandatory_after_usd':'67','scope':'Candidate reallocation within original aggregate cap; actual measured full-run forecast still mandatory before production.'},
        'plan_sha256':digest(plan),'stages':[{'name':s['name'],'work_seconds':s['work_seconds'],'export_reserve_seconds':s['export_reserve_seconds'],'maximum_export_bytes':s['maximum_export_bytes']} for s in plan['stages']],
        'checks':negatives,'builder_sha256':file_hash(Path('.ovllm-cache/select_final_pilot_v1.py'))}
write_json(base/'verification.json',result);write_json(base/'explicit-dry-rental.json',r);write_json(base/'explicit-dry-profile.json',profile)
print(result)
