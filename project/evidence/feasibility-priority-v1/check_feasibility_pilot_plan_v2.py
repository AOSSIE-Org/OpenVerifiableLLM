from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import copy,runpy,shutil,sys,json
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest,inventory,verify_inventory,EvidenceError,file_hash
from ovl_pipeline.supervision import rental_plan
from run_rental_controller import validate as validate_rental
from run_sustained_pilot import validate
base=Path('.ovllm-cache/feasibility-plan-dry-v2');base.mkdir(exist_ok=False)
prior=Path('.ovllm-cache/sustained-candidate-v3')
shutil.copytree(prior/'inputs',base/'inputs')
shutil.copy2(prior/'preliminary-selection.json',base/'preliminary-selection.json')
verify_inventory(base/'inputs',read_json(prior/'input-inventory.json'))
r=read_json(prior/'rental-intent.json');v=copy.deepcopy(r['watchdog_intent']['plan']['input'])
v.update(attempt_id='ovllm-explicit-local-feasibility-dry-run-'+'0'*32,maximum_seconds=18000,spent_usd='0.847606',outstanding_usd='6.935706',reserved_remaining_usd='77.0',allowance_usd='5.0')
p=rental_plan(v);w=r['watchdog_intent'];w['plan']=p
for payload in [w['payload'],r['payload']]:
 payload['name']=v['attempt_id'];payload['terminateAfter']=datetime.fromtimestamp(p['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
validate_rental(r,digest(r))
profile={'schema':'ovl.pod-ssh-profile.v1','pod_id':'explicit-local-dry-run','host':'127.0.0.1','port':22,'user':'root','remote_root':'/workspace/ovllm/feasibility-pilot-v1','endpoint_observation_sha256':'a'*64,'known_hosts_sha256':'b'*64,'host_key_trust':'operator-pinned-TOFU'}
select=runpy.run_path('.ovllm-cache/select_feasibility_pilot_v1.py')['select']
plan=select(base,profile,r,read_json(base/'inputs/prepared-plan.json'))
assert [s['name'] for s in plan['stages']]==['setup','prepared-inputs','record','replay','resume','wiki-timed-record','wiki-timed-replay']
assert len(plan['stages'])==7
for s in plan['stages'][2:]:
 job=read_json(base/'inputs'/s['template_path']);assert s['work_seconds']==1500
 assert all(any(x['path'].endswith('/prepared/'+f['path']) and x['sha256']==f['sha256'] for x in job['required_files']) for f in read_json(base/'inputs/prepared-plan.json')['files'])
 if s['name']=='wiki-timed-record':
  assert job['argv'][job['argv'].index('--seconds')+1]=='600' and '--updates' not in job['argv']
 if s['name']=='wiki-timed-replay':assert '--resume-from' not in job['argv'] and s['parent_stage']=='wiki-timed-record'
negative=[]
for damage in ('missing-input-check','missing-full-replay-parent','insufficient-export','insufficient-total-time'):
 chosen=copy.deepcopy(plan);rental=copy.deepcopy(r)
 if damage=='missing-input-check':
  st=chosen['stages'][-1];job=read_json(base/'inputs'/st['template_path']);job['required_files']=[]
  st['template_path']='damaged-template.json';write_json(base/'inputs'/st['template_path'],job);st['template_sha256']=digest(job)
 elif damage=='missing-full-replay-parent':chosen['stages'][-1]['parent_stage']=None
 elif damage=='insufficient-export':chosen['stages'][-1]['export_reserve_seconds']=60
 else:
  inp=copy.deepcopy(v);inp['maximum_seconds']=10800;rp=rental_plan(inp);rental['watchdog_intent']['plan']=rp
  for payload in (rental['payload'],rental['watchdog_intent']['payload']):payload['terminateAfter']=datetime.fromtimestamp(rp['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
  chosen['rental_intent_sha256']=digest(rental)
 try:validate(chosen,digest(chosen),rental,SimpleNamespace(profile=profile),base/'inputs',Path('scripts/pod_job_worker.py'))
 except EvidenceError as e:negative.append({'case':damage,'result':'REJECTED','reason':str(e)})
 else:raise AssertionError('counterexample accepted: '+damage)
minimum=sum(s['work_seconds']+s['export_reserve_seconds'] for s in plan['stages'])+(sum(u['bytes'] for u in plan['uploads'])+1024**2-1)//1024**2
result={'schema':'ovl.feasibility-pilot-plan-check.v1','result':'PASS','scope':'local plan arithmetic and real validator only; historical quote and explicit fake endpoint; NO provisioning authority','stages':[{'name':s['name'],'work_seconds':s['work_seconds'],'export_reserve_seconds':s['export_reserve_seconds'],'maximum_export_bytes':s['maximum_export_bytes']} for s in plan['stages']],'minimum_seconds':minimum,'selected_work_window_seconds':p['request_checkpoint_epoch']-v['now_epoch'],'maximum_charge_micro_usd':p['maximum_charge_micro_usd'],'hypothetical_allocation_micro_usd':847606+6935706+77000000+p['maximum_charge_micro_usd'],'checks':negative,'builder_sha256':file_hash(Path('.ovllm-cache/select_feasibility_pilot_v1.py')),'production':'NOT_RUN','forecast':'NOT_MEASURED'}
write_json(base/'verification.json',result);write_json(base/'explicit-dry-rental.json',r);write_json(base/'explicit-dry-profile.json',profile)
print(json.dumps(result,indent=2))
