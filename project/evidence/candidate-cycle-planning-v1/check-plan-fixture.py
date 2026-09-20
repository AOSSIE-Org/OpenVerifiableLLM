from pathlib import Path
from decimal import Decimal,ROUND_CEILING
from datetime import datetime,timezone
import copy,importlib.util,os,shutil,sys
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest
from ovl_pipeline.supervision import rental_plan
from pod_fetch_prepared import FILES,selected
out=Path('.ovllm-cache/candidate-cycle-dry-plan-v2');out.mkdir(exist_ok=False)
base=Path('.ovllm-cache/sustained-candidate-v1');shutil.copytree(base/'inputs',out/'inputs',copy_function=os.link)
shutil.copyfile(base/'preliminary-selection.json',out/'preliminary-selection.json')
public=read_json(Path('.ovllm-cache/full-prepared-publication-v1/plan.json'))
prepared={'schema':'ovl.public-prepared-inputs.v1','repo':public['repo'],'revision':'0'*40,
          'preparation_sha256':public['subject_sha256'],'files':[f for f in public['files'] if f['path'] in FILES]}
write_json(out/'inputs/prepared-plan.json',prepared);selected(out/'inputs/prepared-plan.json',digest(prepared))
rental=copy.deepcopy(read_json(Path('.ovllm-cache/live-tiny-cuda-v5/rental-intent.json')))
q=rental['quote'];p=rental['watchdog_intent']['plan']['input'];now=p['now_epoch']
p.update(attempt_id='ovllm-dry-candidate-cycle-'+'0'*32,spent_usd='0.706754',outstanding_usd='3.025030',
         reserved_remaining_usd='83',allowance_usd='3',maximum_seconds=10800,checkpoint_grace_seconds=1500)
p['hourly_upper_usd']=format(((Decimal('.74')+Decimal(80)*Decimal('.10')/672)*Decimal('1.25')).quantize(Decimal('.000001'),rounding=ROUND_CEILING),'f')
p['quote_sha256']=digest(q);rp=rental_plan(p);rental['watchdog_intent']['plan']=rp
for payload in [rental['payload'],rental['watchdog_intent']['payload']]:
    payload.update(name=p['attempt_id'],containerDiskInGb=80,
                   terminateAfter=datetime.fromtimestamp(rp['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))
profile=read_json(Path('.ovllm-cache/live-tiny-cuda-v5/profile.json'))
profile.update(pod_id='dry-fixture-not-created',remote_root='/workspace/ovllm/candidate-cycle-v1',host='127.0.0.1')
spec=importlib.util.spec_from_file_location('selected_local_builder','.ovllm-cache/select_candidate_cycle_v1.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
plan=module.select(out,profile,rental,prepared)
write_json(out/'dry-validation.json',{'schema':'ovl.candidate-cycle-plan-dry-check.v1','result':'STATIC_PLAN_VALIDATION_PASS',
    'plan_sha256':digest(plan),'fixture_public_revision':'0'*40,'fixture_pod':'dry-fixture-not-created','network_or_gpu_work':'NOT_RUN',
    'work_and_export_seconds':sum(s['work_seconds']+s['export_reserve_seconds'] for s in plan['stages']),
    'rental_work_window_seconds':rp['request_checkpoint_epoch']-now,'maximum_rental_charge_micro_usd':rp['maximum_charge_micro_usd'],
    'scope':'synthetic expired rental/profile placeholders; not a live price quote, public-input verification, rental admission or performance measurement'})
print(read_json(out/'dry-validation.json'))
