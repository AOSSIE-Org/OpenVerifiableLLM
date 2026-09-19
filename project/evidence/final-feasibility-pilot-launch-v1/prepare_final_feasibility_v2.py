from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_CEILING
import sys,time,uuid,json,shutil
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash
from ovl_pipeline.supervision import rental_plan
from probe_provider_deadline import account
from run_rental_controller import validate
from rental_quote import AUTHORIZATION_SHA256
from pod_availability_gate import check as check_pod_availability
from ovl_pipeline.canonical import verify_inventory
root=Path.cwd();out=root/'.ovllm-cache/final-feasibility-v1'
# The completed prior pilot must be closed by BOTH existing owners before a
# fresh intent is prepared. This script never shortens or stops their deadlines.
old=root/'.ovllm-cache/sustained-feasibility-v2'
for owner in ('controller','watchdog'):
 result=read_json(old/owner/'result.json')
 assert result['complete'] is True and result['pod_id']=='ujy14dbnadmqye' and not result['residual_network_volumes']
 expected=digest(read_json(old/('rental-intent.json' if owner=='controller' else 'watchdog-intent.json')))
 assert result['intent_sha256']==expected
 state=__import__('subprocess').run(['systemctl','--user','is-active','ovllm-sustained-feasibility-v2-'+owner+'.service'],capture_output=True,text=True)
 assert state.returncode==3 and state.stdout.strip()=='inactive'
assert not(out/'rental-intent.json').exists() and not(out/'watchdog-intent.json').exists()

# Fresh observations are retained separately before this one-shot preparation.
observations=root/'.ovllm-cache/final-feasibility-v1-admission'
raw=(observations/'gpu-catalog.json').read_text()
cat=json.loads(raw);gpu=next(g for g in cat['gpus'] if g['id']=='NVIDIA GeForce RTX 4090')
observation=json.loads((observations/'pod-availability.json').read_text())
selection=check_pod_availability(observation,now=int(time.time()),gpu_id=gpu['id'],
    cuda_versions=['13.0','13.2'],maximum_hourly_usd='0.74')
assert Decimal(str(gpu['price']['secure']))==Decimal(selection['secure_hourly_usd'])
versions=sorted(selection['available_cuda_versions'],key=lambda v:int(v.split('.')[1]))
storage=read_json(root/'project/evidence/provider-survey/storage-pricing-20260919T0340.json',canonical_required=False)
observed=read_json(observations/'catalog-observation.json',canonical_required=False)['observed_epoch']
# This record must be authored from refreshed provider billing after old closure.
# Conservative ceilings remain reserved unless explicitly reconciled.
allocation=read_json(observations/'budget-allocation.json')
assert allocation['old_rental_intent_sha256']==digest(read_json(old/'rental-intent.json'))
assert allocation['outstanding_usd']=='15.187334' and allocation['spent_usd']=='2.214095'
assert allocation['reserved_remaining_usd']=='67.0' and allocation['allowance_usd']=='5.0'
assert file_hash(root/allocation['billing_evidence'])==allocation['billing_sha256']
assert 0<=int(time.time())-allocation['observed_epoch']<=600
baseline=account();now=int(time.time());assert 0<=now-observed<=600
assert not baseline['pods'] and not baseline['volume_ids'] and not baseline['autopay'] and Decimal(baseline['account_hourly_usd'])==0
# Keep the complete reconstruction remainder and24GiB pilot/index headroom,
# while retaining the established 20 percent local free-disk floor.
prepared_bytes=71294010506
reconstructed=sum(p.stat().st_size for p in (root/'.ovllm-cache/reconstruction-preparation-v1').rglob('*') if p.is_file())
disk=shutil.disk_usage(root)
assert disk.free-max(0,prepared_bytes-reconstructed)-24*1024**3>=disk.total//5
quote={'schema':'ovl.rental-quote.v1','observed_epoch':observed,'catalog_response':raw,'catalog_response_sha256':file_hash(observations/'gpu-catalog.json'),'selected_gpu':{'id':gpu['id'],'secure':True,'secure_hourly_usd':'0.74'},'storage_source':'https://docs.runpod.io/pods/pricing','storage_page_sha256':storage['sha256'],'storage_observed_epoch':storage['observed_epoch'],'container_gb_month_usd':'0.10','volume_gb_month_upper_usd':'0.20','monthly_hours':672,'rate_margin_percent':125}
rate=format(((Decimal('.74')+Decimal(80)*Decimal('.10')/672)*Decimal('1.25')).quantize(Decimal('.000001'),rounding=ROUND_CEILING),'f')
plan=rental_plan({'schema':'ovl.rental-budget-input.v2','attempt_id':'ovllm-final-feasibility-'+uuid.uuid4().hex,'now_epoch':now,'spent_usd':allocation['spent_usd'],'outstanding_usd':allocation['outstanding_usd'],'reserved_remaining_usd':allocation['reserved_remaining_usd'],'allowance_usd':'5.0','hourly_upper_usd':rate,'quote_sha256':digest(quote),'maximum_seconds':18000,'checkpoint_grace_seconds':1800,'billing_slack_seconds':300,'external_termination_grace_seconds':120,'authorization_sha256':AUTHORIZATION_SHA256})
assert plan['request_checkpoint_epoch']-now>=16113,'insufficient complete measured-phase planning window before any creation'
payload={'name':plan['input']['attempt_id'],'gpuCount':1,'imageName':'runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851','containerDiskInGb':80,'volumeInGb':0,'terminateAfter':datetime.fromtimestamp(plan['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
w={'schema':'ovl.external-watchdog-intent.v1','plan':plan,'creation_latest_epoch':now+120,'payload':payload,'baseline':baseline}
rental={'schema':'ovl.rental-controller-intent.v2','watchdog_intent':w,'quote':quote,'payload':{**payload,'cloudType':'SECURE','gpuTypeId':gpu['id'],'minVcpuCount':4,'minMemoryInGb':16,'dockerArgs':'','startSsh':True,'startJupyter':False,'ports':'22/tcp','allowedCudaVersions':versions}}
validate(rental,digest(rental))
assert not out.exists(), 'new attempt directory must be exclusive'
out.mkdir()
candidate=root/'.ovllm-cache/final-pilot-candidate-v2'
from ovl_pipeline.training import code_root
selected=read_json(candidate/'preliminary-selection.json')
assert selected['code_root']==code_root() and selected['recipe_sha256']=='a37ec1353bfae2211d2526ebf407531e70a131ae910f3daa58fbc5de1bb0724d'
assert __import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip()=='df7de83c8ec349d0e3237686397fc530151f298a'
ci=read_json(observations/'ci.json',canonical_required=False)
assert ci['headSha']=='df7de83c8ec349d0e3237686397fc530151f298a' and ci['conclusion']=='success'
assert read_json(root/'project/goal_state.json',canonical_required=False)['acceptance']['G02']['status']=='passed'
shutil.copytree(candidate/'inputs',out/'inputs')
shutil.copy2(candidate/'preliminary-selection.json',out/'preliminary-selection.json')
shutil.copy2(candidate/'preliminary-input-inventory.json',out/'input-inventory.json')
verify_inventory(out/'inputs',read_json(out/'input-inventory.json'))
write_json(out/'pod-availability-admission.json',{'observation_sha256':file_hash(observations/'pod-availability.json'),'selection':selection})
private=Path.home()/'.local/share/openverifiablellm/provider-responses'/payload['name']
assert not private.exists() and out not in private.parents
for n,v in [('watchdog-intent.json',w),('rental-intent.json',rental)]:
 p=out/n;assert not p.exists();write_json(p,v)
units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
for name,script,args in [('watchdog','run_external_watchdog.py',f'--intent {out}/watchdog-intent.json --intent-sha256 {digest(w)} --journal {out}/watchdog'),('controller','provider_request_receipts.py',f'--intent {out}/rental-intent.json --intent-sha256 {digest(rental)} --journal {out}/controller --watchdog-heartbeat {out}/watchdog/heartbeat.json --workload-health {out}/health.json --private-responses {private} --diagnostics {out}/provider-diagnostics')]:
 text=f'''[Unit]
Description=OpenVerifiableLLM bounded diagnostic {name}
StartLimitIntervalSec=0
[Service]
Type=simple
WorkingDirectory={root}
Environment=PYTHONPATH={root}/src:{root}/scripts
ExecStart={root}/.venv/bin/python {root}/scripts/{script} {args}
Restart=on-failure
RestartSec=5
MemoryMax=4G
StandardOutput=append:{out}/{name}.stdout
StandardError=append:{out}/{name}.stderr
'''
 p=units/f'ovllm-final-feasibility-v1-{name}.service';assert not p.exists();p.write_text(text);(out/(name+'.service')).write_text(text)
print(json.dumps({'name':payload['name'],'hourly_upper_usd':rate,'max_charge_micro_usd':plan['maximum_charge_micro_usd'],'deadline':payload['terminateAfter'],'creation_latest_epoch':w['creation_latest_epoch'],'intent_sha256':digest(rental)}))
