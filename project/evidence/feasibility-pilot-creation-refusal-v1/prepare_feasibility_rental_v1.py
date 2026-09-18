from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal,ROUND_CEILING
import sys,time,uuid,json
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash
from ovl_pipeline.supervision import rental_plan
from probe_provider_deadline import account
from run_rental_controller import validate
from rental_quote import AUTHORIZATION_SHA256
root=Path.cwd();out=root/'.ovllm-cache/sustained-feasibility-v1'
# Original uncertain creation must be closed by BOTH existing owners before a
# fresh intent is prepared. This script never shortens or stops their deadlines.
old=root/'.ovllm-cache/sustained-candidate-v3'
for owner in ('controller','watchdog'):
 result=read_json(old/owner/'result.json')
 assert result['complete'] is True and result['pod_id'] is None and not result['residual_network_volumes']
 expected=digest(read_json(old/('rental-intent.json' if owner=='controller' else 'watchdog-intent.json')))
 assert result['intent_sha256']==expected
 state=__import__('subprocess').run(['systemctl','--user','is-active','ovllm-sustained-candidate-v3-'+owner+'.service'],capture_output=True,text=True)
 assert state.returncode==3 and state.stdout.strip()=='inactive'
assert not(out/'rental-intent.json').exists() and not(out/'watchdog-intent.json').exists()

raw=(root/'project/evidence/provider-survey/gpu-catalog-feasibility-v1.json').read_text()
cat=json.loads(raw);gpu=next(g for g in cat['gpus'] if g['id']=='NVIDIA GeForce RTX 4090');assert gpu['secure'] is True and gpu['price']['secure']==.74
capacity=json.loads((root/'project/evidence/provider-survey/secure-cuda13-capacity-feasibility-v1.json').read_text())
selected_capacity=next(g for g in capacity['items'] if g['id']==gpu['id'])['cudaVersions']['13.0']
assert selected_capacity['stock'] in ('Low','Medium','High') and selected_capacity['pricePerHr']==gpu['price']['secure']
storage=read_json(root/'project/evidence/provider-survey/storage-pricing-20260918T0357.json',canonical_required=False)
# Retained MCP observation time, not preparation time.
observed=read_json(root/'project/evidence/provider-survey/gpu-catalog-feasibility-v1-observation.json',canonical_required=False)['observed_epoch']
baseline=account();now=int(time.time());assert now-observed<=600
quote={'schema':'ovl.rental-quote.v1','observed_epoch':observed,'catalog_response':raw,'catalog_response_sha256':file_hash(root/'project/evidence/provider-survey/gpu-catalog-feasibility-v1.json'),'selected_gpu':{'id':gpu['id'],'secure':True,'secure_hourly_usd':'0.74'},'storage_source':'https://docs.runpod.io/pods/pricing','storage_page_sha256':storage['sha256'],'storage_observed_epoch':storage['observed_epoch'],'container_gb_month_usd':'0.10','volume_gb_month_upper_usd':'0.20','monthly_hours':672,'rate_margin_percent':125}
rate=format(((Decimal('.74')+Decimal(80)*Decimal('.10')/672)*Decimal('1.25')).quantize(Decimal('.000001'),rounding=ROUND_CEILING),'f')
plan=rental_plan({'schema':'ovl.rental-budget-input.v2','attempt_id':'ovllm-feasibility-pilot-'+uuid.uuid4().hex,'now_epoch':now,'spent_usd':'0.847606','outstanding_usd':'6.935706','reserved_remaining_usd':'77.0','allowance_usd':'5.0','hourly_upper_usd':rate,'quote_sha256':digest(quote),'maximum_seconds':18000,'checkpoint_grace_seconds':1500,'billing_slack_seconds':300,'external_termination_grace_seconds':120,'authorization_sha256':AUTHORIZATION_SHA256})
assert plan['request_checkpoint_epoch']-now>=14913+300,'insufficient complete measured-phase planning window before any creation'
payload={'name':plan['input']['attempt_id'],'gpuCount':1,'imageName':'runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851','containerDiskInGb':80,'volumeInGb':0,'terminateAfter':datetime.fromtimestamp(plan['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
w={'schema':'ovl.external-watchdog-intent.v1','plan':plan,'creation_latest_epoch':now+120,'payload':payload,'baseline':baseline}
rental={'schema':'ovl.rental-controller-intent.v2','watchdog_intent':w,'quote':quote,'payload':{**payload,'cloudType':'SECURE','gpuTypeId':gpu['id'],'minVcpuCount':4,'minMemoryInGb':16,'dockerArgs':'','startSsh':True,'startJupyter':False,'ports':'22/tcp','allowedCudaVersions':['13.0']}}
validate(rental,digest(rental))
for n,v in [('watchdog-intent.json',w),('rental-intent.json',rental)]:
 p=out/n;assert not p.exists();write_json(p,v)
units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
for name,script,args in [('watchdog','run_external_watchdog.py',f'--intent {out}/watchdog-intent.json --intent-sha256 {digest(w)} --journal {out}/watchdog'),('controller','run_rental_controller.py',f'--intent {out}/rental-intent.json --intent-sha256 {digest(rental)} --journal {out}/controller --watchdog-heartbeat {out}/watchdog/heartbeat.json --workload-health {out}/health.json')]:
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
 p=units/f'ovllm-sustained-feasibility-v1-{name}.service';assert not p.exists();p.write_text(text);(out/(name+'.service')).write_text(text)
print(json.dumps({'name':payload['name'],'hourly_upper_usd':rate,'max_charge_micro_usd':plan['maximum_charge_micro_usd'],'deadline':payload['terminateAfter'],'creation_latest_epoch':w['creation_latest_epoch'],'intent_sha256':digest(rental)}))
