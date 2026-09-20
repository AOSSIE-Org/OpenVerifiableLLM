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
root=Path.cwd();out=root/'.ovllm-cache/live-tiny-cuda-v1'
raw=(root/'project/evidence/provider-survey/gpu-catalog-live-tiny-cuda-v1.json').read_text()
cat=json.loads(raw);gpu=next(g for g in cat['gpus'] if g['id']=='NVIDIA GeForce RTX 4090');assert gpu['availability']!='NONE' and gpu['price']['secure']==.74
storage=read_json(root/'project/evidence/provider-survey/storage-pricing-20260918T0357.json',canonical_required=False)
# Retained MCP observation time, not preparation time.
observed=int((root/'project/evidence/provider-survey/gpu-catalog-live-tiny-cuda-v1.json').stat().st_mtime)
baseline=account();now=int(time.time());assert now-observed<=600
quote={'schema':'ovl.rental-quote.v1','observed_epoch':observed,'catalog_response':raw,'catalog_response_sha256':file_hash(root/'project/evidence/provider-survey/gpu-catalog-live-tiny-cuda-v1.json'),'selected_gpu':{'id':gpu['id'],'secure':True,'secure_hourly_usd':'0.74'},'storage_source':'https://docs.runpod.io/pods/pricing','storage_page_sha256':storage['sha256'],'storage_observed_epoch':storage['observed_epoch'],'container_gb_month_usd':'0.10','volume_gb_month_upper_usd':'0.20','monthly_hours':672,'rate_margin_percent':125}
rate=format(((Decimal('.74')+Decimal(40)*Decimal('.10')/672)*Decimal('1.25')).quantize(Decimal('.000001'),rounding=ROUND_CEILING),'f')
plan=rental_plan({'schema':'ovl.rental-budget-input.v2','attempt_id':'ovllm-tiny-cuda-'+uuid.uuid4().hex,'now_epoch':now,'spent_usd':'0.068185','outstanding_usd':'0.833792','reserved_remaining_usd':'87','allowance_usd':'1.25','hourly_upper_usd':rate,'quote_sha256':digest(quote),'maximum_seconds':1800,'checkpoint_grace_seconds':420,'billing_slack_seconds':300,'external_termination_grace_seconds':120,'authorization_sha256':AUTHORIZATION_SHA256})
payload={'name':plan['input']['attempt_id'],'gpuCount':1,'imageName':'runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851','containerDiskInGb':40,'volumeInGb':0,'terminateAfter':datetime.fromtimestamp(plan['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}
w={'schema':'ovl.external-watchdog-intent.v1','plan':plan,'creation_latest_epoch':now+120,'payload':payload,'baseline':baseline}
rental={'schema':'ovl.rental-controller-intent.v2','watchdog_intent':w,'quote':quote,'payload':{**payload,'cloudType':'SECURE','gpuTypeId':gpu['id'],'minVcpuCount':4,'minMemoryInGb':16,'dockerArgs':'','startSsh':True,'startJupyter':False,'ports':'22/tcp','allowedCudaVersions':['13.0','13.2']}}
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
 p=units/f'ovllm-live-tiny-cuda-v1-{name}.service';assert not p.exists();p.write_text(text);(out/(name+'.service')).write_text(text)
print(json.dumps({'name':payload['name'],'hourly_upper_usd':rate,'max_charge_micro_usd':plan['maximum_charge_micro_usd'],'deadline':payload['terminateAfter'],'creation_latest_epoch':w['creation_latest_epoch'],'intent_sha256':digest(rental)}))
