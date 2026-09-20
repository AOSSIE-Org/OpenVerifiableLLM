"""Local-only preparation integration check. Never starts units or calls RunPod."""
from pathlib import Path
import sys,json,runpy,tempfile,shutil,subprocess,copy,time
from unittest.mock import patch
repo=Path.cwd();sys.path[:0]=[str(repo/'src'),str(repo/'scripts')]
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash
import probe_provider_deadline as provider
script=repo/'.ovllm-cache/prepare_feasibility_rental_v2.py'
report=repo/'.ovllm-cache/feasibility-v2-preparation-check.json';assert not report.exists()
# Actual still-open attempt must refuse before any API call or new output.
with patch.object(provider,'account',side_effect=AssertionError('API call forbidden')):
 try:runpy.run_path(str(script))
 except FileNotFoundError as e:
  assert str(repo/'.ovllm-cache/sustained-feasibility-v1/controller/result.json') in str(e)
 else:raise AssertionError('open attempt admitted')
assert not (repo/'.ovllm-cache/sustained-feasibility-v2').exists()
with tempfile.TemporaryDirectory(prefix='ovllm-explicit-offline-preparation-') as td:
 root=Path(td);home=root/'home';home.mkdir()
 old=root/'.ovllm-cache/sustained-feasibility-v1';old.mkdir(parents=True)
 actual=repo/'.ovllm-cache/sustained-feasibility-v1'
 for name in ('rental-intent.json','watchdog-intent.json','input-inventory.json','preliminary-selection.json','qualification-overlap-admission.json'):
  shutil.copy2(actual/name,old/name)
 shutil.copytree(actual/'inputs',old/'inputs')
 # Fake closure is confined to disposable fixture. No acceptance or retry credit.
 for owner in ('controller','watchdog'):
  (old/owner).mkdir()
  write_json(old/owner/'result.json',{'complete':True,'pod_id':None,'residual_network_volumes':[],
   'intent_sha256':digest(read_json(old/('rental-intent.json' if owner=='controller' else 'watchdog-intent.json')))})
 (root/'.ovllm-cache/reconstruction-preparation-v1').symlink_to(repo/'.ovllm-cache/reconstruction-preparation-v1',target_is_directory=True)
 survey=root/'project/evidence/provider-survey';survey.mkdir(parents=True)
 shutil.copy2(repo/'project/evidence/provider-survey/storage-pricing-20260918T0357.json',survey)
 obs=root/'.ovllm-cache/feasibility-v2-admission';obs.mkdir()
 shutil.copy2(repo/'project/evidence/provider-survey/gpu-catalog-feasibility-v1.json',obs/'gpu-catalog.json')
 availability=json.loads((repo/'project/evidence/provider-creation-diagnostics-v1/pod-availability-observation.json').read_text())
 now=availability['observed_epoch'];(obs/'pod-availability.json').write_text(json.dumps(availability))
 write_json(obs/'catalog-observation.json',{'observed_epoch':now})
 write_json(root/'explicit-fixture-billing.json',{'scope':'EXPLICIT LOCAL FIXTURE, NO PROVIDER BILLING CLAIM'})
 write_json(obs/'budget-allocation.json',{'old_rental_intent_sha256':digest(read_json(old/'rental-intent.json')),
  'outstanding_usd':'11.744764','spent_usd':'0.847606','reserved_remaining_usd':'72.0','allowance_usd':'5.0',
  'billing_evidence':'explicit-fixture-billing.json','billing_sha256':file_hash(root/'explicit-fixture-billing.json'),'observed_epoch':now})
 baseline=copy.deepcopy(read_json(actual/'watchdog-intent.json')['baseline'])
 baseline['observed_epoch']=now
 # No subprocess is permitted other than the exact read-only unit-state query.
 def unit_state(argv,**kwargs):
  assert argv[:3]==['systemctl','--user','is-active'] and argv[3] in ['ovllm-sustained-feasibility-v1-'+o+'.service' for o in ('controller','watchdog')]
  return subprocess.CompletedProcess(argv,3,'inactive\n','')
 with patch.object(Path,'cwd',return_value=root),patch.object(Path,'home',return_value=home),patch.object(time,'time',return_value=now),patch.object(provider,'account',return_value=baseline),patch.object(subprocess,'run',side_effect=unit_state):
  runpy.run_path(str(script))
 out=root/'.ovllm-cache/sustained-feasibility-v2'
 rental=read_json(out/'rental-intent.json');assert rental['payload']['allowedCudaVersions']==['13.0','13.2']
 text=(out/'controller.service').read_text()
 assert 'provider_request_receipts.py' in text and '--private-responses '+str(home) in text
 assert '/provider-responses/' in text and '--diagnostics '+str(out)+'/provider-diagnostics' in text
 assert rental['watchdog_intent']['plan']['input']['maximum_seconds']==18000
 assert rental['watchdog_intent']['plan']['maximum_charge_micro_usd']==4809058
result={'schema':'ovl.local-feasibility-preparation-check.v1','result':'PASS','scope':'Explicit disposable local mocks only; no provider calls, no units started, no closure or production credit',
 'checks':['real absent closure rejects before account read or attempt creation','valid historical POD observation accepts only available CUDA versions','real rental and watchdog validators accept conservative allocation','same 5-hour/120-second limits','private response path outside attempt evidence'],
 'preparation_sha256':file_hash(script),'workload_driver_sha256':file_hash(repo/'.ovllm-cache/start_feasibility_workload_v2.py')}
write_json(report,result);print(json.dumps(result))
