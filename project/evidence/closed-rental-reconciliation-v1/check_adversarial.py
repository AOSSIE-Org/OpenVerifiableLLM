"""Mutate copied cost evidence only; no provider calls or live-state writes."""
from pathlib import Path
import importlib.util,json,shutil,sys,tempfile
spec=importlib.util.spec_from_file_location('cost_reconcile',Path(__file__).with_name('reconcile.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
root=Path(sys.argv[1]);good=m.reconcile(root);assert good['result']=='PASS'
cases=['missing-watchdog','wrong-resource','changed-journal','foreign-bill','excess-bill','nonfinite-bill','duplicate-bill-key','zero-rate-parent-changed']
for damage in cases:
 with tempfile.TemporaryDirectory(prefix='ovl-cost-adversarial-') as temporary:
  dest=Path(temporary)/'case';shutil.copytree(root/'live-diagnostic-v1',dest)
  if damage=='missing-watchdog':(dest/'watchdog/result.json').unlink()
  elif damage=='wrong-resource':
   p=dest/'controller/result.json';v=m.read_json(p);v['pod_id']='foreign-resource';m.write_json(p,v)
  elif damage=='changed-journal':
   p=dest/'controller/event-00000000.json';v=m.read_json(p);v['body']['payload']['gpuCount']=2;m.write_json(p,v)
  elif damage=='zero-rate-parent-changed':
   p=dest/'zero-rate-rental-intent.json';v=m.read_json(p);v['watchdog_intent']['baseline']['account_hourly_usd']='1';m.write_json(p,v)
  else:
   p=dest/'billing.json';v=json.loads(p.read_text())
   if damage=='foreign-bill':v['records'][0]['podId']='foreign-resource'
   elif damage=='excess-bill':v['metadata']['totals']['totalAmount']=999
   elif damage=='nonfinite-bill':v['metadata']['totals']['totalAmount']=float('nan')
   elif damage=='duplicate-bill-key':
    p.write_text('{"metadata":{},"metadata":{},"records":[]}')
   if damage!='duplicate-bill-key':p.write_text(json.dumps(v))
  try:m.check(dest)
  except (m.EvidenceError,OSError,ValueError):pass
  else:raise AssertionError('accepted '+damage)
print('PASS complete ten-rental positive check and eight evidence/identity/billing counterexamples; no guard or provider changes')
