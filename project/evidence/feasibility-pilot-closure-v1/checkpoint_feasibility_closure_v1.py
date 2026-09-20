"""Archive actual both-owner closure only. No provider mutation or new rental."""
from pathlib import Path
from datetime import datetime,timezone
import sys,tarfile,shutil,json,time,subprocess,hashlib
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest,inventory,file_hash
from ovl_pipeline.supervision import Journal
from probe_provider_deadline import account
base=Path('.ovllm-cache/sustained-feasibility-v1');out=Path('project/evidence/feasibility-pilot-closure-v1')
r=read_json(base/'rental-intent.json');w=read_json(base/'watchdog-intent.json');results={}
for owner,intent in [('controller',r),('watchdog',w)]:
 result=read_json(base/owner/'result.json')
 assert result['complete'] is True and result['pod_id'] is None and not result['residual_network_volumes']
 assert result['intent_sha256']==digest(intent)
 assert result['confirmed_absent_epoch']>w['plan']['external_terminate_epoch']+180
 unit='ovllm-sustained-feasibility-v1-'+owner+'.service'
 state=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True)
 assert state.returncode==3 and state.stdout.strip()=='inactive'
 assert subprocess.check_output(['systemctl','--user','show',unit,'-p','Result','--value'],text=True).strip()=='success'
 results[owner]=result
observation=account();assert not observation['pods'] and not observation['volume_ids'] and observation['account_hourly_usd']=='0' and not observation['autopay']
out.mkdir(exist_ok=False)
entries=[]
for owner in ('controller','watchdog'):
 events=Journal(base/owner)._read()
 assert any(e['kind']=='teardown' and e['body'].get('complete') is True for e in events)
 assert not any(e['kind']=='creation-observed' for e in events)
 for i,e in enumerate(events):entries.append((owner+f'/event-{i:08d}.json',base/owner/f'event-{i:08d}.json'))
 for name in ('result.json','heartbeat.json'):
  if (base/owner/name).exists():entries.append((owner+'/'+name,base/owner/name))
for name in ('rental-intent.json','watchdog-intent.json','controller.service','watchdog.service','controller.stdout','controller.stderr','watchdog.stdout','watchdog.stderr'):
 entries.append((name,base/name))
manifest=[{'path':n,'bytes':p.stat().st_size,'sha256':file_hash(p)} for n,p in entries]
write_json(out/'journal-inventory.json',manifest)
archive=out/'complete-journals.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name,p in entries:
  assert not p.is_symlink();info=tar.gettarinfo(str(p),arcname=name);info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
  with p.open('rb') as f:tar.addfile(info,f)
with tarfile.open(archive,'r:gz') as tar:
 members=tar.getmembers();assert len(members)==len(manifest)
 assert [m.name for m in members]==[f['path'] for f in manifest]
 for member,f in zip(members,manifest):
  assert member.isfile() and member.size==f['bytes']
  content=tar.extractfile(member).read();assert hashlib.sha256(content).hexdigest()==f['sha256']
for owner,result in results.items():write_json(out/(owner+'-result.json'),result)
write_json(out/'account.json',observation)
write_json(out/'archive-verification.json',{'result':'PASS','files':len(manifest),'bytes':sum(f['bytes'] for f in manifest),'archive_sha256':file_hash(archive),'scope':'Every archived file membership, size and SHA256 checked against retained closed-attempt inventory; not training evidence.'})
value={'schema':'ovl.feasibility-pilot-closure.v1','observed_utc':datetime.now(timezone.utc).isoformat(),'attempt_id':r['payload']['name'],'status':'BOTH_GUARDS_CONFIRMED_ABSENCE_INACTIVE_SUCCESS','rental_intent_sha256':digest(r),'watchdog_intent_sha256':digest(w),'closure_epochs':{k:v['confirmed_absent_epoch'] for k,v in results.items()},'provider_deadline_epoch':w['plan']['provider_terminate_epoch'],'external_deadline_epoch':w['plan']['external_terminate_epoch'],'automatic_provider_termination':'UNVERIFIED','actual_pod_id':None,'original_creation_cause':'UNKNOWN; original raw response was not retained','billing':'No zero-charge claim; current4.809058USD upper reservation retained until actual reconciliation, prior6.935706USD also retained','pilot':'NOT_RUN','throughput':'NOT_MEASURED','production':'NOT_RUN','evidence':inventory(out,sorted(p.name for p in out.iterdir()))}
write_json(out/'checkpoint.json',value)
p=Path('project/goal_state.json');s=read_json(p,canonical_required=False)
s['feasibility_pilot_v1'].update(status=value['status'],closure_checkpoint=str(out/'checkpoint.json'),closure_sha256=digest(value),closure_epochs=value['closure_epochs'])
s['feasibility_priority']['status']='PRIOR_ATTEMPT_CLOSED_FRESH_PILOT_ADMISSION_PENDING'
s['budget']['prior_unsettled_reservation_usd']='11.744764';s['budget']['current_rental_projected_maximum_usd']='0'
s['budget']['remaining_mandatory_reservation_scope']='Previous closed attempt ceiling transferred to pending reservations; no new allocation or rental yet. Refresh actual billing/quote before next intent.'
s['budget']['project_spend_status']='Ten attributed pods last reported0.847605468283291015USD, not final settlement. Current unknown-ID attempt now closed by both owners; its4.809058USD ceiling remains reserved alongside prior6.935706USD pending ceilings.'
s['budget']['latest_account_observation']={'path':str(out/'account.json'),'sha256':file_hash(out/'account.json'),'scope':'Authenticated post-closure read: empty account, no provider mutation.'}
s['current_execution']['gpu_resources']=[]
s['next_action']='Refresh actual billing and exact POD/CUDA quote, then prepare one new bounded feasibility attempt with v2 driver if all checks pass; no production or optional infrastructure.'
s['updated_date']=value['observed_utc'];p.write_text(json.dumps(s,indent=2,sort_keys=True)+'\n')
print(json.dumps({'checkpoint_sha256':digest(value),'archive_files':len(manifest),'archive_bytes':archive.stat().st_size,'account_hourly_usd':observation['account_hourly_usd']}))
