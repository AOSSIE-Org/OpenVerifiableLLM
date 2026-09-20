from pathlib import Path
import sys,subprocess,runpy,json,os
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash
from pod_transfer import Transport
from run_sustained_pilot import validate
root=Path.cwd();out=root/'.ovllm-cache/sustained-feasibility-v2'
e=json.loads((out/'endpoint-observation.json').read_text())
endpoint=e['ssh']['direct'];assert endpoint; events=[json.loads(p.read_text()) for p in sorted((out/'controller').glob('event-*.json'))]; assert [v['body']['id'] for v in events if v['kind']=='creation-observed']==[e['id']]
print('Selected provider endpoint',endpoint)
known=out/'known-hosts';assert not known.exists()
r=subprocess.run(['ssh-keyscan','-T','15','-t','ed25519','-p',str(endpoint['port']),endpoint['host']],capture_output=True,timeout=20)
assert r.returncode==0 and len(r.stdout)<65536 and r.stdout.count(b'\n')==1
known.write_bytes(r.stdout);known.chmod(0o600)
profile={'schema':'ovl.pod-ssh-profile.v1','pod_id':e['id'],'host':endpoint['host'],'port':endpoint['port'],'user':'root','remote_root':'/workspace/ovllm/feasibility-pilot-v2','endpoint_observation_sha256':file_hash(out/'endpoint-observation.json'),'known_hosts_sha256':file_hash(known),'host_key_trust':'operator-pinned-TOFU'}
write_json(out/'profile.json',profile)
from ovl_pipeline.canonical import verify_inventory
base=out
admission=read_json(base/'qualification-overlap-admission.json')
decision=read_json(Path('project/evidence/candidate-cycle-overlap-v1/decision.json'))
assert admission['schema']=='ovl.candidate-qualification-overlap.v1' and admission['decision_sha256']==digest(decision)
assert admission['on_pod_complete15file_download']=='REQUIRED_BEFORE_PILOT' and admission['production_admission']=='NOT_RUN'
integrity=read_json(Path('project/evidence/complete-prepared-integrity-v1/verification.json'))
assert digest(integrity)==admission['integrity_sha256'] and integrity['result']=='PASS'
prepared=read_json(base/'inputs/prepared-plan.json')
full=read_json(Path('.ovllm-cache/full-prepared-publication-v1/plan.json'))
uploaded=read_json(Path('.ovllm-cache/full-prepared-publication-v1/upload/upload.json'))
assert prepared['preparation_sha256']==full['subject_sha256']==integrity['preparation_sha256']
assert prepared['revision']==uploaded['revision'] and prepared['repo']==full['repo']==uploaded['repo']
assert all(f in full['files'] for f in prepared['files'])
verify_inventory(base/'inputs',read_json(base/'input-inventory.json'))
select=runpy.run_path('.ovllm-cache/select_feasibility_pilot_v1.py')['select']
select(out,profile,read_json(out/'rental-intent.json'),prepared)
plan=read_json(out/'workload-plan.json');rental=read_json(out/'rental-intent.json')
key=Path.home()/'.local/share/openverifiablellm/ssh/runpod-ed25519';inputs=out/'inputs';worker=root/'scripts/pod_job_worker.py'
transport=Transport(profile,key,known);validate(plan,digest(plan),rental,transport,inputs,worker)
unit=f"""[Unit]
Description=OpenVerifiableLLM guarded complete candidate CUDA cycle
StartLimitIntervalSec=60
StartLimitBurst=3
[Service]
Type=simple
WorkingDirectory={root}
Environment=PYTHONPATH={root}/src:{root}/scripts
ExecStart={root}/.venv/bin/python {root}/scripts/run_sustained_pilot.py --plan {out}/workload-plan.json --plan-sha256 {digest(plan)} --rental-intent {out}/rental-intent.json --controller-journal {out}/controller --watchdog-heartbeat {out}/watchdog/heartbeat.json --profile {out}/profile.json --key {key} --known-hosts {known} --inputs {inputs} --worker {worker} --output {out}/workload --health {out}/health.json
Restart=on-failure
RestartSec=5
MemoryMax=4G
StandardOutput=append:{out}/workload.stdout
StandardError=append:{out}/workload.stderr
"""
u=Path.home()/'.config/systemd/user/ovllm-sustained-feasibility-v2-workload.service';assert not u.exists();u.write_text(unit);(out/'workload.service').write_text(unit)
print('Selected complete candidate plan',digest(plan),'Unit prepared, not launched.')
