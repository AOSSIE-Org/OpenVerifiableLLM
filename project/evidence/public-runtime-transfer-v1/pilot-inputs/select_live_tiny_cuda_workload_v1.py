from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash
from pod_transfer import Transport
from run_workload_coordinator import validate
root=Path.cwd();out=root/'.ovllm-cache/live-tiny-cuda-v1';profile=read_json(out/'profile.json');rental=read_json(out/'rental-intent.json');start=rental['watchdog_intent']['plan']['input']['now_epoch'];remote=profile['remote_root'];inputs=out/'inputs';worker=root/'scripts/pod_job_worker.py'
selected=read_json(out/'input-inventory.json')
uploads=[{'path':p['path'],'remote_path':'inputs/'+p['path'],'bytes':p['bytes'],'sha256':p['sha256']} for p in selected]
required=[{'path':remote+'/'+u['remote_path'],'bytes':u['bytes'],'sha256':u['sha256']} for u in uploads]
common={'schema':'ovl.pod-job.v1','cwd':remote,'environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},'stop_grace_seconds':15,'minimum_free_bytes':8*1024**3,'required_files':required}
setup={'kind':'setup','argv':['/usr/bin/python3','-I','-S',remote+'/inputs/pod_public_setup.py','--config',remote+'/inputs/public-config.json','--config-sha256',file_hash(inputs/'public-config.json'),'--inputs',remote+'/inputs','--runtime',remote+'/runtime','--output',remote+'/setup-evidence','--deadline',str(start+650)],'deadline_epoch':start+650,'export_roots':[remote+'/setup-evidence'],**common}
pilot={'kind':'pilot','argv':[remote+'/runtime/public-python/python/bin/python3.12','-I','-S',remote+'/inputs/pod_tiny_cuda_probe.py','--setup-script',remote+'/inputs/pod_runtime_setup.py','--setup-sha256',file_hash(inputs/'pod_runtime_setup.py'),'--config',remote+'/inputs/offline-config.json','--config-sha256',file_hash(inputs/'offline-config.json'),'--inputs',remote+'/inputs','--runtime',remote+'/runtime','--stream',remote+'/inputs/source/fixture/wikipedia','--recipe',remote+'/inputs/source/fixture/recipe.json','--kernel',remote+'/inputs/source/fixture/kernel.json','--output',remote+'/tiny-cuda','--deadline',str(start+1000)],'deadline_epoch':start+1000,'export_roots':[remote+'/tiny-cuda'],**common}
python=root/'.ovllm-cache/public-python-v1/python/bin/python3.12'
pilot['required_files']=[*required,{'path':remote+'/runtime/public-python/python/bin/python3.12','bytes':python.stat().st_size,'sha256':file_hash(python)}]
stages=[]
for name,job,maximum in [('setup',setup,16*1024**2),('tiny-cuda',pilot,64*1024**2)]:
 file=name+'-job.json';write_json(inputs/file,job);stages.append({'name':name,'job_path':file,'job_sha256':digest(job),'maximum_export_bytes':maximum})
plan={'schema':'ovl.finite-workload-plan.v3','rental_intent_sha256':digest(rental),'profile_sha256':digest(profile),'worker_sha256':file_hash(worker),'timing':{'export_reserve_seconds':360,'transfer_floor_bytes_per_second':256*1024,'hash_floor_bytes_per_second':100*1024**2,'basis':'Conservative diagnostic planning: selected8MiB upload completed within observed6s counter interval and download3s; choose256KiB/s floor, cached hashing observed1GiB/0.732s choose100MiB/s. Whole-workload rates NOT_RUN. Original34MiB upload bounded161s; final64MiB export+ten hash passes<360s. No production admission.'},'uploads':uploads,'stages':stages}
key=Path.home()/'.local/share/openverifiablellm/ssh/runpod-ed25519';transport=Transport(profile,key,out/'known-hosts');validate(plan,digest(plan),rental,transport,inputs,worker);write_json(out/'workload-plan.json',plan)
text=f'''[Unit]
Description=OpenVerifiableLLM bounded actual tiny CUDA workload
StartLimitIntervalSec=60
StartLimitBurst=3
[Service]
Type=simple
WorkingDirectory={root}
Environment=PYTHONPATH={root}/src:{root}/scripts
ExecStart={root}/.venv/bin/python {root}/scripts/run_workload_coordinator.py --plan {out}/workload-plan.json --plan-sha256 {digest(plan)} --rental-intent {out}/rental-intent.json --controller-journal {out}/controller --watchdog-heartbeat {out}/watchdog/heartbeat.json --profile {out}/profile.json --key {key} --known-hosts {out}/known-hosts --inputs {inputs} --worker {worker} --output {out}/workload --health {out}/health.json
Restart=on-failure
RestartSec=5
MemoryMax=4G
StandardOutput=append:{out}/workload.stdout
StandardError=append:{out}/workload.stderr
'''
p=Path.home()/'.config/systemd/user/ovllm-live-tiny-cuda-v1-workload.service';assert not p.exists();p.write_text(text);(out/'workload.service').write_text(text)
print('Validated selected tiny CUDA plan',digest(plan))
