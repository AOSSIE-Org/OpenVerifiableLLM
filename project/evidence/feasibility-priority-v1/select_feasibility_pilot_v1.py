"""Build one candidate cycle followed by a sustained Wikipedia record and full replay. No provisioning."""
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash,inventory,verify_inventory
from sustained_pilot_selection import DEADLINE,RECORD
from run_sustained_pilot import validate


def select(base,profile,rental,prepared_plan):
    base=Path(base);inputs=base/'inputs';remote=profile['remote_root']
    selected=inventory(inputs,[p.name for p in inputs.iterdir() if p.is_file()])
    uploads=[{'path':f['path'],'remote_path':'inputs/'+f['path'],'bytes':f['bytes'],'sha256':f['sha256']} for f in selected]
    required=[{'path':remote+'/'+f['remote_path'],'bytes':f['bytes'],'sha256':f['sha256']} for f in uploads]
    common={'schema':'ovl.pod-job.v1','cwd':remote,'environment':{'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},
            'stop_grace_seconds':15,'minimum_free_bytes':16*1024**3,'required_files':required,'deadline_epoch':DEADLINE}
    specs=[]
    def add(name,job,work,reserve,maximum,**bindings):
        path=inputs/(name+'-template.json');assert not path.exists();write_json(path,job)
        stage={'name':name,'template_path':path.name,'template_sha256':digest(job),'work_seconds':work,
            'export_reserve_seconds':reserve,'maximum_export_bytes':maximum,
            'parent_stage':None,'parent_record_root':None,'parent_binding':None,'validation_binding':None,
            'download_binding':None,'retention':None,**bindings}
        specs.append(stage)
    setup={**common,'kind':'setup','argv':['/usr/bin/python3','-I','-S',remote+'/inputs/pod_public_setup.py',
        '--config',remote+'/inputs/public-config.json','--config-sha256',file_hash(inputs/'public-config.json'),
        '--inputs',remote+'/inputs','--runtime',remote+'/runtime','--output',remote+'/setup-evidence','--deadline',DEADLINE],
        'export_roots':[remote+'/setup-evidence']}
    add('setup',setup,300,90,16*1024**2)
    download={**common,'kind':'setup','environment':{**common['environment'],'OVL_ACTIVITY_FILE':remote+'/input-evidence/activity.json'},
        'argv':['/usr/bin/python3','-I','-S',remote+'/inputs/pod_fetch_prepared.py','--plan',remote+'/inputs/prepared-plan.json',
                '--plan-sha256',digest(prepared_plan),'--output',remote+'/prepared','--report',remote+'/input-evidence/result.json','--deadline',DEADLINE],
        'export_roots':[remote+'/input-evidence']}
    add('prepared-inputs',download,1500,90,16*1024**2,download_binding={'schema':'ovl.public-input-binding.v1',
        'plan_sha256':digest(prepared_plan),'bytes':sum(f['bytes'] for f in prepared_plan['files'])})
    python=Path('.ovllm-cache/public-python-v1/python/bin/python3.12')
    numerical_required=[*required,{'path':remote+'/runtime/public-python/python/bin/python3.12','bytes':python.stat().st_size,'sha256':file_hash(python)},
        *[{**f,'path':remote+'/prepared/'+f['path']} for f in prepared_plan['files']]]
    original=Path('.ovllm-cache/production-preparation-v1')
    stream=read_json(original/'wikipedia/stream.json')
    selection=read_json(base/'preliminary-selection.json')
    binding={'schema':'ovl.pilot-record-parent-binding.v1','recipe_sha256':selection['recipe_sha256'],
        'kernel_sha256':selection['kernel_sha256'],'stream_sha256':digest(stream),'code_root':selection['code_root']}
    for name,mode,parent in [('record','record',None),('replay','replay','record'),('resume','resume','record'),('wiki-timed-record','record',None),('wiki-timed-replay','replay','wiki-timed-record')]:
        timed=name.startswith('wiki-timed-')
        control=remote+'/control-'+name;output=remote+'/'+name
        args=['record' if mode=='record' else 'replay','--stream',remote+'/prepared/wikipedia','--output',output]
        if mode=='record':args+=['--recipe',remote+'/inputs/source/candidate/recipe.json','--kernel',remote+'/inputs/source/candidate/kernel.json',
                                 *(['--seconds','600','--checkpoint-every','1000000'] if timed else ['--updates','8','--checkpoint-every','4']),'--warmup-updates','4']
        else:
            args+=['--record-directory',remote+'/'+parent,'--expected-record-sha256',RECORD]
            if mode=='resume':args+=['--resume-from','1']
        job={**common,'kind':'pilot','required_files':numerical_required,
            'environment':{**common['environment'],'OVL_ACTIVITY_FILE':control+'/activity.json'},
            'argv':[remote+'/runtime/public-python/python/bin/python3.12','-I','-S',remote+'/inputs/pod_sustained_pilot.py',
                    '--setup-script',remote+'/inputs/pod_runtime_setup.py','--setup-sha256',file_hash(inputs/'pod_runtime_setup.py'),
                    '--config',remote+'/inputs/offline-config.json','--config-sha256',file_hash(inputs/'offline-config.json'),
                    '--inputs',remote+'/inputs','--runtime',remote+'/runtime','--control',control,'--deadline',DEADLINE,'--',*args],
            'export_roots':[output,control]}
        add(name,job,1500,900 if timed else 1200,768*1024**2 if timed else 1024**3,parent_stage=parent,
            parent_record_root=None if parent is None else remote+'/'+parent,parent_binding=binding,
            validation_binding={'schema':'ovl.pilot-validation-binding.v1','stream_sha256':digest(stream),'documents':stream['documents']},
            retention={'schema':'ovl.pilot-initial-retention.v1','mode':mode,'output_root':output,'phase':'wikipedia','maximum_initial_bytes':192*1024**2})
    plan={'schema':'ovl.sustained-pilot-plan.v1','rental_intent_sha256':digest(rental),'profile_sha256':digest(profile),
        'worker_sha256':file_hash(Path('scripts/pod_job_worker.py')),'timing':{
            'transfer_floor_bytes_per_second':1024**2,'hash_floor_bytes_per_second':100*1024**2,
            'basis':'Qualification and feasibility only. Complete input verification unchanged. Original tiny cycle actual setup107s and prior full-stream local validation230.727374s are observations, not remote guarantees. Freeze1500s numerical windows before creation, with no shorter nested timeout; keep1200s/1GiB full cycle exports. Timed Wikipedia record requests600s excluding warmup and full replay recomputes every update. Its checkpoint interval1000000 bounds a successful timed record to initial/final states (148046771+333138942 bytes measured for this recipe); each complete timed output capped768MiB with900s export reserve at previously selected1MiB/s transfer/100MiB/s hash floors plus10hash passes and30s overhead. No checkpoint-density or full-run admission credit until actual rates, final complete recovery schedule, chat timing and all fixed costs are checked. Original300s progress/1800s durable-export guards and100USD/90USD/10USD reserve remain unchanged.'},
        'uploads':uploads,'stages':specs}
    from types import SimpleNamespace
    validate(plan,digest(plan),rental,SimpleNamespace(profile=profile),inputs,Path('scripts/pod_job_worker.py'))
    write_json(base/'workload-plan.json',plan)
    return plan


