"""Build final-recipe Wiki/chat sustained records and full replays. No provisioning."""
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
    selection=read_json(base/'preliminary-selection.json')
    # Tiny final-recipe CUDA record/full replay/resume on the complete chat
    # stream checks hardware compatibility without three additional Wiki scans.
    # These are development probes, excluded from throughput forecasts.
    import copy
    for phase,interval in [('wikipedia',8192),('conversation',293)]:
      stream=read_json(original/phase/'stream.json')
      binding={'schema':'ovl.pilot-record-parent-binding.v1','recipe_sha256':selection['recipe_sha256'],
        'kernel_sha256':selection['kernel_sha256'],'stream_sha256':digest(stream),'code_root':selection['code_root']}
      for mode in ('record','replay'):
        name=phase+'-'+mode;parent=None if mode=='record' else phase+'-record'
        control=remote+'/control-'+name;output=remote+'/'+name
        args=['record' if mode=='record' else 'replay','--stream',remote+'/prepared/'+phase,'--output',output]
        if mode=='record':args+=['--recipe',remote+'/inputs/source/candidate/recipe.json','--kernel',remote+'/inputs/source/candidate/kernel.json',
                                 '--seconds','600','--checkpoint-every',str(interval),'--warmup-updates','4']
        else:
            args+=['--record-directory',remote+'/'+parent,'--expected-record-sha256',RECORD]
        session=digest({'candidate':selection,'phase':phase,'mode':mode,'attempt':rental['watchdog_intent']['plan']['input']['attempt_id']})
        args+=['--delivery-session',session,'--delivery-deadline',DEADLINE,'--delivery-timeout','420','--delivery-maximum-bytes',str(352*1024**2)]
        job={**common,'kind':'pilot','required_files':numerical_required,
            'environment':{**common['environment'],'OVL_ACTIVITY_FILE':control+'/activity.json'},
            'argv':[remote+'/runtime/public-python/python/bin/python3.12','-I','-S',remote+'/inputs/pod_sustained_pilot.py',
                    '--setup-script',remote+'/inputs/pod_runtime_setup.py','--setup-sha256',file_hash(inputs/'pod_runtime_setup.py'),
                    '--config',remote+'/inputs/offline-config.json','--config-sha256',file_hash(inputs/'offline-config.json'),
                    '--inputs',remote+'/inputs','--runtime',remote+'/runtime','--control',control,'--deadline',DEADLINE,'--',*args],
            'export_roots':[output,control]}
        add(name,job,2100,1200 if phase=='wikipedia' else 1650,2*1024**3 if phase=='wikipedia' else 10*1024**3,parent_stage=parent,
            parent_record_root=None if parent is None else remote+'/'+parent,parent_binding=binding,
            validation_binding={'schema':'ovl.pilot-validation-binding.v1','stream_sha256':digest(stream),'documents':stream['documents']},
            retention={'schema':'ovl.pilot-checkpoint-retention.v1','session':session,'mode':mode,'output_root':output,'phase':phase,'maximum_checkpoint_bytes':352*1024**2,'copy_timeout_seconds':420,'maximum_uncached_export_bytes':512*1024**2,'binding':binding})
    chat_record=next(s for s in specs if s['name']=='conversation-record')
    chat_replay=next(s for s in specs if s['name']=='conversation-replay')
    tiny=[]
    for mode in ('record','replay','resume'):
        src=chat_record if mode=='record' else chat_replay
        job=copy.deepcopy(read_json(inputs/src['template_path']))
        name='cuda-'+mode;old='conversation-'+('record' if mode=='record' else 'replay')
        job['argv']=[a.replace('/'+old,'/'+name).replace('/control-'+old,'/control-'+name) for a in job['argv']]
        args=job['argv']
        for flag in ('--delivery-session','--delivery-deadline','--delivery-timeout','--delivery-maximum-bytes'):
            i=args.index(flag);del args[i:i+2]
        if mode=='record':
            i=args.index('--seconds');args[i:i+2]=['--updates','4']
            args[args.index('--checkpoint-every')+1]='2'
        else:
            args[args.index('--record-directory')+1]=remote+'/cuda-record'
            if mode=='resume':args+=['--resume-from','1']
        job['environment']['OVL_ACTIVITY_FILE']=remote+'/control-'+name+'/activity.json'
        job['export_roots']=[remote+'/'+name,remote+'/control-'+name]
        add(name,job,300,1200,1024**3,
            parent_stage=None if mode=='record' else 'cuda-record',
            parent_record_root=None if mode=='record' else remote+'/cuda-record',
            parent_binding=copy.deepcopy(src['parent_binding']),validation_binding=copy.deepcopy(src['validation_binding']),
            retention={'schema':'ovl.pilot-initial-retention.v1','mode':mode,'output_root':remote+'/'+name,
                       'phase':'conversation','maximum_initial_bytes':192*1024**2})
        tiny.append(specs.pop())
    specs[2:2]=tiny
    plan={'schema':'ovl.sustained-pilot-plan.v1','rental_intent_sha256':digest(rental),'profile_sha256':digest(profile),
        'worker_sha256':file_hash(Path('scripts/pod_job_worker.py')),'timing':{
            'transfer_floor_bytes_per_second':1024**2,'hash_floor_bytes_per_second':100*1024**2,
            'basis':'RTX5090 focused qualification: reused complete public data/reconstruction. Tiny4-update conversation record, full replay, then2-update resume; excluded from timing. Larger2GiB Wiki logical export ceiling accommodates faster5090 without changing512MiB uncached transfer bound. Original absolute guards preserved within6.5hour rental. Final candidate only: both real full streams,600second minimum records and complete sequential replays.2100second phase ceilings cover measured750second Wiki setup,600second numerical timing and actual bounded checkpoint deliveries. Each complete checkpoint waits for actual off-pod safe-state ACK under one fixed<=420second copy deadline; those waits are included in timed work except initial setup. Pilot intervals8192Wiki/293chat cover selected36160primary/9040recovery production density including final tails. Terminal limits count all logical bytes and actual rehashed uncached content before transfer. Floors1MiB/s transfer and100MiB/s hashing, ten hash passes plus30seconds, preserved. All four phase and export budgets plus setup/upload fit the selected6.5hour rental with1800second graceful reserve, unchanged1800second durable-export age,300second progress age, and100USD/90USD/10USD caps. No production admission.'},
        'uploads':uploads,'stages':specs}
    from types import SimpleNamespace
    validate(plan,digest(plan),rental,SimpleNamespace(profile=profile),inputs,Path('scripts/pod_job_worker.py'))
    write_json(base/'workload-plan.json',plan)
    return plan


