from pathlib import Path
import sys
sys.path[:0]=['scripts','src']
from ovl_pipeline.canonical import read_json,digest,write_json,file_hash
from run_sustained_pilot import retained_stage
from verify_pilot_cycle import verify
root=Path('.ovllm-cache/rtx5090-feasibility-v2');output=root/'workload'
plan=read_json(root/'workload-plan.json');assert digest(plan)=='09f777a14a8681865b58a59315d691a0e76e72cfc002bf83d3237eddb3f39020'
profile=read_json(root/'profile.json');selected={}
for name in ('cuda-record','cuda-replay','cuda-resume'):
 stage=next(x for x in plan['stages'] if x['name']==name)
 result=retained_stage(stage,output,profile)
 assert result['exit']['state']=='EXITED' and result['exit']['exit_code']==0
 entry=next(x for x in result['exports'] if x['remote_root']==name)
 selected[name]=entry
binding=next(x for x in plan['stages'] if x['name']=='cuda-record')['parent_binding']
s={'schema':'ovl.retained-pilot-cycle-selection.v1','binding':binding,'expected_record':'75fc0bb287bf61ee1c95f1ffb09a2cb01b9d2c60ab18b8ed2ef4f0ddea71a0d3','resume_from':1}
for phase,name in [('record','cuda-record'),('replay','cuda-replay'),('resume','cuda-resume')]:
 s[phase+'_directory']=selected[name]['directory'];s[phase+'_files']=selected[name]['files']
out=Path('.ovllm-cache/rtx5090-cycle-verification-v1');out.mkdir(exist_ok=False)
write_json(out/'selection.json',s)
result=verify(**{k:v for k,v in s.items() if k!='schema'})
write_json(out/'verification.json',result)
print(digest(result),result)
