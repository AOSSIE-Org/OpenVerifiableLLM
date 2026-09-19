from pathlib import Path
import sys,time,os
sys.path[:0]=['src']
from ovl_pipeline import production_observation as p
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash
root=Path('.ovllm-cache/production-preflight-profile-v3');root.mkdir(exist_ok=False)
recipe=read_json(Path('.ovllm-cache/final-recipe-census-v1/recipe.json'))
base=Path('.ovllm-cache/production-preparation-v1')
write_json(root/'intent.json',{'schema':'ovl.preflight-comparison-intent.v1','scope':'Complete local two-phase startup checks, identical corpus and recipe. No CUDA, reconstruction, training or replay credit. Baseline and proposed process-local reuse executed sequentially; page cache and CPU load confound timing.','recipe_sha256':digest(recipe),'source_sha256':file_hash(Path(p.__file__))})
def run(name):
 out=root/name;out.mkdir();os.environ['OVL_ACTIVITY_FILE']=str((out/'activity.json').resolve());p._pass_index=0
 started=time.monotonic_ns();result={};phases=[]
 def timed(label,call):
  begin=time.monotonic_ns();value=call();phases.append({'name':label,'elapsed_ms':(time.monotonic_ns()-begin+999999)//1000000});write_json(out/'phases.json',phases);print(name,label,phases[-1]['elapsed_ms'],flush=True);return value
 for phase in ['wikipedia','conversation']:
  directory=base/phase
  c=timed(phase+'-census',lambda:p.schedule_counts(directory,recipe))
  steps=[0]+list(range(recipe['boundary_every'],c['updates'],recipe['boundary_every']))+[c['updates']]
  b=timed(phase+'-cursors',lambda:p.boundary_cursors(directory,recipe,c['stream_sha256'],steps))
  result[phase]={'census':c,'cursors':b}
 timed('initialization-input-validation',lambda:p.validate_stream(base/'wikipedia',read_json(base/'wikipedia/stream.json')))
 write_json(out/'result.json',{'result':result,'total_ms':(time.monotonic_ns()-started+999999)//1000000,'phases':phases});return result
before=run('baseline');after=p.checked_stream_scope(run)('revalidated-reuse')
assert before==after
write_json(root/'comparison.json',{'schema':'ovl.local-preflight-comparison.v1','result':'PASS','identical_complete_results':True,'baseline':read_json(root/'baseline/result.json'),'revalidated_reuse':read_json(root/'revalidated-reuse/result.json'),'numerical_or_production_credit':False})
print('COMPLETE',flush=True)
