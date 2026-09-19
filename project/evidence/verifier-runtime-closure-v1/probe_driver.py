from pathlib import Path
import sys,json,time
sys.path[:0]=['scripts','src']
from ovl_pipeline.canonical import read_json,write_json,file_hash
from pod_transfer import Transport
r=Path('.ovllm-cache/rtx5090-feasibility-v2');p=read_json(r/'profile.json');remote=p['remote_root']
out=Path('.ovllm-cache/rtx5090-verifier-closure-v3');out.mkdir(exist_ok=False)
source=Path('project/evidence/source-anchor-preparation-1')
payload={name:(source/name).read_text() for name in ('statement.json','statement.sigstore.json','operator-selected-policy.json')}
write_json(out/'public-inputs.json',payload)
program='''import sys,json,tempfile,os
from pathlib import Path
sys.path.insert(0,SOURCE)
from ovl_pipeline import gpu_pilot,gpu
from ovl_pipeline.anchoring import PublisherPolicy,verify_anchor
from ovl_pipeline.canonical import read_json
payload=json.load(sys.stdin)
baseline=gpu.mapped_libraries()
from sigstore.models import Bundle,ClientTrustConfig
from sigstore.verify import Verifier,policy
preloaded=gpu.mapped_libraries()
with tempfile.TemporaryDirectory(prefix="ovl-verifier-closure-") as d:
 root=Path(d);os.environ["XDG_CACHE_HOME"]=str(root/"cache")
 for name,data in payload.items():(root/name).write_text(data)
 verification=verify_anchor(root/"statement.json",root/"statement.sigstore.json",PublisherPolicy(**read_json(root/"operator-selected-policy.json")))
 after=gpu.mapped_libraries()
 print(json.dumps({"schema":"ovl.actual-runtime-verifier-closure.v1","scope":"existing5090pinnedruntime CPU-only signature/import observation; no CUDA configure, allocation, training or qualification credit","baseline":baseline,"preloaded":preloaded,"after_verification":after,"new_after_imports":[x for x in preloaded if x not in baseline],"new_after_verification":[x for x in after if x not in preloaded],"verification":verification},sort_keys=True))
'''.replace('SOURCE',repr(remote+'/inputs/source/src'))
(out/'program.py').write_text(program)
t=Transport(p,Path.home()/'.local/share/openverifiablellm/ssh/runpod-ed25519',r/'known-hosts')
with (out/'public-inputs.json').open('rb') as source_file, (out/'result.json').open('xb') as result_file:
 res=t.stream([remote+'/runtime/venv/bin/python','-B','-c',program],result_file,2**20,int(time.time())+120,source=source_file,source_bytes=(out/'public-inputs.json').stat().st_size)
write_json(out/'transport.json',res)
v=json.loads((out/'result.json').read_text());print({'verification':v['verification']['result'],'new_after_imports':[x['name'] for x in v['new_after_imports']],'new_after_verification':[x['name'] for x in v['new_after_verification']],'report_sha256':file_hash(out/'result.json')})
