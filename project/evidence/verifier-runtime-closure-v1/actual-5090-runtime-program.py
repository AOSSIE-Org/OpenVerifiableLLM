import sys,json,tempfile,os
from pathlib import Path
sys.path.insert(0,'/workspace/ovllm/rtx5090-feasibility-v2/inputs/source/src')
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
