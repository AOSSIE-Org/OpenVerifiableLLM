from pathlib import Path
import sys
sys.path.insert(0,'src')
from huggingface_hub import hf_hub_download,constants
from ovl_pipeline.canonical import read_json,write_json,digest,verify_inventory
from ovl_pipeline.anchoring import PublisherPolicy,verify_anchor
r=Path('.ovllm-cache/source-anchor-v1');v=read_json(r/'upload.json');out=r/'fresh-public-download'
if not constants.HF_HUB_DISABLE_XET:raise RuntimeError('fresh process Xet disabled required')
out.mkdir(exist_ok=False)
for e in v['files']:
 hf_hub_download(v['repo'],repo_type='dataset',revision=v['revision'],filename=v['prefix']+'/'+e['path'],local_dir=out/'downloaded',cache_dir=out/'fresh-cache',token=False,force_download=True,endpoint='https://huggingface.co')
root=out/'downloaded'/v['prefix'];verify_inventory(root,v['files'])
if (root/'statement.json').read_bytes()!=(r/'expected-statement.json').read_bytes():raise RuntimeError('download differs from externally reconstructed source statement')
policy=PublisherPolicy(**read_json(r/'operator-selected-policy.json'))
check=verify_anchor(root/'statement.json',root/'statement.sigstore.json',policy,policy_origin='operator-reconstructed-from-source')
write_json(out/'anchor-verification.json',check)
receipt={'schema':'ovl.source-anchor-download.v1','result':'PASS','repo':v['repo'],'revision':v['revision'],'prefix':v['prefix'],'files':v['files'],'anchor_check_sha256':digest(check),'source_statement_sha256':digest(read_json(root/'statement.json')),'anonymous':True,'force_download':True,'fresh_directory':True,'policy_source':'operator reconstructed from GitHub request commit, outside downloaded artifact','scope':'public source endorsement and archive integrity; reconstruction/training/replay NOT_RUN'}
write_json(out/'verification.json',receipt)
print('fresh_public_source_anchor',receipt['result'],'receipt_sha256',digest(receipt))
