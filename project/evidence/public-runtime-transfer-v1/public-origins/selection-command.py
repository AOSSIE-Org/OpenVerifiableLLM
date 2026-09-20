from pathlib import Path
import concurrent.futures,hashlib,json,sys,time,urllib.request,urllib.parse
sys.path.insert(0,'src')
from ovl_pipeline.canonical import read_json,write_json,file_hash
from ovl_pipeline.runtime_audit import locked_requirements
out=Path('.ovllm-cache/runtime-public-origins-v1');out.mkdir(exist_ok=False);(out/'metadata').mkdir()
manifest=read_json(Path('.ovllm-cache/gpu-wheel-audit-v1/wheel-payloads.json'));locked=locked_requirements(Path('requirements/gpu.lock'))
def select(p):
 name=p['name'];pin=locked[name]
 if pin['url']:
  url=pin['url'];metadata=None
 else:
  url='https://pypi.org/pypi/'+urllib.parse.quote(name,safe='')+'/'+urllib.parse.quote(p['version'],safe='')+'/json'
  req=urllib.request.Request(url,headers={'User-Agent':'OpenVerifiableLLM-public-origin-selection/1','Accept-Encoding':'identity'})
  with urllib.request.urlopen(req,timeout=30) as r:
   if r.url!=url or r.status!=200:raise ValueError('unexpected metadata response')
   data=r.read(2*1024**2+1)
  if len(data)>2*1024**2:raise ValueError('oversized metadata')
  (out/'metadata'/(name+'.json')).write_bytes(data);v=json.loads(data)
  choices=[x for x in v['urls'] if x['filename']==p['wheel']]
  if len(choices)!=1:raise ValueError('exact wheel unavailable: '+name)
  item=choices[0]
  if item['size']!=p['bytes'] or item['digests']['sha256']!=p['sha256']:raise ValueError('public metadata differs: '+name)
  metadata={'url':url,'sha256':hashlib.sha256(data).hexdigest()};url=item['url']
 u=urllib.parse.urlsplit(url)
 if u.scheme!='https' or u.hostname not in ('files.pythonhosted.org','download-r2.pytorch.org') or u.username or u.password or u.fragment:raise ValueError('unselected public origin')
 return {'path':'wheels/'+p['wheel'],'bytes':p['bytes'],'sha256':p['sha256'],'url':url,'upstream_metadata':metadata}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:files=list(pool.map(select,manifest['packages']))
files.sort(key=lambda f:f['path'])
result={'schema':'ovl.runtime-public-origin-selection.v1','observed_epoch':int(time.time()),'dependency_lock_sha256':file_hash(Path('requirements/gpu.lock')),'wheel_manifest_sha256':file_hash(Path('.ovllm-cache/gpu-wheel-audit-v1/wheel-payloads.json')),'files':files,'scope':'public exact wheel locations matched to already fully audited local archives; no new binary download or remote setup yet'}
write_json(out/'selection.json',result);print('Selected',len(files),'public wheels',sum(x['bytes'] for x in files),'bytes',file_hash(out/'selection.json'))
