from pathlib import Path
import io,os,shutil,sys,tarfile
sys.path[:0]=['src']
from ovl_pipeline.canonical import file_hash,inventory,write_json,read_json
from ovl_pipeline.fixture import recipe
base=Path('.ovllm-cache/live-tiny-cuda-v1');base.mkdir(exist_ok=True);inputs=base/'inputs';inputs.mkdir(exist_ok=False)
source=base/'source-staging';source.mkdir()
for p in sorted(Path('src').rglob('*.py')):
 q=source/p;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
p=source/'requirements/gpu.lock';p.parent.mkdir();shutil.copyfile('requirements/gpu.lock',p)
fixture=Path('.ovllm-cache/complete-preparation-fixture-v6/original')
for p in sorted((fixture/'wikipedia').iterdir()):
 if p.is_file():
  q=source/'fixture/wikipedia'/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
write_json(source/'fixture/recipe.json',recipe(320));write_json(source/'fixture/kernel.json',{'schema':'ovl.gpu-kernel.v1','precision':'bf16'})
files=inventory(source,[p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()])
with tarfile.open(inputs/'source.tar.gz','w:gz',format=tarfile.USTAR_FORMAT) as tar:
 for f in files:
  i=tarfile.TarInfo(f['path']);i.size=f['bytes'];i.mode=0o400;i.mtime=0
  with (source/f['path']).open('rb') as raw:tar.addfile(i,raw)
archive=inputs/'python.tar.gz';os.link('.ovllm-cache/python-origin-v1/cpython-3.12.14+20260814-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz',archive)
for name in ['pod_fetch_runtime.py','pod_runtime_setup.py','pod_public_setup.py','pod_tiny_cuda_probe.py']:shutil.copyfile('scripts/'+name,inputs/name)
origins=read_json(Path('.ovllm-cache/runtime-public-origins-v1/selection.json'))
write_json(inputs/'wheel-plan.json',{'schema':'ovl.public-wheel-download.v1','files':[{'path':f['path'].removeprefix('wheels/'),'url':f['url'],'bytes':f['bytes'],'sha256':f['sha256']} for f in origins['files']]})
wheels=Path('.ovllm-cache/gpu-locked-wheels-v1')
offline={'schema':'ovl.offline-runtime-setup.v1','source_root':'source','source_files':files,'dependency_lock':'requirements/gpu.lock','interpreter_archive':'python.tar.gz','interpreter_sha256':file_hash(archive),'wheels':'wheels','bootstrap_wheels':[{'path':p.name,'sha256':file_hash(p)} for p in sorted(wheels.glob('*.whl')) if p.name.split('-')[0] in ('rfc8785','packaging')]}
write_json(inputs/'offline-config.json',offline)
public={'schema':'ovl.public-runtime-setup.v1','offline_config':'offline-config.json','fetch_script':'pod_fetch_runtime.py','setup_script':'pod_runtime_setup.py','wheel_plan':'wheel-plan.json','source_archive':'source.tar.gz','download_seconds':180}
for k in ('offline_config','fetch_script','setup_script','wheel_plan','source_archive'):public[k+'_sha256']=file_hash(inputs/public[k])
write_json(inputs/'public-config.json',public)
write_json(base/'input-inventory.json',inventory(inputs,[p.name for p in inputs.iterdir()]))
print('Prepared selected tiny CUDA inputs; no provisioning',len(files),'source files')
