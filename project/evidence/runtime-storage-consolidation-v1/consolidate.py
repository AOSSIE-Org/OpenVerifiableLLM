"""One-shot byte-checked deduplication of three inactive generated runtimes."""
from pathlib import Path
import os, sys, time, uuid
sys.path[:0]=['src']
from ovl_pipeline.canonical import read_json, write_json, file_hash, digest, confined

out=Path('project/evidence/runtime-storage-consolidation-v1');out.mkdir(exist_ok=False)
manifest_path=Path('.ovllm-cache/offline-runtime-setup-v1/evidence/wheel-payloads.json')
manifest=read_json(manifest_path)
assert digest(manifest)=='7efc6d5eace59b57e11b3bf4928dd9d736f41fae742578047aa50319bfbf6c6e'
roots=[Path('.ovllm-cache/gpu-venv'),Path('.ovllm-cache/offline-runtime-setup-v1/runtime/venv'),Path('.ovllm-cache/offline-runtime-setup-v2/runtime/venv')]
def path(root,item):
    assert item['scheme'] in ('site','prefix')
    return confined(root/('lib/python3.12/site-packages' if item['scheme']=='site' else ''),item['path'])
selected=[]
for item in manifest['payloads']:
    paths=[path(root,item) for root in roots]
    for p in paths:
        assert p.is_file() and not p.is_symlink() and p.stat().st_size==item['bytes'] and file_hash(p)==item['sha256'], str(p)
    assert len({p.stat().st_dev for p in paths})==1
    selected.append((item,paths))
intent={'schema':'ovl.runtime-storage-consolidation.v1','observed_epoch':int(time.time()),
    'manifest_path':str(manifest_path),'manifest_sha256':file_hash(manifest_path),'roots':[str(p) for p in roots],
    'payload_count':len(selected),'payload_bytes_per_runtime':sum(item['bytes'] for item,_ in selected),
    'precondition':'Every complete payload in all three runtimes rehashed against retained locked-wheel manifest before any replacement.',
    'operation':'Atomically replace only byte-identical duplicate package payloads with same-filesystem hard links; retain all logical paths, generated metadata, public origins, archives and audit evidence.',
    'limitation':'Shared payload inodes are not independent physical copies. Recreate a runtime before installing or modifying packages; existing launch audits still rehash every payload.',
    'production_acceptance':'NOT_RUN'}
write_json(out/'intent.json',intent)
replaced=0;previous_unique=set();current_unique=set()
for item,paths in selected:
    source=paths[0]
    for p in paths:previous_unique.add((p.stat().st_dev,p.stat().st_ino))
    for target in paths[1:]:
        if os.path.samefile(source,target):continue
        # Recheck the source and target immediately before this replacement.
        assert file_hash(source)==item['sha256'] and file_hash(target)==item['sha256']
        temporary=target.with_name(target.name+'.ovl-dedup-'+uuid.uuid4().hex)
        os.link(source,temporary);os.replace(temporary,target)
        fd=os.open(target.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        replaced+=1
    assert all(os.path.samefile(source,p) for p in paths)
    assert file_hash(source)==item['sha256']
    current_unique.add((source.stat().st_dev,source.stat().st_ino))
write_json(out/'verification.json',{'schema':'ovl.runtime-storage-consolidation-result.v1','result':'PASS',
    'intent_sha256':digest(intent),'completed_epoch':int(time.time()),'payloads_replaced':replaced,
    'prior_unique_payload_inodes':len(previous_unique),'retained_unique_payload_inodes':len(current_unique),
    'all_paths_and_payload_bytes_preserved':True,'all_final_payloads_rehashed':True,
    'generated_metadata_and_evidence':'UNCHANGED','production_acceptance':'NOT_RUN'})
print('PASS: complete payload bytes and all logical runtime paths preserved;',replaced,'duplicate payloads consolidated.')
