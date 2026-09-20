from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,os,shutil,stat,time,uuid
from ovl_pipeline.canonical import read_json,write_json,file_hash
root=Path.cwd();out=root/'project/evidence/raw-local-consolidation-v1';out.mkdir()
source=root/'.ovllm-cache/raw-download-v1/downloaded/raw/3a7f5106d4b6898fd2c8b10f71d9054ae4daacc9e1cbef248cb5069779525af1/wikipedia/enwiki-20260901-pages-articles.xml.bz2'
targets=[root/'.ovllm-cache/wikipedia/20260901/enwiki-20260901-pages-articles.xml.bz2',root/'.ovllm-cache/public-raw-stage-v1/wikipedia/enwiki-20260901-pages-articles.xml.bz2']
expected='859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116';size=25680955982
receipt=root/'project/evidence/raw-archive/download-v1-verification.json'
assert file_hash(receipt)=='a90be73051b7034a51c751172fcd4c7c8bd098beef97ca6b522fb01ddd517a9f'
def info(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_size==size
 return {'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'links':s.st_nlink,'mode':stat.S_IMODE(s.st_mode),'allocated_bytes':s.st_blocks*512}
before={str(p.relative_to(root)):info(p) for p in [source,*targets]}
assert before[str(targets[0].relative_to(root))]['inode']==before[str(targets[1].relative_to(root))]['inode']
assert before[str(source.relative_to(root))]['inode']!=before[str(targets[0].relative_to(root))]['inode']
assert len({v['device'] for v in before.values()})==1
started=int(time.time());free_before=shutil.disk_usage(root).free
with ThreadPoolExecutor(max_workers=2) as pool:hashes=list(pool.map(file_hash,[source,targets[0]]))
assert hashes==[expected,expected]
for p in [source,*targets]:assert info(p)==before[str(p.relative_to(root))]
write_json(out/'before.json',{'schema':'ovl.verified-raw-consolidation-intent.v1','observed_epoch':int(time.time()),'started_epoch':started,'scope':'full hashes of both existing physical raw copies before consolidating task-owned duplicate storage; preserve all logical paths and public archive','files':before,'sha256':expected,'fresh_full_hashes':hashes,'retained_source':str(source.relative_to(root)),'public_archive_revision':'9bead1f229420f8beb9a4a5cdea4e963d485b223','prior_full_public_download_receipt_sha256':file_hash(receipt),'disk_free_before':free_before})
# No source bytes are edited. The retained verified inode remains throughout.
os.chmod(source,0o444)
for p in targets:
 old=info(p);assert old['inode']==before[str(p.relative_to(root))]['inode'] and old['mtime_ns']==before[str(p.relative_to(root))]['mtime_ns']
 temporary=p.with_name('.verified-raw-link-'+uuid.uuid4().hex)
 os.link(source,temporary,follow_symlinks=False);os.replace(temporary,p)
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
after={str(p.relative_to(root)):info(p) for p in [source,*targets]}
assert len({(x['device'],x['inode']) for x in after.values()})==1
write_json(out/'after.json',{'schema':'ovl.verified-raw-consolidation-result.v1','observed_epoch':int(time.time()),'result':'PASS','sha256':expected,'intent_sha256':file_hash(out/'before.json'),'files':after,'disk_free_after':shutil.disk_usage(root).free,'logical_paths_preserved':True,'source_bytes_modified':False,'independent_local_physical_copies':1,'public_copy_retained':True,'scope':'storage consolidation only; original acquisition receipts remain historical; no new download or reconstruction credit','limitation':'all local paths now intentionally share one read-only inode; public verified HF archive and upstream remain additional retrieval locations; no local disaster-recovery claim'})
print({'result':'PASS','evidence':str(out.relative_to(root)),'released_duplicate_inode_allocated_bytes':before[str(targets[0].relative_to(root))]['allocated_bytes'],'retained_paths':len(after)})
