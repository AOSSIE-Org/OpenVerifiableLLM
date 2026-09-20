"""One bounded SSH stream for an already selected inventory of regular files.

The caller still inventories/hashes before and after, and checks semantic state.
The stream is plain concatenated file bytes in the independently supplied order;
no archive member names or commands are accepted from its output.
"""
import hashlib
import io
import os
import shutil
import time
from pathlib import Path
from ovl_pipeline.canonical import EvidenceError,canonical,confined,digest,require_digest,verify_inventory,write_json
from ovl_pipeline.schema import fields,integer
from pod_transfer import relative

REMOTE_BULK=r'''
import hashlib,json,os,stat,sys
from pathlib import Path
root,name,selection=sys.argv[1:]
if not root.startswith('/') or '..' in Path(root).parts or str(Path(root))!=root:raise ValueError('root')
if selection=='root':
 if name!='.':raise ValueError('whole root')
elif selection!='subtree' or any(x in ('','.','..') for x in name.split('/')):raise ValueError('subtree')
data=sys.stdin.buffer.read(16*1024**2+1)
if len(data)>16*1024**2:raise ValueError('inventory limit')
files=json.loads(data)
if type(files) is not list or not 1<=len(files)<=100000:raise ValueError('inventory shape')
for item in files:
 if set(item)!={'path','bytes','sha256'} or any(x in ('','.','..') for x in item['path'].split('/')):raise ValueError('file shape')
 path=Path(root)/name/item['path'];current=Path('/')
 for part in path.parts[1:]:
  current=current/part
  if current.is_symlink():raise ValueError('symlink')
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as f:
  info=os.fstat(f.fileno())
  if not stat.S_ISREG(info.st_mode) or info.st_size!=item['bytes']:raise ValueError('type or length')
  h=hashlib.sha256();count=0
  while True:
   block=f.read(min(1024**2,item['bytes']-count+1))
   if not block:break
   count+=len(block)
   if count>item['bytes']:raise ValueError('file grew')
   h.update(block);sys.stdout.buffer.write(block)
  if count!=item['bytes'] or h.hexdigest()!=item['sha256']:raise ValueError('file differs')
sys.stdout.buffer.flush()
'''


def receive(transport,name,files,output,deadline,*,progress=None,whole_root=False,wall=time.time,monotonic=time.monotonic):
    started=wall();end=monotonic()+deadline-started
    def check_deadline():
        if min(deadline-wall(),end-monotonic())<=0:raise EvidenceError('bulk export deadline expired; preserve partials')
    check_deadline()
    if whole_root:
        if name!='.':raise EvidenceError('explicit whole-root selection required')
    else:relative(name)
    if type(files) is not list or not 1<=len(files)<=100000:raise EvidenceError('nonempty bounded bulk inventory required')
    names=[];total=0
    for f in files:
        fields(f,'path bytes sha256','selected bulk file');relative(f['path']);require_digest(f['sha256'])
        integer(f['bytes'],0,2**40,'selected bulk file bytes');names.append(f['path']);total+=f['bytes']
    if names!=sorted(set(names)) or total>2**40:raise EvidenceError('unique sorted bounded bulk files required')
    selected=set(names)
    for name_parts in (n.split('/') for n in names):
        if any('/'.join(name_parts[:i]) in selected for i in range(1,len(name_parts))):raise EvidenceError('bulk file/parent collision')
    body=canonical(files)
    if len(body)>16*1024**2:raise EvidenceError('bulk selection exceeds framing bound')
    output=Path(output).absolute()
    if any(p.is_symlink() for p in [output,*output.parents]):raise EvidenceError('bulk output symlink')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    space=shutil.disk_usage(output)
    if space.free-2*total<(space.total+4)//5:raise EvidenceError('bulk export needs duplicate stream space and twenty percent headroom')
    target=output/'files';target.mkdir(mode=0o700);partial=output/'stream.partial'
    operation=digest({'profile':digest(transport.profile),'tree':name,'files':files})
    with partial.open('xb') as raw:
        receipt=transport.stream(['/usr/bin/python3','-c',REMOTE_BULK,transport.profile['remote_root'],name,'root' if whole_root else 'subtree'],
            raw,total,deadline,source=io.BytesIO(body),source_bytes=len(body),
            progress=None if progress is None else lambda counts:progress(operation,counts,total))
        raw.flush();os.fsync(raw.fileno())
    if receipt['bytes_received']!=total:raise EvidenceError('incomplete bulk stream; preserve partial')
    check_deadline()
    with partial.open('rb') as raw:
        for item in files:
            check_deadline()
            dest=confined(target,item['path']);dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            remaining=item['bytes'];h=hashlib.sha256()
            with dest.open('xb') as f:
                while remaining:
                    check_deadline()
                    block=raw.read(min(1024**2,remaining))
                    if not block:raise EvidenceError('truncated selected bulk file')
                    remaining-=len(block);h.update(block);f.write(block)
                f.flush();os.fsync(f.fileno())
            if h.hexdigest()!=item['sha256']:raise EvidenceError('bulk file differs; preserve all bytes')
        if raw.read(1):raise EvidenceError('bulk trailing bytes')
    verify_inventory(target,files)
    check_deadline()
    result={'schema':'ovl.selected-bulk-transfer.v1','result':'PASS','operation_sha256':operation,'files':files,'transfer':receipt,
            'operator_observed_start_epoch':int(started),'operator_observed_finish_epoch':int(wall()),'original_deadline_epoch':deadline,
            'retained_file_bytes':total,'peak_stream_and_split_bytes_bound':2*total,
            'files_directory':str(target),'scope':'all selected file bytes independently hashed; peer identity/training checks separate'}
    write_json(output/'transfer.json',result)
    # The verified split files preserve the exact stream bytes in selected order.
    # Only this successful duplicate is removed; failed partials are retained.
    partial.unlink()
    return result
