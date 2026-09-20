"""Read a caller-selected metadata bundle in one bounded SSH exchange.

Each file is opened once and checked separately. This is not an atomic snapshot
across files, an authenticated checkpoint inventory, or useful-work telemetry.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
from ovl_pipeline.canonical import EvidenceError,parse_json,read_json,require_digest
from ovl_pipeline.schema import fields,integer
from pod_transfer import relative


REMOTE_OBSERVE=r'''
import base64,hashlib,json,os,stat,sys
from pathlib import Path
root,selection,maximum=sys.argv[1:];names=json.loads(selection);maximum=int(maximum)
if not root.startswith('/') or '..' in Path(root).parts or str(Path(root))!=root:raise ValueError('root')
if type(names) is not list or not 1<=len(names)<=16 or names!=sorted(set(names)) or not 1<=maximum<=65536:raise ValueError('bounds')
result=[]
for name in names:
 if not isinstance(name,str) or any(p in ('','.','..') for p in name.split('/')):raise ValueError('path')
 path=Path(root)/name;current=Path('/')
 for part in path.parts[1:]:
  current=current/part
  if current.is_symlink():raise ValueError('symlink path')
 try:fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 except FileNotFoundError:
  result.append({'path':name,'present':False});continue
 with os.fdopen(fd,'rb') as f:
  info=os.fstat(f.fileno())
  if not stat.S_ISREG(info.st_mode) or not 0<=info.st_size<=maximum:raise ValueError('file type/size')
  data=f.read(maximum+1)
  if len(data)!=info.st_size or len(data)>maximum:raise ValueError('changing file size')
 result.append({'path':name,'present':True,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data_b64':base64.b64encode(data).decode()})
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(',',':')))
'''


def observe_many(transport,selection,maximum,deadline):
    """Retain every present file before parsing its canonical JSON contents."""
    if type(selection) is not dict or not 1<=len(selection)<=16:raise EvidenceError('bounded metadata selection required')
    integer(maximum,1,65536,'metadata per-file bound')
    names=sorted(selection)
    for name in names:relative(name)
    destinations=[Path(selection[n]) for n in names]
    if len({p.resolve() for p in destinations})!=len(names):raise EvidenceError('distinct metadata destinations required')
    for p in destinations:
        if any(a.is_symlink() for a in [p,*p.parents]) or p.exists():raise EvidenceError('fresh confined metadata destination required')
    reply=io.BytesIO()
    try:
        transport.stream(['/usr/bin/python3','-c',REMOTE_OBSERVE,transport.profile['remote_root'],json.dumps(names),str(maximum)],
                         reply,len(names)*(4*((maximum+2)//3)+512)+len(json.dumps(names).encode()),deadline)
        values=parse_json(reply.getvalue(),canonical_required=False)
        if type(values) is not list or len(values)!=len(names):raise EvidenceError('metadata selection count differs')
        decoded=[]
        for name,value in zip(names,values):
            if type(value) is dict and value=={'path':name,'present':False} and value['present'] is False:
                decoded.append(None);continue
            fields(value,'path present bytes sha256 data_b64','bundled metadata bytes');require_digest(value['sha256'])
            integer(value['bytes'],0,maximum,'metadata bytes')
            if value['path']!=name or value['present'] is not True or type(value['data_b64']) is not str:raise EvidenceError('metadata selection differs')
            try:data=base64.b64decode(value['data_b64'],validate=True)
            except Exception:raise EvidenceError('invalid metadata encoding') from None
            if len(data)!=value['bytes'] or hashlib.sha256(data).hexdigest()!=value['sha256']:raise EvidenceError('metadata framing differs')
            decoded.append(data)
        for path,data in zip(destinations,decoded):
            if data is not None:
                fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(parent)
                finally:os.close(parent)
        return {name:None if data is None else read_json(path) for name,path,data in zip(names,destinations,decoded)}
    except Exception as error:
        try:
            from private_transport_diagnostics import bounded_bytes
            error.metadata_response_diagnostic=bounded_bytes(reply.getvalue())
        except Exception:pass  # preserve strict original transport/framing failure
        raise
