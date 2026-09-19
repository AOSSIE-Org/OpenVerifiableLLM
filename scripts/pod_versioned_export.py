"""Complete stable remote tree snapshots with immutable local byte reuse.

Every selected file is retained in every snapshot by a read-only hardlink to a
verified content object. Reusing bytes saves transfer/storage, never inventory,
hash or state checks. A peer tree remains a peer observation, not attestation.
"""
import os
from pathlib import Path
import uuid
from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,inventory,verify_inventory,write_json
from pod_job_client import tree


def regular_directory(path,*,fresh=False):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in [path,*path.parents]):raise EvidenceError('export directory symlink')
    path.mkdir(mode=0o700,parents=True,exist_ok=not fresh)
    return path


def retained_object(path,item):
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=item['bytes'] or file_hash(path)!=item['sha256']:
        raise EvidenceError('retained export object differs; preserve and refuse')


def export(transport,name,store,output,deadline,*,progress=None,whole_root=False,expected_files=None,maximum_bytes=None,maximum_uncached_bytes=None):
    store=regular_directory(store);output=regular_directory(output,fresh=True)
    if store==output or store in output.parents or output in store.parents:raise EvidenceError('object store and snapshots must be separate')
    files=tree(transport,name,deadline,whole_root=whole_root);write_json(output/'inventory.json',files)
    if expected_files is not None and files!=expected_files:
        raise EvidenceError('remote snapshot differs from independently selected file inventory')
    if (maximum_bytes is None)!=(maximum_uncached_bytes is None):raise EvidenceError('both export bounds required')
    if maximum_bytes is not None:
        from ovl_pipeline.schema import integer
        integer(maximum_bytes,0,2**40,'remaining logical export bytes')
        integer(maximum_uncached_bytes,0,2**40,'remaining uncached export bytes')
        if sum(f['bytes'] for f in files)>maximum_bytes:raise EvidenceError('logical export bound exceeded before transfer')
    prefix='' if whole_root else name+'/'
    target=regular_directory(output/'files');objects=regular_directory(store/'objects');incoming=regular_directory(store/'incoming')
    transfers=[];reused=[];missing=[];selected_hashes=set()
    for item in files:
        obj=confined(objects,item['sha256'])
        if obj.exists():retained_object(obj,item)
        elif item['sha256'] not in selected_hashes:
            missing.append(item);selected_hashes.add(item['sha256'])
    missing_bytes=sum(f["bytes"] for f in missing)
    if maximum_uncached_bytes is not None and missing_bytes>maximum_uncached_bytes:
        raise EvidenceError("uncached export bound exceeded before transfer")
    bulk_files=None
    if len(missing)>=16:
        from pod_bulk_export import receive
        attempt=incoming/uuid.uuid4().hex
        bulk=receive(transport,name,missing,attempt,deadline,progress=progress,whole_root=whole_root)
        bulk_files=Path(bulk['files_directory']);transfers.append(bulk)
    for item in files:
        obj=confined(objects,item['sha256'])
        if obj.exists():retained_object(obj,item);reused.append(item['path'])
        else:
            if bulk_files is not None:
                staged=confined(bulk_files,item['path'])
            else:
                attempt=regular_directory(incoming/uuid.uuid4().hex,fresh=True);staged=attempt/'verified'
                transfer=transport.get(prefix+item['path'],staged,{**item,'path':prefix+item['path']},deadline,
                    progress=None if progress is None else lambda counts,item=item:progress(digest({'profile':digest(transport.profile),'tree':name,'file':item}),counts,item['bytes']))
                transfers.append(transfer)
            retained_object(staged,item);staged.chmod(0o400)
            try:os.link(staged,obj,follow_symlinks=False)
            except FileExistsError:retained_object(obj,item)
            fd=os.open(objects,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:os.fsync(fd)
            finally:os.close(fd)
            # Keep both names; no unlink of partial or sole evidence is needed.
        destination=confined(target,item['path']);destination.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        os.link(obj,destination,follow_symlinks=False)
    verify_inventory(target,files)
    after=tree(transport,name,deadline,whole_root=whole_root);write_json(output/'after-inventory.json',after)
    if after!=files:raise EvidenceError('remote snapshot changed; preserve copies without completion')
    actual=inventory(target,[p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()])
    if actual!=files:raise EvidenceError('local snapshot does not preserve exact remote inventory')
    receipt={'schema':'ovl.offpod-versioned-tree-export.v1','result':'PASS','pod_id':transport.profile['pod_id'],
             'profile_sha256':digest(transport.profile),'root':name,'files':files,'transfers':transfers,'reused_paths':reused,
             'files_directory':str(target),'object_store':str(store),
             'scope':'complete stable selected peer tree rehashed off pod; read-only hardlinks share physical bytes',
             'independent_physical_copies':False,'numerical_verification':'NOT_RUN'}
    if maximum_bytes is not None:
        receipt.update(schema='ovl.offpod-versioned-tree-export.v2',bounds={'maximum_bytes':maximum_bytes,
            'maximum_uncached_bytes':maximum_uncached_bytes,'selected_missing_bytes':missing_bytes})
    write_json(output/'export.json',receipt);return receipt
