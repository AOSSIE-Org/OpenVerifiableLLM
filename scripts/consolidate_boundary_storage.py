"""Share checked immutable boundary bytes after successful public delivery.

This runs only after the publisher's existing full download/signature/state gates.
It preserves all logical evidence paths and creates no verification, progress,
export or physical-replica credit. Interrupted replacements are safely repeatable.
"""
import os
from pathlib import Path
import uuid
from ovl_pipeline.canonical import EvidenceError,confined,digest,file_hash,read_json,verify_inventory
from pod_job_client import save_once
from pod_versioned_export import regular_directory,retained_object


def consolidate(output,result,store):
    output=Path(output);store=regular_directory(store);objects=regular_directory(store/'objects')
    state=output/'boundaries'/f"boundary-{result['index']:05d}"
    if read_json(state/'complete.json')!=result:raise EvidenceError('boundary consolidation requires completed delivery')
    publication=output/'publications'/f"boundary-{result['index']:05d}"
    ack=read_json(publication/'ack.json')
    if (digest(ack)!=result['ack_sha256'] or ack['registration_sha256']!=result['registration_sha256']
        or ack['boundary_sha256']!=result['boundary_sha256']):
        raise EvidenceError('completed publication identity differs')
    files=ack['checkpoint_archive']['inventory'];download=ack['checkpoint_download']
    if (download['result']!='PASS' or download['files']!=files
        or download['revision']!=ack['checkpoint_archive']['revision']):
        raise EvidenceError('complete public download required before consolidation')
    selection={'schema':'ovl.boundary-storage-selection.v1','completion_sha256':digest(result),
               'object_store':str(store),'files':files}
    record=state/'storage-consolidation';record.mkdir(exist_ok=True)
    save_once(record/'selection.json',selection)
    done=record/'verification.json'
    expected={'schema':'ovl.boundary-storage-consolidation.v1','selection_sha256':digest(selection),'result':'PASS',
              'scope':'checked byte-identical hardlinks; all paths retained; no new verification or health credit',
              'independent_physical_copies':False}
    if done.exists():
        if read_json(done)!=expected:raise EvidenceError('storage consolidation receipt differs')
        return expected
    snap=state/'snapshot'
    if not(snap/'export.json').exists():snap=state/'snapshot-retry'
    exported=read_json(snap/'export.json')
    if (exported['boundary_sha256']!=result['boundary_sha256'] or exported['files']!=files
        or exported['registration_sha256']!=result['registration_sha256']):
        raise EvidenceError('snapshot differs from completed public checkpoint')
    checkpoint=confined(snap,exported['checkpoint_path']);verify_inventory(checkpoint,files)
    matches=[]
    for candidate in sorted((publication/'checkpoint').glob('download-*')):
        proof=candidate/'verification.json'
        if proof.is_file() and read_json(proof)==download:matches.append(candidate)
    if len(matches)!=1:raise EvidenceError('unique original successful fresh public download required')
    fresh=matches[0];target=fresh/'downloaded';cache=fresh/'transport-cache'
    if read_json(fresh/'intent.json')['force_download'] is not True:
        raise EvidenceError('public checkpoint was not freshly downloaded')
    verify_inventory(target,files)
    # Cache symlinks are transport metadata only. Select their regular targets by
    # complete byte identity with the checked downloaded files, never by a hash
    # claimed in a filename. This also recovers partially replaced hardlinks. Never traverse outside this one fresh download cache.
    for directory in (checkpoint,target,cache):
        if directory.is_symlink() or any(p.is_symlink() for p in directory.parents):raise EvidenceError('storage input directory symlink')
    cached=[]
    for p in cache.rglob('*'):
        if p.is_symlink():
            if not p.resolve(strict=True).is_relative_to(cache.resolve()):raise EvidenceError('download cache symlink escape')
        elif p.is_file():cached.append(p)
        elif not p.is_dir():raise EvidenceError('nonregular download cache entry')
    cached_bytes={p:(p.stat().st_size,file_hash(p)) for p in cached}
    groups=[]
    for item in files:
        source=confined(target,item['path'])
        aliases=[p for p in cached if cached_bytes[p]==(item['bytes'],item['sha256'])]
        obj=confined(objects,item['sha256'])
        paths=[confined(checkpoint,item['path']),source,*aliases]
        if obj.exists():retained_object(obj,item)
        # Complete preflight precedes every replacement; preserve original bytes
        # on any discrepancy, including an interrupted earlier consolidation.
        for p in paths:retained_object(p,item)
        if any(p.stat().st_dev!=objects.stat().st_dev for p in paths):raise EvidenceError('consolidation requires one filesystem')
        groups.append((item,obj,paths))
    for item,obj,paths in groups:
        if not obj.exists():
            paths[0].chmod(0o400)
            try:os.link(paths[0],obj,follow_symlinks=False)
            except FileExistsError:pass
        retained_object(obj,item);obj.chmod(0o400)
        for p in paths:
            retained_object(p,item)
            if p.samefile(obj):continue
            temporary=p.with_name('.consolidating-'+uuid.uuid4().hex)
            os.link(obj,temporary,follow_symlinks=False)
            os.replace(temporary,p)
            fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:os.fsync(fd)
            finally:os.close(fd)
            retained_object(p,item)
    verify_inventory(checkpoint,files);verify_inventory(target,files)
    save_once(done,expected);return expected
