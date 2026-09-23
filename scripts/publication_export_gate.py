"""Exact local review plus installed export scan before an SDK publication.

No approval is generated here. A missing review can wait only inside the caller's
original deadline. Local requests, reviews and scan receipts never enter exports.
"""
import json
import hashlib
from contextlib import contextmanager,ExitStack
import tempfile
import os
from pathlib import Path
import stat
import subprocess
import time

from ovl_pipeline.canonical import EvidenceError,digest,file_hash,read_json,verify_inventory,write_json
from ovl_pipeline.schema import fields


def regular(path):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in [path,*path.parents]):
        raise EvidenceError('publication review path contains symlink')
    return path


def exact_tree(staging,entries):
    staging=regular(staging);actual=[]
    for p in staging.rglob('*'):
        s=p.lstat()
        if stat.S_ISDIR(s.st_mode):continue
        if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:
            raise EvidenceError('publication staging requires unshared regular files')
        actual.append(p.relative_to(staging).as_posix())
    if sorted(actual)!=[e['path'] for e in entries]:
        raise EvidenceError('reviewed publication inventory changed')
    verify_inventory(staging,entries)


def private_review(path):
    regular(path);s=path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid() or s.st_nlink!=1 or s.st_mode&0o077 or s.st_size>1024**2:
        raise EvidenceError('review must be a bounded owner-private regular file')


def require_review(plan_path,staging,*,deadline=None,monotonic_deadline=None,execute=subprocess.run,
                   wall=time.time,monotonic=time.monotonic,sleep=time.sleep):
    """Return a private gate receipt; caller still verifies actual public bytes."""
    plan_path=regular(plan_path);staging=regular(staging);plan=read_json(plan_path)
    if deadline is not None and (type(deadline) is not int or deadline<=0):
        raise EvidenceError('invalid original publication deadline')
    limit=monotonic()+max(0,deadline-wall()) if deadline is not None else monotonic()+600
    if monotonic_deadline is not None:limit=min(limit,monotonic_deadline)
    def remaining():
        value=min(limit-monotonic(),deadline-wall()) if deadline is not None else limit-monotonic()
        if value<=0:raise EvidenceError('original publication review deadline expired')
        return value
    request={'schema':'ovl.local-export-review-request.v1','plan_sha256':digest(plan),
             'destination':'https://huggingface.co/datasets/'+plan['repo'],
             'prefix':plan['prefix'],'staging':str(staging),'files':plan['files']}
    request_path=plan_path.with_name(plan_path.name+'.privacy-request.json')
    review_path=plan_path.with_name(plan_path.name+'.privacy-review.json')
    for p in (request_path,review_path):
        regular(p)
        if p.is_relative_to(staging):raise EvidenceError('private review overlaps publication staging')
    if request_path.exists():
        if read_json(request_path)!=request:raise EvidenceError('retained export review request differs')
    else:
        write_json(request_path,request)
        print(json.dumps({'event':'publication-review-required','request_sha256':digest(request),'plan':str(plan_path)}),flush=True)
    while not review_path.exists():
        remaining()
        if deadline is None:raise EvidenceError('exact semantic publication review required')
        sleep(min(2,remaining()))
    remaining();private_review(review_path)
    review=read_json(review_path)
    fields(review,'schema request_sha256 result repository_context content_review','local export review')
    if (review['schema']!='ovl.local-export-review.v1' or review['request_sha256']!=digest(request)
        or review['result']!='PASS' or type(review['content_review']) is not str or not review['content_review'].strip()):
        raise EvidenceError('missing or mismatched semantic publication review')
    # The context selects exact repository/path/rule exceptions in the installed
    # guard. It must represent this destination, never another repository.
    context=regular(review['repository_context'])
    result=execute(['git','-C',str(context),'remote','get-url','--all','origin'],
                   capture_output=True,text=True,check=True,timeout=remaining())
    if result.stdout.strip()!=request['destination']:
        raise EvidenceError('privacy review repository destination differs')
    exact_tree(staging,plan['files']);remaining()
    try:
        result=execute(['publication-privacy','export',str(staging),'--repository',str(context)],
                       capture_output=True,text=True,check=True,timeout=remaining())
        scan=json.loads(result.stdout)
    except (subprocess.SubprocessError,OSError,ValueError) as error:
        raise EvidenceError('installed publication privacy scan failed: '+type(error).__name__) from None
    remaining()
    if (scan.get('schema')!='local.privacy-scan.v1' or scan.get('result')!='PASS'
        or scan.get('files')!=plan['files']):
        raise EvidenceError('privacy scan did not verify the exact export inventory')
    private_review(review_path)
    if read_json(review_path)!=review or read_json(plan_path)!=plan:
        raise EvidenceError('publication selection changed during review')
    exact_tree(staging,plan['files']);remaining()
    receipt={'schema':'ovl.local-export-gate.v1','result':'PASS','request_sha256':digest(request),
             'review_sha256':file_hash(review_path),'plan_sha256':digest(plan),'scan':scan,'deadline_epoch':deadline,
             'scope':'Reviewed exact export and installed scanner only; no scientific acceptance'}
    write_json(plan_path.with_name(plan_path.name+'.privacy-scan.json'),receipt)
    return receipt


@contextmanager
def frozen_payloads(staging,entries,temporary_directory,check_deadline):
    """Unlinked private copies with read-only handles for SDK consumption.

    The scanner/reviewer bind these exact hashes. Source-path changes cannot
    change payloads while the SDK reads them. Original evidence stays intact.
    This guards ordinary concurrent edits, not a hostile privileged local host.
    """
    with ExitStack() as stack:
        frozen=[]
        for entry in entries:
            check_deadline();source=regular(Path(staging)/entry['path'])
            fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as inp:
                st=os.fstat(inp.fileno())
                if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_size!=entry['bytes']:
                    raise EvidenceError('reviewed source changed before immutable upload copy')
                with tempfile.TemporaryFile(dir=temporary_directory) as copy:
                    hashed=hashlib.sha256();size=0
                    while block:=inp.read(4*1024**2):
                        check_deadline();size+=len(block)
                        if size>entry['bytes']:raise EvidenceError('upload copy exceeded reviewed size')
                        hashed.update(block);copy.write(block)
                    if size!=entry['bytes'] or hashed.hexdigest()!=entry['sha256']:
                        raise EvidenceError('upload copy differs from reviewed bytes')
                    copy.flush();os.fsync(copy.fileno());check_deadline()
                    # Reopen read-only before the sole writable handle closes.
                    # The temporary inode has no directory name for a later edit.
                    handle=stack.enter_context(os.fdopen(os.open('/proc/self/fd/'+str(copy.fileno()),os.O_RDONLY),'rb'))
                frozen.append((entry,handle))
        yield frozen
