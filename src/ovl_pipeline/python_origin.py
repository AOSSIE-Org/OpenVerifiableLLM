"""Reconstruct and audit an unmodified public install-only Python distribution.

The separately selected archive digest is the trust input. This proves installed
payload identity to those public binary bytes, not a reproducible CPython build.
The trusted verifier must run outside the target installation. Generated PEP3147
bytecode is reported separately and must be excluded by the constrained launcher.
"""
import argparse
import hashlib
import os
from pathlib import Path,PurePosixPath
import posixpath
import tarfile

from .canonical import EvidenceError,digest,file_hash,require_digest,write_json

MAX_BYTES=1024**3


def name(value):
    if type(value) is not str or not value or len(value)>4096 or '\\' in value or '\x00' in value:raise EvidenceError('invalid Python archive path')
    p=PurePosixPath(value)
    if p.is_absolute() or len(p.parts)>32 or '..' in p.parts or '.' in value.split('/') or str(p)!=value or p.parts[0]!='python':
        raise EvidenceError('Python archive must have one confined python/ root')
    return p


def manifest(archive,expected_sha256):
    require_digest(expected_sha256)
    if archive.is_symlink() or not archive.is_file() or archive.stat().st_size>MAX_BYTES or file_hash(archive)!=expected_sha256:
        raise EvidenceError('Python archive differs from external selection')
    entries={};total=0
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            p=name(m.name)
            if m.name in entries or len(entries)>=15000:raise EvidenceError('duplicate/oversized Python archive inventory')
            if m.mode&0o7000:raise EvidenceError('privileged archive permissions refused')
            entry={'path':m.name,'mode':m.mode&0o777}
            if m.isfile():
                total+=m.size
                if not 0<=m.size<=128*1024**2 or total>MAX_BYTES:raise EvidenceError('Python archive expanded size exceeds bound')
                h=hashlib.sha256();n=0
                with tar.extractfile(m) as f:
                    while chunk:=f.read(1024*1024):h.update(chunk);n+=len(chunk)
                if n!=m.size:raise EvidenceError('truncated Python archive member')
                entry.update(kind='file',bytes=n,sha256=h.hexdigest())
            elif m.isdir():entry.update(kind='directory')
            elif m.issym():
                target=m.linkname
                if not target or '\\' in target or '\x00' in target or PurePosixPath(target).is_absolute():
                    raise EvidenceError('unsafe Python archive symlink')
                resolved=posixpath.normpath(str(p.parent/target));name(resolved)
                entry.update(kind='symlink',target=target,resolved=resolved)
            else:raise EvidenceError('unsupported Python archive member type')
            entries[m.name]=entry
    # Install-only distributions may omit all directory records. Derive only
    # the parents of actual members, while retaining collision rejection.
    for key in list(entries):
        for parent in PurePosixPath(key).parents:
            if str(parent)=='.':break
            entries.setdefault(str(parent),{'path':str(parent),'mode':0o755,'kind':'directory'})
    if entries.get('python',{}).get('kind')!='directory':raise EvidenceError('missing Python distribution root')
    for key,e in entries.items():
        for parent in PurePosixPath(key).parents:
            if str(parent)=='.':break
            if entries.get(str(parent),{}).get('kind')!='directory':raise EvidenceError('archive parent is not a declared regular directory')
        if e['kind']=='symlink':
            target=e['resolved'];seen={key}
            while entries.get(target,{}).get('kind')=='symlink':
                if target in seen:raise EvidenceError('cyclic Python archive symlink')
                seen.add(target);target=entries[target]['resolved']
            if entries.get(target,{}).get('kind')!='file':raise EvidenceError('Python archive symlink must terminate at a regular member')
    return {'schema':'ovl.python-archive-payloads.v1','archive_sha256':expected_sha256,
            'files':sorted(entries.values(),key=lambda e:e['path']),'expanded_bytes':total,
            'scope':'public binary distribution inventory; reproducible source build not established'}


def bytecode(path):
    p=PurePosixPath(path);return '__pycache__' in p.parts and p.suffix=='.pyc'


def audit(value,root):
    """Full installed source/binary comparison; never import target code."""
    if root.is_symlink() or not root.is_dir():raise EvidenceError('regular Python distribution root required')
    expected={e['path']:e for e in value['files']};observed=set();ignored=[];checked=0;total=0
    # root is the extraction parent and must contain exactly python/.
    for parent,dirs,files in os.walk(root,followlinks=False):
        for item in [*dirs,*files]:
            path=Path(parent)/item;relative=path.relative_to(root).as_posix();e=expected.get(relative);observed.add(relative)
            if e is None:
                if path.is_file() and not path.is_symlink() and bytecode(relative):ignored.append(relative);continue
                if path.is_dir() and not path.is_symlink() and PurePosixPath(relative).name=='__pycache__':continue
                raise EvidenceError('unexpected installed Python payload: '+relative)
            if e['kind']=='symlink':
                if not path.is_symlink() or os.readlink(path)!=e['target']:raise EvidenceError('installed Python symlink differs')
                if not path.resolve(strict=True).is_relative_to((root/'python').resolve()):raise EvidenceError('installed Python link escapes distribution')
            elif path.is_symlink():raise EvidenceError('installed Python payload unexpectedly symlinked')
            elif e['kind']=='directory':
                if not path.is_dir():raise EvidenceError('installed Python directory differs')
            else:
                if not path.is_file():raise EvidenceError('missing installed Python regular payload')
                if bytecode(relative):ignored.append(relative);continue
                if path.stat().st_size!=e['bytes'] or file_hash(path)!=e['sha256']:raise EvidenceError('installed Python bytes differ: '+relative)
                if bool(path.stat().st_mode&0o111)!=bool(e['mode']&0o111):raise EvidenceError('installed Python executable mode differs')
                checked+=1;total+=e['bytes']
    if set(expected)-observed:raise EvidenceError('missing installed Python archive members')
    return {'schema':'ovl.installed-python-audit.v1','result':'PASS','archive_manifest_sha256':digest(value),
            'archive_sha256':value['archive_sha256'],'regular_payload_files_checked':checked,'regular_payload_bytes_checked':total,
            'ignored_bytecode':sorted(ignored),'bytecode_execution_exclusion':'REQUIRED_BY_LAUNCHER_NOT_VERIFIED_HERE',
            'scope':'full installed non-bytecode payload and symlink identity to externally selected archive; not hardware attestation'}


def extract(archive,expected_sha256,output):
    if output.exists():raise EvidenceError('Python extraction output must be fresh')
    value=manifest(archive,expected_sha256);output.mkdir(parents=True,exist_ok=False)
    entries={e['path']:e for e in value['files']}
    for e in value['files']:
        if e['kind']=='directory':(output/e['path']).mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        for m in tar:
            if not m.isfile():continue
            e=entries[m.name];path=output/m.name
            with tar.extractfile(m) as source,path.open('xb') as dest:
                while chunk:=source.read(1024*1024):dest.write(chunk)
                dest.flush();os.fsync(dest.fileno())
            path.chmod(e['mode'])
    for e in value['files']:
        if e['kind']=='symlink':(output/e['path']).symlink_to(e['target'])
    return value,audit(value,output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['extract','audit'])
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    if a.evidence.exists():p.exit(1,'preserve existing Python audit evidence\n')
    try:
        if a.action=='extract':value,result=extract(a.archive,a.sha256,a.root)
        else:value=manifest(a.archive,a.sha256);result=audit(value,a.root)
        a.evidence.mkdir(parents=True,exist_ok=False);write_json(a.evidence/'manifest.json',value);write_json(a.evidence/'audit.json',result)
        print('Python payload identity PASS '+digest(value))
    except Exception as error:p.exit(1,'Python audit refused: '+str(error)+'\n')

if __name__=='__main__':main()
