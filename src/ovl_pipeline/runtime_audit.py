"""Compare installed package payloads with the exact hash-locked public wheels.

This is a package-file audit, not an attestation of the container, host, interpreter
or executed machine code. It deliberately does not trust installed RECORD hashes.
The closed inventory is rebuilt from every selected wheel's actual member bytes.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import re
import stat
import sysconfig
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name,parse_wheel_filename

from .canonical import EvidenceError,confined,digest,file_hash,read_json,write_json


def locked_requirements(path):
    text=path.read_text();rows=[];pending=''
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith('#') or line.startswith('--index-url ') or line.startswith('--extra-index-url '):continue
        pending+=line[:-1]+' ' if line.endswith('\\') else line
        if line.endswith('\\'):continue
        rows.append(pending);pending=''
    if pending:raise EvidenceError('unfinished dependency lock line')
    result={}
    for line in rows:
        hashes=re.findall(r' --hash=sha256:([0-9a-f]{64})(?= |$)',line)
        requirement=Requirement(re.sub(r' --hash=sha256:[0-9a-f]{64}(?= |$)','',line).strip())
        if requirement.marker or requirement.extras or not hashes:raise EvidenceError('only complete unconditional hashed lock supported')
        name=canonicalize_name(requirement.name)
        if name in result:raise EvidenceError('duplicate locked package')
        if requirement.url:
            if not requirement.url.startswith('https://') or not requirement.url.split('#')[0].endswith('.whl'):
                raise EvidenceError('only HTTPS wheel direct dependencies supported')
            version=None
        else:
            specs=list(requirement.specifier)
            if len(specs)!=1 or specs[0].operator!='==' or '*' in specs[0].version:raise EvidenceError('exact pinned package version required')
            version=specs[0].version
        result[name]={'version':version,'hashes':set(hashes),'url':requirement.url}
    if not result:raise EvidenceError('empty dependency lock')
    return result


def wheel_manifest(lock,wheels):
    """Fully hash archives and stream every member; never extract archive paths."""
    locked=locked_requirements(lock);packages=[];payloads={};seen=set();dist_infos=[]
    candidates=sorted(wheels.iterdir())
    if not candidates:raise EvidenceError('missing wheel inventory')
    for wheel in candidates:
        if wheel.is_symlink() or not wheel.is_file() or wheel.suffix!='.whl':raise EvidenceError('wheel directory must contain only regular wheels')
        name,version,_,_=parse_wheel_filename(wheel.name);name=canonicalize_name(name)
        if name in seen or name not in locked:raise EvidenceError('duplicate/unlocked wheel package')
        seen.add(name);pin=locked[name];sha=file_hash(wheel)
        if sha not in pin['hashes'] or (pin['version'] is not None and str(version)!=pin['version']):raise EvidenceError('wheel differs from hash/version lock')
        with zipfile.ZipFile(wheel) as z:
            infos=z.infolist();names=[i.filename for i in infos]
            if len(infos)>500000 or sum(i.file_size for i in infos)>128*1024**3:
                raise EvidenceError('wheel member inventory exceeds audit bounds')
            if len(names)!=len(set(names)):raise EvidenceError('duplicate wheel member')
            dist=[n[:-len('/METADATA')] for n in names if re.fullmatch(r'[^/]+\.dist-info/METADATA',n)]
            if len(dist)!=1:raise EvidenceError('wheel needs one metadata directory')
            info_root=dist[0];dist_infos.append(info_root)
            if info_root+'/WHEEL' not in names or info_root+'/RECORD' not in names:raise EvidenceError('incomplete wheel metadata')
            if z.getinfo(info_root+'/METADATA').file_size>16*1024**2:raise EvidenceError('oversized wheel metadata')
            from email.parser import BytesParser
            meta=BytesParser().parsebytes(z.read(info_root+'/METADATA'))
            if canonicalize_name(meta['Name'])!=name or meta['Version']!=str(version):raise EvidenceError('wheel metadata identity differs')
            data_root=info_root[:-len('.dist-info')]+'.data/'
            for member in infos:
                n=member.filename
                confined(Path('.'),n.rstrip('/'))
                if member.is_dir():continue
                mode=(member.external_attr>>16)&0xffff
                if stat.S_IFMT(mode) not in (0,stat.S_IFREG):raise EvidenceError('nonregular wheel payload')
                if member.file_size>8*1024**3:raise EvidenceError('oversized wheel member')
                # RECORD is legitimately rewritten on installation. The complete
                # wheel archive hash already authenticates its original bytes.
                if n==info_root+'/RECORD':continue
                scheme='site';target=n
                if n.startswith(data_root):
                    section,sep,target=n[len(data_root):].partition('/')
                    if not sep or section not in ('purelib','platlib','data','scripts','headers'):raise EvidenceError('unsupported wheel install scheme')
                    scheme={'purelib':'site','platlib':'site','data':'prefix','scripts':'scripts','headers':'headers'}[section]
                key=scheme+'/'+target
                if key in payloads:raise EvidenceError('overlapping wheel payloads')
                h=hashlib.sha256();length=0
                with z.open(member) as f:
                    for block in iter(lambda:f.read(4*1024**2),b''):h.update(block);length+=len(block)
                if length!=member.file_size:raise EvidenceError('wheel member length differs')
                payloads[key]={'scheme':scheme,'path':target,'bytes':length,'sha256':h.hexdigest()}
        packages.append({'name':name,'version':str(version),'wheel':wheel.name,'bytes':wheel.stat().st_size,'sha256':sha})
    if seen!=set(locked):raise EvidenceError('wheel inventory omits locked packages')
    return {'schema':'ovl.locked-wheel-payloads.v1','dependency_lock_sha256':file_hash(lock),
            'packages':sorted(packages,key=lambda p:p['name']),'dist_info_directories':sorted(dist_infos),
            'payloads':[payloads[k] for k in sorted(payloads)],
            'scope':'all selected wheel archive hashes and uncompressed payload bytes; not installed runtime yet'}


def verify_installed(manifest,paths,*,allowed_generated=None):
    """Rehash every installed payload and reject unregistered importable files.

    Caller selects the manifest produced from actual locked wheels. Installation
    metadata and compiler bytecode are inventoried separately, not trusted as
    original wheel bytes. The training launcher must use a fresh empty pycache
    prefix to prevent loading installed .pyc files, and exact selected sys.path.
    """
    if set(paths)!={'site','prefix','scripts','headers'}:raise EvidenceError('complete install scheme required')
    expected={};records=[]
    for e in manifest['payloads']:
        p=confined(paths[e['scheme']],e['path']);key=e['scheme']+'/'+e['path']
        if key in expected:raise EvidenceError('duplicate installed payload')
        expected[key]=e
        if not p.is_file() or p.stat().st_size!=e['bytes'] or file_hash(p)!=e['sha256']:
            raise EvidenceError('installed payload differs from locked wheel: '+key)
    # These generated files may differ between installers but cannot import code.
    generated_names={'RECORD','INSTALLER','REQUESTED','direct_url.json','uv_cache.json'}
    generated=[];ignored_bytecode=[]
    allowed_generated=allowed_generated or {}
    for p in paths['site'].rglob('*'):
        if p.is_symlink() or not(p.is_file() or p.is_dir()):raise EvidenceError('nonregular installed package path')
        if p.is_dir():continue
        name=p.relative_to(paths['site']).as_posix()
        if 'site/'+name in expected:continue
        parts=name.split('/')
        if p.suffix=='.pyc' and '__pycache__' in parts:
            ignored_bytecode.append(name);continue
        is_metadata=len(parts)==2 and parts[0] in manifest['dist_info_directories'] and parts[1] in generated_names
        if not is_metadata and (name not in allowed_generated or file_hash(p)!=allowed_generated[name]):
            raise EvidenceError('unregistered installed package file: '+name)
        generated.append({'path':name,'bytes':p.stat().st_size,'sha256':file_hash(p)})
    return {'schema':'ovl.installed-wheel-audit.v1','result':'PASS','wheel_manifest_sha256':digest(manifest),
            'dependency_lock_sha256':manifest['dependency_lock_sha256'],'payload_count':len(expected),
            'payload_bytes':sum(e['bytes'] for e in expected.values()),'generated_files':sorted(generated,key=lambda e:e['path']),
            'ignored_installer_bytecode':sorted(ignored_bytecode),'bytecode_execution_exclusion':'REQUIRED_BY_LAUNCHER_NOT_VERIFIED_HERE',
            'scope':'complete locked wheel payload equality and closed site-packages file inventory',
            'container_identity':'NOT_RUN','interpreter_stdlib_identity':'NOT_RUN','hardware_attestation':'NOT_RUN','production_admission':'NOT_RUN'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lock',required=True,type=Path);p.add_argument('--wheels',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--verify-current',action='store_true')
    p.add_argument('--allowed-generated',type=Path)
    a=p.parse_args()
    if a.output.exists():raise EvidenceError('audit requires fresh output directory')
    m=wheel_manifest(a.lock,a.wheels);a.output.mkdir(parents=True);write_json(a.output/'wheel-payloads.json',m)
    if a.verify_current:
        import sys
        site=Path(sysconfig.get_path('purelib'))
        if site!=Path(sysconfig.get_path('platlib')):raise EvidenceError('split pure/plat installation unsupported')
        result=verify_installed(m,{'site':site,'prefix':Path(sys.prefix),'scripts':Path(sysconfig.get_path('scripts')),
                                 'headers':Path(sysconfig.get_path('include'))},allowed_generated=read_json(a.allowed_generated) if a.allowed_generated else None)
        write_json(a.output/'installed-audit.json',result)

if __name__=='__main__':main()
