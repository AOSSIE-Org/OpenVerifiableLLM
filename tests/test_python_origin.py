"""Public archive payload identity with synthetic safe/malicious tar members."""
from pathlib import Path
import io
import tarfile
import pytest
from ovl_pipeline import python_origin as m
from ovl_pipeline.canonical import EvidenceError,file_hash


def archive(tmp_path,mutation=None):
    path=tmp_path/'python.tar.gz'
    items=[('python/bin/python3.12',b'public interpreter'),('python/lib/python3.12/example.py',b'PUBLIC=True\n'),
           ('python/lib/python3.12/__pycache__/example.cpython-312.pyc',b'ignored compiler bytes')]
    with tarfile.open(path,'w:gz') as t:
        for n,b in items:
            x=tarfile.TarInfo(n);x.mode=0o755 if '/bin/' in n else 0o644;x.size=len(b);t.addfile(x,io.BytesIO(b))
        x=tarfile.TarInfo('python/bin/python');x.type=tarfile.SYMTYPE;x.linkname='python3.12';t.addfile(x)
        if mutation:
            x=tarfile.TarInfo('python/extra');x.size=1
            if mutation=='traversal':x.name='python/../escape'
            elif mutation=='absolute':x.name='/escape'
            elif mutation=='duplicate':x.name=items[0][0]
            elif mutation=='parent-file':x.name='python/bin/python3.12/child'
            elif mutation=='external-link':x.type=tarfile.SYMTYPE;x.linkname='../../outside'
            elif mutation=='cyclic-link':x.type=tarfile.SYMTYPE;x.linkname='extra'
            elif mutation=='hardlink':x.type=tarfile.LNKTYPE;x.linkname='python/bin/python3.12'
            elif mutation=='setuid':x.mode=0o4755
            t.addfile(x,io.BytesIO(b'x') if x.isfile() else None)
    return path,file_hash(path)


def test_unmodified_extraction_and_generated_cache_exclusion(tmp_path):
    tar,root=archive(tmp_path);out=tmp_path/'installation';manifest,checked=m.extract(tar,root,out)
    assert checked['regular_payload_files_checked']==2 and (out/'python/bin/python').resolve()==out/'python/bin/python3.12'
    p=out/'python/lib/python3.12/__pycache__/example.cpython-312.pyc';p.write_bytes(b'changed pycache excluded by launcher')
    assert m.audit(manifest,out)['result']=='PASS'
    with pytest.raises(EvidenceError,match='fresh'):m.extract(tar,root,out)


@pytest.mark.parametrize('mutation',['traversal','absolute','duplicate','parent-file','external-link','cyclic-link','hardlink','setuid'])
def test_invalid_archive_rejected_before_any_extraction(tmp_path,mutation):
    tar,root=archive(tmp_path,mutation);out=tmp_path/'installation'
    with pytest.raises(EvidenceError):m.extract(tar,root,out)
    assert not out.exists()


@pytest.mark.parametrize('mutation',['interpreter','stdlib','extra-code','orphan-bytecode','missing','changed-link','mode'])
def test_modified_or_extra_installed_payload_fails_closed(tmp_path,mutation):
    tar,root=archive(tmp_path);out=tmp_path/'installation';manifest,_=m.extract(tar,root,out)
    if mutation=='interpreter':(out/'python/bin/python3.12').write_bytes(b'changed')
    elif mutation=='stdlib':(out/'python/lib/python3.12/example.py').write_bytes(b'changed')
    elif mutation=='extra-code':(out/'python/lib/python3.12/evil.py').write_bytes(b'changed')
    elif mutation=='orphan-bytecode':(out/'python/lib/python3.12/evil.pyc').write_bytes(b'changed')
    elif mutation=='missing':(out/'python/lib/python3.12/example.py').unlink()
    elif mutation=='changed-link':
        p=out/'python/bin/python';p.unlink();p.symlink_to('/usr/bin/python3')
    else:(out/'python/bin/python3.12').chmod(0o644)
    with pytest.raises(EvidenceError):m.audit(manifest,out)


def test_archive_digest_is_external_not_self_described(tmp_path):
    tar,root=archive(tmp_path)
    with pytest.raises(EvidenceError,match='external selection'):m.manifest(tar,'0'*64)
