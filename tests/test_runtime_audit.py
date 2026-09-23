from pathlib import Path
import hashlib
import zipfile
import pytest
from ovl_pipeline.canonical import EvidenceError,file_hash
from ovl_pipeline.runtime_audit import locked_requirements,wheel_manifest,verify_installed


def fixture(tmp_path,*,extra=None):
    wheel_dir=tmp_path/'wheels';wheel_dir.mkdir();wheel=wheel_dir/'example-1.0-py3-none-any.whl'
    files={'example/__init__.py':b'VALUE = 3\n','example/payload.bin':b'actual binary data',
           'example-1.0.dist-info/METADATA':b'Metadata-Version: 2.1\nName: example\nVersion: 1.0\n',
           'example-1.0.dist-info/WHEEL':b'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
           'example-1.0.dist-info/RECORD':b'rewritten-on-installation'}
    files.update(extra or {})
    with zipfile.ZipFile(wheel,'w') as z:
        for n,b in files.items():z.writestr(n,b)
    lock=tmp_path/'lock.txt';lock.write_text('example==1.0 \\\n    --hash=sha256:'+file_hash(wheel)+'\n')
    paths={k:tmp_path/k for k in ('site','prefix','scripts','headers')}
    for p in paths.values():p.mkdir()
    for n,b in files.items():
        path=paths['site']/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    return lock,wheel_dir,paths


def test_rebuilds_from_archive_and_compares_actual_installed_bytes(tmp_path):
    lock,wheels,paths=fixture(tmp_path);m=wheel_manifest(lock,wheels)
    result=verify_installed(m,paths)
    assert result['result']=='PASS' and result['payload_count']==4
    assert result['production_admission']=='NOT_RUN'
    assert result['generated_files'][0]['path'].endswith('/RECORD')


def test_altered_payload_and_forged_installed_record_fail(tmp_path):
    lock,wheels,paths=fixture(tmp_path);m=wheel_manifest(lock,wheels)
    (paths['site']/'example/__init__.py').write_bytes(b'VALUE = 4\n')
    (paths['site']/'example-1.0.dist-info/RECORD').write_text('attacker fabricated matching record')
    with pytest.raises(EvidenceError,match='differs'):verify_installed(m,paths)


@pytest.mark.parametrize('name',['shadow.py','example/extra.so','example/__init__.pyc','_virtualenv.py'])
def test_unknown_importable_code_is_rejected(tmp_path,name):
    lock,wheels,paths=fixture(tmp_path);m=wheel_manifest(lock,wheels);p=paths['site']/name;p.write_bytes(b'unregistered')
    with pytest.raises(EvidenceError,match='unregistered'):verify_installed(m,paths)


def test_installer_hook_requires_separately_pinned_exact_bytes(tmp_path):
    lock,wheels,paths=fixture(tmp_path);m=wheel_manifest(lock,wheels);p=paths['site']/'_virtualenv.py';p.write_bytes(b'known installer hook')
    assert verify_installed(m,paths,allowed_generated={'_virtualenv.py':file_hash(p)})['result']=='PASS'
    p.write_bytes(b'changed')
    with pytest.raises(EvidenceError):verify_installed(m,paths,allowed_generated={'_virtualenv.py':'a'*64})


def test_bytecode_inventory_does_not_claim_execution_exclusion(tmp_path):
    lock,wheels,paths=fixture(tmp_path);m=wheel_manifest(lock,wheels);p=paths['site']/'example/__pycache__/__init__.cpython-312.pyc';p.parent.mkdir();p.write_bytes(b'not executed by this audit')
    r=verify_installed(m,paths)
    assert r['ignored_installer_bytecode']==['example/__pycache__/__init__.cpython-312.pyc']
    assert r['bytecode_execution_exclusion']=='REQUIRED_BY_LAUNCHER_NOT_VERIFIED_HERE'


def test_lock_omission_wrong_hash_or_version_fail(tmp_path):
    lock,wheels,paths=fixture(tmp_path);text=lock.read_text()
    lock.write_text(text+'absent==1.0 --hash=sha256:'+'a'*64+'\n')
    with pytest.raises(EvidenceError,match='omits'):wheel_manifest(lock,wheels)
    lock.write_text(text.replace('example==1.0','example==2.0'))
    with pytest.raises(EvidenceError,match='hash/version'):wheel_manifest(lock,wheels)
    lock.write_text('example==1.0 --hash=sha256:'+'a'*64+'\n')
    with pytest.raises(EvidenceError,match='hash/version'):wheel_manifest(lock,wheels)


def test_symlink_payload_and_archive_refused(tmp_path):
    lock,wheels,paths=fixture(tmp_path);m=wheel_manifest(lock,wheels);p=paths['site']/'example/payload.bin';other=tmp_path/'other';p.rename(other);p.symlink_to(other)
    with pytest.raises(EvidenceError,match='symlink'):verify_installed(m,paths)
    wheel=next(wheels.iterdir());other=tmp_path/wheel.name;wheel.rename(other);wheel.symlink_to(other)
    with pytest.raises(EvidenceError,match='regular wheels'):wheel_manifest(lock,wheels)


def test_directory_traversal_is_never_extracted(tmp_path):
    lock,wheels,paths=fixture(tmp_path,extra={'../escape':b'not allowed'})
    with pytest.raises(EvidenceError,match='escaping'):wheel_manifest(lock,wheels)


def test_duplicate_archive_member_rejected(tmp_path):
    lock,wheels,paths=fixture(tmp_path);wheel=next(wheels.iterdir())
    with pytest.warns(UserWarning,match='Duplicate name'):
        with zipfile.ZipFile(wheel,'a') as z:z.writestr('example/payload.bin',b'ambiguous')
    lock.write_text('example==1.0 --hash=sha256:'+file_hash(wheel)+'\n')
    with pytest.raises(EvidenceError,match='duplicate wheel'):wheel_manifest(lock,wheels)


def test_direct_url_is_still_bound_to_selected_archive_hash(tmp_path):
    lock,wheels,paths=fixture(tmp_path);wheel=next(wheels.iterdir())
    lock.write_text('example @ https://example.invalid/'+wheel.name+' --hash=sha256:'+file_hash(wheel)+'\n')
    assert wheel_manifest(lock,wheels)['packages'][0]['name']=='example'


@pytest.mark.parametrize('line',['example>=1.0','example==1.*','example[extra]==1.0','example==1.0 ; python_version>="3.12"'])
def test_unfrozen_dependency_forms_fail(tmp_path,line):
    p=tmp_path/'lock';p.write_text(line+' --hash=sha256:'+'a'*64+'\n')
    with pytest.raises(EvidenceError):locked_requirements(p)


@pytest.mark.parametrize('name',['../escape','/absolute','a/../b','a/./b','a//b','a\\b','a\x00b','','.','a/'])
def test_virtual_archive_path_validation_rejects_escaping_or_ambiguous_names(name):
    from ovl_pipeline.runtime_audit import archive_name
    with pytest.raises(EvidenceError):archive_name(name)


def test_virtual_members_do_not_consult_unrelated_working_directory(tmp_path,monkeypatch):
    lock,wheels,paths=fixture(tmp_path);expected=wheel_manifest(lock,wheels)
    cwd=tmp_path/'unrelated';cwd.mkdir();(cwd/'example').symlink_to(tmp_path/'absent')
    monkeypatch.chdir(cwd)
    import ovl_pipeline.runtime_audit as module
    def no_real_path_lookup(*args,**kwargs):raise AssertionError('virtual member consulted real filesystem')
    monkeypatch.setattr(module,'confined',no_real_path_lookup)
    assert wheel_manifest(lock,wheels)==expected
