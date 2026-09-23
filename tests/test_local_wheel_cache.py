"""Local archive placement preserves full lock/member audits and fresh-copy bounds."""
from pathlib import Path
import os
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_runtime_setup as m
from ovl_pipeline.runtime_audit import wheel_manifest
from ovl_pipeline.canonical import EvidenceError
from test_runtime_audit import fixture


def test_copied_archives_produce_identical_complete_manifest(tmp_path):
    lock,source,_=fixture(tmp_path);cache=tmp_path/'cache'
    expected=wheel_manifest(lock,source);m.cache_wheels(source,cache)
    assert wheel_manifest(lock,cache)==expected
    for f in source.iterdir():
        assert f.read_bytes()==(cache/f.name).read_bytes()
        assert f.stat().st_ino!=(cache/f.name).stat().st_ino
    f=next(cache.iterdir());data=f.read_bytes();f.write_bytes(data[:-1]+bytes([data[-1]^1]))
    with pytest.raises(EvidenceError,match='hash/version'):wheel_manifest(lock,cache)


@pytest.mark.parametrize('fault',['source-symlink','extra-file','existing-cache','cache-symlink','missing-wheel','extra-wheel','oversize','nested-cache'])
def test_bad_copy_or_cache_is_not_accepted(tmp_path,fault):
    lock,source,_=fixture(tmp_path);cache=tmp_path/'cache';wheel=next(source.iterdir())
    if fault=='source-symlink':
        other=tmp_path/'original';wheel.rename(other);wheel.symlink_to(other)
    elif fault=='extra-file':(source/'unselected.txt').write_text('unselected')
    elif fault=='existing-cache':cache.mkdir();(cache/'sole-copy').write_bytes(b'preserve')
    elif fault=='cache-symlink':cache.symlink_to(source,target_is_directory=True)
    elif fault=='oversize':
        with wheel.open('r+b') as f:f.truncate(4*1024**3+1)
    elif fault=='nested-cache':cache=source/'cache'
    if fault in ('missing-wheel','extra-wheel'):
        m.cache_wheels(source,cache)
        if fault=='missing-wheel':next(cache.iterdir()).unlink()
        else:(cache/'extra-1.0-py3-none-any.whl').write_bytes(b'unselected')
        with pytest.raises(EvidenceError):wheel_manifest(lock,cache)
    else:
        with pytest.raises(ValueError):m.cache_wheels(source,cache)
    if fault=='existing-cache':assert (cache/'sole-copy').read_bytes()==b'preserve'


def test_input_replaced_after_size_admission_is_refused(tmp_path,monkeypatch):
    _,source,_=fixture(tmp_path);cache=tmp_path/'cache';wheel=next(source.iterdir());original=os.open
    def changed(path,flags,*a,**kw):
        if Path(path)==wheel:
            wheel.unlink();wheel.write_bytes(b'replaced')
        return original(path,flags,*a,**kw)
    monkeypatch.setattr(os,'open',changed)
    with pytest.raises(ValueError,match='changed before'):m.cache_wheels(source,cache)


def test_setup_and_audited_launch_select_the_same_local_cache(tmp_path,monkeypatch):
    runtime=tmp_path/'runtime';seen=[]
    def selected(*args,**kwargs):
        seen.append(kwargs);raise ValueError('stop after selection capture')
    monkeypatch.setattr(m,'selected',selected)
    with pytest.raises(ValueError,match='selection capture'):
        m.setup(tmp_path/'config','0'*64,tmp_path/'inputs',runtime,tmp_path/'setup-evidence')
    with pytest.raises(ValueError,match='selection capture'):
        m.audited(tmp_path/'config','0'*64,tmp_path/'inputs',runtime,tmp_path/'audit-evidence','module',[])
    assert seen==[{'wheel_cache':runtime/'wheels','populate_cache':True},{'wheel_cache':runtime/'wheels'}]


def test_fifo_replacement_cannot_block_before_identity_check(tmp_path,monkeypatch):
    _,source,_=fixture(tmp_path);cache=tmp_path/'cache';wheel=next(source.iterdir());original=os.open
    def changed(path,flags,*a,**kw):
        if Path(path)==wheel:
            assert flags & os.O_NONBLOCK
            wheel.unlink();os.mkfifo(wheel)
        return original(path,flags,*a,**kw)
    monkeypatch.setattr(os,'open',changed)
    with pytest.raises(ValueError,match='changed before'):m.cache_wheels(source,cache)


def real_helper_selection(tmp_path):
    import importlib.metadata
    import shutil
    import zipfile
    from ovl_pipeline.canonical import file_hash,inventory,write_json
    root=tmp_path/'inputs';root.mkdir();source=root/'source';source.mkdir();wheels=root/'wheels';wheels.mkdir()
    package=source/'src/ovl_pipeline';package.mkdir(parents=True);(package/'__init__.py').write_text('')
    original=Path(__file__).parents[1]/'src/ovl_pipeline'
    for name in ('canonical.py','runtime_audit.py'):shutil.copyfile(original/name,package/name)
    helper=[];rows=[]
    for name in ('packaging','rfc8785'):
        module=__import__(name);version=importlib.metadata.version(name);directory=Path(module.__file__).parent
        wheel=wheels/f'{name}-{version}-py3-none-any.whl';dist=f'{name}-{version}.dist-info'
        with zipfile.ZipFile(wheel,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(directory.rglob('*.py')):z.writestr(name+'/'+f.relative_to(directory).as_posix(),f.read_bytes())
            z.writestr(dist+'/METADATA',f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
            z.writestr(dist+'/WHEEL','Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
            z.writestr(dist+'/RECORD','')
        helper.append({'path':wheel.name,'sha256':file_hash(wheel)});rows.append(name+'=='+version+' --hash=sha256:'+file_hash(wheel)+'\n')
    extra=tmp_path/'example-fixture';extra.mkdir();_,example,_=fixture(extra);wheel=next(example.iterdir());shutil.copyfile(wheel,wheels/wheel.name)
    rows.append('example==1.0 --hash=sha256:'+file_hash(wheel)+'\n');(source/'lock').write_text(''.join(rows))
    archive=root/'python.tar.gz';archive.write_bytes(b'synthetic archive identity; selection does not execute it')
    value={'schema':'ovl.offline-runtime-setup.v1','source_root':'source','source_files':inventory(source,[f.relative_to(source).as_posix() for f in source.rglob('*') if f.is_file()]),
           'dependency_lock':'lock','interpreter_archive':'python.tar.gz','interpreter_sha256':file_hash(archive),'wheels':'wheels','bootstrap_wheels':helper}
    config=root/'config.json';write_json(config,value);return root,config,wheels


def test_fresh_isolated_selection_imports_cached_helpers_and_audits_cached_bytes(tmp_path):
    import json
    import subprocess
    from ovl_pipeline.canonical import file_hash
    root,config,wheels=real_helper_selection(tmp_path);runtime=tmp_path/'runtime';runtime.mkdir();cache=runtime/'wheels'
    script='''import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('selected_setup',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
if sys.argv[7]=='launch':m.audited(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),Path(sys.argv[5]).parent,Path(sys.argv[5]).parent.parent/'launch-evidence','ovl_pipeline.runtime_launch',['--inspect-current'])
v=m.selected(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),wheel_cache=Path(sys.argv[5]),populate_cache=sys.argv[6]=='yes')
import packaging,rfc8785
print(json.dumps({'wheels':str(v[2]),'origins':[packaging.__file__,rfc8785.__file__],'packages':len(v[4]['packages'])}))
'''
    def invoke(populate,action='select'):
        return subprocess.run([sys.executable,'-I','-S','-c',script,str(Path(m.__file__).resolve()),str(config),file_hash(config),str(root),str(cache),populate,action],capture_output=True,text=True,timeout=20)
    first=invoke('yes');assert first.returncode==0,first.stderr
    expected=json.loads(first.stdout);assert expected['wheels']==str(cache) and expected['packages']==3
    assert all(origin.startswith(str(cache)+'/') for origin in expected['origins'])
    wheels.rename(root/'original-wheels-disabled')
    again=invoke('no');assert again.returncode==0,again.stderr;assert json.loads(again.stdout)==expected
    numerical=next(cache.glob('example-*.whl'));numerical.write_bytes(b'changed cached numerical wheel')
    changed=invoke('no','launch');assert changed.returncode!=0 and 'hash/version' in changed.stderr and not changed.stdout


def test_setup_installer_and_inspection_receive_cached_archives(tmp_path,monkeypatch):
    """Routing check with synthetic installer/inspection; no installation credit."""
    from ovl_pipeline.canonical import file_hash
    import ovl_pipeline.python_origin as origin
    import ovl_pipeline.runtime_audit as audit
    import ovl_pipeline.runtime_launch as launcher
    root,config,wheels=real_helper_selection(tmp_path);runtime=tmp_path/'runtime';output=tmp_path/'output';calls=[]
    def extract(archive,pin,destination):
        target=destination/'python/bin/python3.12';target.parent.mkdir(parents=True);target.write_bytes(b'synthetic interpreter')
        return {'synthetic':True},{'synthetic':True}
    def execute(command,**kwargs):
        if 'venv' in command:
            (runtime/'venv/bin').mkdir(parents=True);(runtime/'venv/bin/python').write_bytes(b'synthetic venv')
        else:
            lines=(output/'offline-install.txt').read_text().splitlines()
            assert len(lines)==3 and all(line.startswith(str(runtime/'wheels')+'/') for line in lines)
            calls.append('install')
    def installed(manifest,paths):
        assert manifest==wheel_manifest(root/'source/lock',runtime/'wheels');calls.append('installed-audit')
        return {'synthetic':True}
    def inspect(lock,selected_wheels,*args,**kwargs):
        assert selected_wheels==runtime/'wheels';assert wheel_manifest(lock,selected_wheels)==wheel_manifest(lock,wheels)
        calls.append('inspection');return {'synthetic':True}
    monkeypatch.setattr(origin,'extract',extract);monkeypatch.setattr(audit,'verify_installed',installed);monkeypatch.setattr(launcher,'launch',inspect)
    monkeypatch.setattr(sys,'path',list(sys.path));monkeypatch.setattr(sys,'dont_write_bytecode',sys.dont_write_bytecode)
    result=m.setup(config,file_hash(config),root,runtime,output,execute=execute)
    assert result['result']=='PASS' and calls==['install','installed-audit','inspection']
