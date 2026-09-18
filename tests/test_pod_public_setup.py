"""Source archive confinement and ordered setup checks with explicit subprocess doubles."""
import hashlib,io,json,sys,tarfile,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_public_setup as m
from ovl_pipeline.canonical import file_hash,write_json


def archive(tmp_path,fault=None):
    data=b'print("selected")\n';name='src/module.py'
    files=[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]
    p=tmp_path/'source.tar.gz'
    with tarfile.open(p,'w:gz',format=tarfile.USTAR_FORMAT) as t:
        i=tarfile.TarInfo(name);i.size=len(data)
        if fault=='traversal':i.name='../outside'
        elif fault=='symlink':i.type=tarfile.SYMTYPE;i.linkname='../outside';i.size=0
        elif fault=='hardlink':i.type=tarfile.LNKTYPE;i.linkname='../outside';i.size=0
        elif fault=='size':i.size-=1
        elif fault=='bytes':data=b'x'*len(data)
        t.addfile(i,io.BytesIO(data))
        if fault=='extra':t.addfile(tarfile.TarInfo('unlisted'))
    return p,files


def fixture(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir();a,files=archive(inputs)
    offline={'schema':'ovl.offline-runtime-setup.v1','source_root':'source','source_files':files,'wheels':'wheels'}
    write_json(inputs/'offline.json',offline)
    (inputs/'fetch.py').write_text('# pinned test double\n');(inputs/'setup.py').write_text('# pinned test double\n')
    write_json(inputs/'plan.json',{'schema':'ovl.public-wheel-download.v1','files':[{'path':'fixture.whl','bytes':1,'sha256':'a'*64,'url':'https://files.pythonhosted.org/fixture.whl'}]})
    v={'schema':'ovl.public-runtime-setup.v1','offline_config':'offline.json','fetch_script':'fetch.py','setup_script':'setup.py','wheel_plan':'plan.json','source_archive':a.name,'download_seconds':120}
    for k in ('offline_config','fetch_script','setup_script','wheel_plan','source_archive'):v[k+'_sha256']=file_hash(inputs/v[k])
    config=inputs/'config.json';write_json(config,v)
    return inputs,config,v


def test_selected_archive_is_reconstructed_exactly(tmp_path):
    a,files=archive(tmp_path);out=tmp_path/'out';m.extract(a,file_hash(a),out,files)
    assert file_hash(out/files[0]['path'])==files[0]['sha256']


@pytest.mark.parametrize('fault',['traversal','symlink','hardlink','size','bytes','extra','bad-pin','bytecode','duplicate','preserve-existing'])
def test_bad_archive_has_no_success_and_preserves_partial(tmp_path,fault):
    a,files=archive(tmp_path,fault);out=tmp_path/'out';pin=file_hash(a)
    if fault=='bad-pin':pin='0'*64
    elif fault=='bytecode':files[0]['path']='src/__pycache__/module.pyc'
    elif fault=='duplicate':files*=2
    elif fault=='preserve-existing':out.mkdir();(out/'sole').write_bytes(b'evidence')
    with pytest.raises(ValueError):m.extract(a,pin,out,files)
    assert not(tmp_path/'outside').exists()
    if fault=='preserve-existing':assert (out/'sole').read_bytes()==b'evidence'


def test_pinned_stages_run_in_order_in_sanitized_environment(tmp_path):
    inputs,config,v=fixture(tmp_path);calls=[];output=tmp_path/'output';runtime=tmp_path/'runtime'
    def execute(argv,**kw):
        calls.append((argv,kw));assert argv[:3]==[sys.executable,'-I','-S']
        assert set(kw['env'])=={'PATH','LANG','HOME','PYTHONDONTWRITEBYTECODE'} and 0<kw['timeout']<=300
        if len(calls)==1:
            assert argv[3].endswith('fetch.py') and (inputs/'source/src/module.py').exists()
            write_json(output/'downloads.json',{'schema':'ovl.public-wheel-download-result.v1','plan_sha256':v['wheel_plan_sha256'],'files':[{**json.loads((inputs/'plan.json').read_bytes())['files'][0],'result':'COMPLETE_HASH_MATCH'}]})
        else:
            assert argv[3].endswith('setup.py');(output/'offline').mkdir();write_json(output/'offline/setup.json',{'schema':'ovl.offline-runtime-setup-result.v1','result':'PASS','config_sha256':v['offline_config_sha256']})
    r=m.setup(config,file_hash(config),inputs,runtime,output,int(time.time())+280,execute=execute)
    assert len(calls)==2 and r['downloads_sha256']==file_hash(output/'downloads.json')
    assert 'no CUDA/training' in r['scope']


@pytest.mark.parametrize('fault',['fetch-tamper','source-tamper','offline-tamper','config-pin','overlong','same-root','output-exists','failed-download','missing-result'])
def test_setup_cannot_pass_bad_inputs_or_failed_stages(tmp_path,fault):
    inputs,config,v=fixture(tmp_path);output=tmp_path/'output';runtime=tmp_path/'runtime';pin=file_hash(config);deadline=int(time.time())+280;calls=[]
    if fault.endswith('-tamper'):
        key={'fetch-tamper':'fetch_script','source-tamper':'source_archive','offline-tamper':'offline_config'}[fault]
        (inputs/v[key]).write_bytes(b'changed')
    elif fault=='config-pin':pin='0'*64
    elif fault=='overlong':deadline+=1000
    elif fault=='same-root':runtime=inputs
    elif fault=='output-exists':output.mkdir();(output/'sole').write_bytes(b'evidence')
    def execute(*args,**kw):
        calls.append(args)
        if fault=='failed-download':raise RuntimeError('download failed')
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):m.setup(config,pin,inputs,runtime,output,deadline,execute=execute)
    assert not(output/'setup.json').exists()
    if fault=='failed-download':assert len(calls)==1
    elif fault=='missing-result':assert len(calls)==2
    else:assert not calls
    if fault=='output-exists':assert (output/'sole').read_bytes()==b'evidence'


def test_nonadjacent_source_parent_collision_refused_before_extraction(tmp_path):
    a,files=archive(tmp_path);files=[{**files[0],'path':n} for n in ['a','a.x','a/b']]
    with pytest.raises(ValueError,match='parent collision'):m.extract(a,file_hash(a),tmp_path/'out',files)
    assert not(tmp_path/'out').exists()


@pytest.mark.parametrize('fault',['download-parent','download-file','offline-fail','offline-parent'])
def test_zero_exit_cannot_substitute_forged_child_receipts(tmp_path,fault):
    inputs,config,v=fixture(tmp_path);output=tmp_path/'output'
    def execute(argv,**kw):
        if argv[3].endswith('fetch.py'):
            f=json.loads((inputs/'plan.json').read_bytes())['files'][0]
            value={'schema':'ovl.public-wheel-download-result.v1','plan_sha256':v['wheel_plan_sha256'],'files':[{**f,'result':'COMPLETE_HASH_MATCH'}]}
            if fault=='download-parent':value['plan_sha256']='0'*64
            if fault=='download-file':value['files'][0]['sha256']='0'*64
            write_json(output/'downloads.json',value)
        else:
            (output/'offline').mkdir();value={'schema':'ovl.offline-runtime-setup-result.v1','result':'PASS','config_sha256':v['offline_config_sha256']}
            if fault=='offline-fail':value['result']='FAIL'
            if fault=='offline-parent':value['config_sha256']='0'*64
            write_json(output/'offline/setup.json',value)
    with pytest.raises(ValueError,match='receipt'):m.setup(config,file_hash(config),inputs,tmp_path/'runtime',output,int(time.time())+280,execute=execute)
    assert not(output/'setup.json').exists()


def test_backward_wall_clock_cannot_extend_setup_budget(tmp_path,monkeypatch):
    inputs,config,v=fixture(tmp_path);output=tmp_path/'output';wall=[100];mono=[100];calls=[]
    monkeypatch.setattr(m.time,'time',lambda:wall[0]);monkeypatch.setattr(m.time,'monotonic',lambda:mono[0])
    def execute(*args,**kwargs):calls.append(kwargs['timeout']);wall[0]=0;mono[0]=400
    with pytest.raises(TimeoutError):m.setup(config,file_hash(config),inputs,tmp_path/'runtime',output,380,execute=execute)
    assert calls==[280] and not(output/'setup.json').exists()


def test_actual_job_supervisor_reaps_grandchild_after_subprocess_timeout(tmp_path):
    from test_pod_job_worker import fixture as job_fixture,start,exited
    marker=tmp_path/'late-write'
    grandchild='import time;from pathlib import Path;time.sleep(2);Path('+repr(str(marker))+').write_text("orphan")'
    child='import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",'+repr(grandchild)+']);time.sleep(20)'
    code='import subprocess,sys\nsubprocess.run([sys.executable,"-c",'+repr(child)+'],timeout=.3,check=True)\n'
    job,value,root,worker=job_fixture(tmp_path,code)
    assert start(job,root,worker).returncode==0
    assert exited(job)['exit_code']!=0
    time.sleep(2.2)
    assert not marker.exists()


def test_pax_metadata_archive_is_refused_instead_of_silently_interpreted(tmp_path):
    a,files=archive(tmp_path);data=b'print("selected")\n'
    with tarfile.open(a,'w:gz',format=tarfile.PAX_FORMAT) as t:
        i=tarfile.TarInfo(files[0]['path']);i.size=len(data);i.pax_headers={'comment':'unselected metadata'};t.addfile(i,io.BytesIO(data))
    with pytest.raises(ValueError,match='member differs'):m.extract(a,file_hash(a),tmp_path/'out',files)
