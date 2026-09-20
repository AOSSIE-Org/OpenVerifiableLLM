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


def test_git_pack_sized_member_preserves_hash_checks_and_total_bound(tmp_path):
    data=b'complete selected Git pack fixture\n'*(600000)
    assert 16*1024**2<len(data)<64*1024**2
    name='.git/objects/pack/pack-fixture.pack';a=tmp_path/'source.tar.gz'
    files=[{'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]
    with tarfile.open(a,'w:gz',format=tarfile.USTAR_FORMAT) as t:
        member=tarfile.TarInfo(name);member.size=len(data);t.addfile(member,io.BytesIO(data))
    m.extract(a,file_hash(a),tmp_path/'out',files)
    assert (tmp_path/'out'/name).read_bytes()==data
    bad=[{**files[0],'sha256':'0'*64}]
    with pytest.raises(ValueError,match='source bytes'):m.extract(a,file_hash(a),tmp_path/'changed',bad)
    for bad in ([{**files[0],'bytes':64*1024**2+1}],
                [{**files[0],'path':n,'bytes':40*1024**2} for n in ('a','b')]):
        with pytest.raises(ValueError,match='bound'):m.extract(a,file_hash(a),tmp_path/'oversize',bad)
        assert not (tmp_path/'oversize').exists()


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
    elif fault=='missing-result':assert len(calls)==1
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


def test_measured_download_cap_does_not_extend_original_total_bound(tmp_path):
    inputs,config,v=fixture(tmp_path);v['download_seconds']=600;write_json(config,v);calls=[]
    def stop(argv,**kwargs):
        calls.append((argv,kwargs));raise RuntimeError('stop before actual network')
    original=int(time.time())+280
    with pytest.raises(RuntimeError):m.setup(config,file_hash(config),inputs,tmp_path/'runtime',tmp_path/'evidence',original,execute=stop)
    assert len(calls)==1 and int(calls[0][0][-1])<=original and calls[0][1]['timeout']<=280
    other=tmp_path/'other';other.mkdir();i,c,v=fixture(other);v['download_seconds']=601;write_json(c,v)
    with pytest.raises(ValueError,match='download time bound'):m.setup(c,file_hash(c),i,other/'runtime',other/'evidence',int(time.time())+280,execute=stop)
    assert len(calls)==1


def bootstrap_fixture(tmp_path):
    inputs,config,v=fixture(tmp_path)
    data=(inputs/'source.tar.gz').read_bytes();(inputs/'source.tar.gz').unlink()
    python=b'synthetic interpreter archive'
    offline=json.loads((inputs/'offline.json').read_bytes())
    offline.update(interpreter_archive='bootstrap/python.tar.gz',interpreter_sha256=hashlib.sha256(python).hexdigest())
    write_json(inputs/'offline.json',offline)
    v.update(schema='ovl.public-runtime-setup.v2',source_archive='bootstrap/source.tar.gz',offline_config_sha256=file_hash(inputs/'offline.json'),
             bootstrap_script='bootstrap.py',bootstrap_plan='bootstrap-plan.json',bootstrap_seconds=90)
    (inputs/'bootstrap.py').write_text('# selected synthetic helper\n')
    plan={'schema':'ovl.public-bootstrap.v1','repo':'AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence','files':[
        {'path':n,'repo_path':'public/'+n,'revision':'a'*40,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        for n,b in zip(('python.tar.gz','source.tar.gz'),(python,data))]}
    write_json(inputs/'bootstrap-plan.json',plan)
    for k in ('bootstrap_script','bootstrap_plan'):v[k+'_sha256']=file_hash(inputs/v[k])
    write_json(config,v)
    return inputs,config,v,plan,(python,data)


@pytest.mark.parametrize('fault',[None,'parent','files','receipt-fail','receipt-deadline','bytes','size','missing','plan-parent','helper-pin','late'])
def test_public_bootstrap_precedes_extraction_and_installer(tmp_path,fault,monkeypatch):
    inputs,config,v,plan,data=bootstrap_fixture(tmp_path);output=tmp_path/'output';calls=[]
    wall=[100];mono=[100]
    monkeypatch.setattr(m.time,'time',lambda:wall[0]);monkeypatch.setattr(m.time,'monotonic',lambda:mono[0])
    if fault=='plan-parent':
        plan['files'][0]['sha256']='0'*64;write_json(inputs/'bootstrap-plan.json',plan);v['bootstrap_plan_sha256']=file_hash(inputs/'bootstrap-plan.json');write_json(config,v)
    elif fault=='helper-pin':(inputs/'bootstrap.py').write_text('changed')
    def execute(argv,**kwargs):
        calls.append(argv[3]);assert set(kwargs['env'])=={'PATH','LANG','HOME','PYTHONDONTWRITEBYTECODE'}
        if argv[3].endswith('bootstrap.py'):
            assert not(inputs/'source').exists() and not(inputs/'wheels').exists()
            assert 0<kwargs['timeout']<=90
            (inputs/'bootstrap').mkdir()
            for item,b in zip(plan['files'],data):(inputs/'bootstrap'/item['path']).write_bytes(b)
            receipt={'schema':'ovl.public-bootstrap-result.v1','result':'PASS','plan_sha256':v['bootstrap_plan_sha256'],'files':plan['files'],'original_deadline_epoch':int(argv[-1])}
            if fault=='parent':receipt['plan_sha256']='0'*64
            elif fault=='files':receipt['files']=[]
            elif fault=='receipt-fail':receipt['result']='FAIL'
            elif fault=='receipt-deadline':receipt['original_deadline_epoch']+=1
            elif fault=='bytes':(inputs/'bootstrap/python.tar.gz').write_bytes(b'x'*len(data[0]))
            elif fault=='size':(inputs/'bootstrap/python.tar.gz').write_bytes(data[0]+b'x')
            elif fault=='late':wall[0]=0;mono[0]=400
            if fault!='missing':write_json(output/'bootstrap.json',receipt)
        elif argv[3].endswith('fetch.py'):
            assert (inputs/'source/src/module.py').exists()
            write_json(output/'downloads.json',{'schema':'ovl.public-wheel-download-result.v1','plan_sha256':v['wheel_plan_sha256'],'files':[{**json.loads((inputs/'plan.json').read_bytes())['files'][0],'result':'COMPLETE_HASH_MATCH'}]})
        else:
            (output/'offline').mkdir();write_json(output/'offline/setup.json',{'schema':'ovl.offline-runtime-setup-result.v1','result':'PASS','config_sha256':v['offline_config_sha256']})
    args=(config,file_hash(config),inputs,tmp_path/'runtime',output,380)
    if fault:
        with pytest.raises((ValueError,FileNotFoundError,TimeoutError)):m.setup(*args,execute=execute)
        assert len(calls)==(0 if fault in ('plan-parent','helper-pin') else 1)
        assert not(output/'setup.json').exists() and not(inputs/'source').exists()
    else:
        r=m.setup(*args,execute=execute)
        assert len(calls)==3 and r['schema']=='ovl.public-runtime-setup-result.v2'
        assert r['bootstrap_receipt_sha256']==file_hash(output/'bootstrap.json')


def _v2_stage_double(inputs, output, value, plan, data, calls, *,
                     bad_download=False):
    def execute(argv, **kwargs):
        script = Path(argv[3]).name
        calls.append(script)

        if script == "bootstrap.py":
            root = inputs / "bootstrap"
            root.mkdir()
            for item, payload in zip(plan["files"], data):
                (root / item["path"]).write_bytes(payload)
            write_json(output / "bootstrap.json", {
                "schema": "ovl.public-bootstrap-result.v1",
                "result": "PASS",
                "plan_sha256": value["bootstrap_plan_sha256"],
                "files": plan["files"],
                "original_deadline_epoch": int(argv[-1]),
            })
        elif script == "fetch.py":
            files = json.loads((inputs / "plan.json").read_bytes())["files"]
            write_json(output / "downloads.json", {
                "schema": "ovl.public-wheel-download-result.v1",
                "plan_sha256": (
                    "0" * 64 if bad_download
                    else value["wheel_plan_sha256"]
                ),
                "files": [
                    {**item, "result": "COMPLETE_HASH_MATCH"}
                    for item in files
                ],
            })
        else:
            assert script == "setup.py"
            (output / "offline").mkdir()
            write_json(output / "offline/setup.json", {
                "schema": "ovl.offline-runtime-setup-result.v1",
                "result": "PASS",
                "config_sha256": value["offline_config_sha256"],
            })
    return execute


def test_v2_receipt_hashing_cannot_publish_after_deadline(tmp_path, monkeypatch):
    inputs, config, value, plan, data = bootstrap_fixture(tmp_path)
    output = tmp_path / "output"
    calls = []
    wall, mono = [100], [100]
    monkeypatch.setattr(m.time, "time", lambda: wall[0])
    monkeypatch.setattr(m.time, "monotonic", lambda: mono[0])

    original_sha = m.sha

    def slow_receipt_hash(path):
        result = original_sha(path)
        if Path(path) == output / "downloads.json":
            # Expire during work after the existing final deadline check.
            wall[0], mono[0] = 0, 381
        return result

    monkeypatch.setattr(m, "sha", slow_receipt_hash)
    execute = _v2_stage_double(
        inputs, output, value, plan, data, calls
    )
    with pytest.raises(TimeoutError):
        m.setup(
            config, file_hash(config), inputs, tmp_path / "runtime",
            output, 380, execute=execute,
        )
    assert not (output / "setup.json").exists()


def test_v2_bad_download_receipt_blocks_installer(tmp_path, monkeypatch):
    inputs, config, value, plan, data = bootstrap_fixture(tmp_path)
    output = tmp_path / "output"
    calls = []
    monkeypatch.setattr(m.time, "time", lambda: 100)
    monkeypatch.setattr(m.time, "monotonic", lambda: 100)
    execute = _v2_stage_double(
        inputs, output, value, plan, data, calls, bad_download=True
    )

    with pytest.raises(ValueError, match="receipt"):
        m.setup(
            config, file_hash(config), inputs, tmp_path / "runtime",
            output, 380, execute=execute,
        )
    assert calls == ["bootstrap.py", "fetch.py"]
    assert not (output / "setup.json").exists()


def test_v2_success_receipt_flush_expiry_keeps_pending_only(tmp_path,monkeypatch):
    inputs,config,value,plan,data=bootstrap_fixture(tmp_path);output=tmp_path/'output';calls=[];mono=[100]
    monkeypatch.setattr(m.time,'time',lambda:100);monkeypatch.setattr(m.time,'monotonic',lambda:mono[0])
    fsync=m.os.fsync
    def slow(fd):
        fsync(fd)
        if (output/'setup.json.pending').exists():mono[0]=381
    monkeypatch.setattr(m.os,'fsync',slow)
    with pytest.raises(TimeoutError):m.setup(config,file_hash(config),inputs,tmp_path/'runtime',output,380,execute=_v2_stage_double(inputs,output,value,plan,data,calls))
    assert not(output/'setup.json').exists() and (output/'setup.json.pending').exists()


@pytest.mark.parametrize('backward_clock',[False,True])
def test_measured_setup_allocation_retains_original_deadline(tmp_path,monkeypatch,backward_clock):
    inputs,config,v=fixture(tmp_path);v['download_seconds']=600;write_json(config,v)
    output=tmp_path/'output';wall=[100];mono=[100];calls=[]
    monkeypatch.setattr(m.time,'time',lambda:wall[0]);monkeypatch.setattr(m.time,'monotonic',lambda:mono[0])
    def execute(argv,**kwargs):
        calls.append((argv,kwargs))
        if argv[3].endswith('fetch.py'):
            assert argv[-1]=='700' and kwargs['timeout']==900
            wall[0]=0 if backward_clock else 375;mono[0]=375
            f=json.loads((inputs/'plan.json').read_bytes())['files'][0]
            write_json(output/'downloads.json',{'schema':'ovl.public-wheel-download-result.v1','plan_sha256':v['wheel_plan_sha256'],'files':[{**f,'result':'COMPLETE_HASH_MATCH'}]})
        else:
            assert kwargs['timeout']==625
            mono[0]=1001
            if not backward_clock:wall[0]=1001
            (output/'offline').mkdir();write_json(output/'offline/setup.json',{'schema':'ovl.offline-runtime-setup-result.v1','result':'PASS','config_sha256':v['offline_config_sha256']})
    with pytest.raises(TimeoutError,match='original setup deadline'):
        m.setup(config,file_hash(config),inputs,tmp_path/'runtime',output,1000,execute=execute)
    assert len(calls)==2 and not(output/'setup.json').exists()
