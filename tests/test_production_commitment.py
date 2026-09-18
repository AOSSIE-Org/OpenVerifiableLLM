"""Synthetic git/public-download tests; no remote signing or production credit."""
from dataclasses import asdict
import subprocess
import pytest
from test_production_anchoring import packet
from ovl_pipeline import production_commitment as m,training
from ovl_pipeline.canonical import EvidenceError,canonical,digest,file_hash,inventory,read_json,sha256,write_json
from ovl_pipeline.anchoring import REPOSITORY


def request(tmp_path):
    root,production,source=packet(tmp_path)
    value={'schema':'ovl.production-signing-request.v1','registration_sha256':production.statement_sha256,
           'packet':{'repo':'AOSSIE/openverifiable-synthetic-test-evidence','revision':'1'*40,
                     'prefix':'production-registration/'+production.statement_sha256,'inventory':inventory(root,sorted(m.PACKET_FILES))},
           'source_policy':asdict(source)}
    return root,value


def test_full_public_packet_download(tmp_path):
    root,r=request(tmp_path);p=r['packet'];calls=[]
    def fetch(url):
        calls.append(url)
        if '/api/datasets/' in url:return canonical([{'type':'file','path':p['prefix']+'/'+e['path'],'size':e['bytes']} for e in p['inventory']])
        return (root/url.rsplit('/',1)[1]).read_bytes()
    m.download_packet(r,tmp_path/'download',fetch=fetch)
    assert len(calls)==1+len(m.PACKET_FILES)
    assert inventory(tmp_path/'download',sorted(m.PACKET_FILES))==p['inventory']
    with pytest.raises(EvidenceError):m.download_packet(r,tmp_path/'download',fetch=fetch)

@pytest.mark.parametrize('change',['repo','revision','prefix','missing','duplicate','size','root','source-identity'])
def test_public_request_restrictions(tmp_path,change):
    root,r=request(tmp_path);p=r['packet']
    if change=='repo':p['repo']='attacker/evidence'
    elif change=='revision':p['revision']='main'
    elif change=='prefix':p['prefix']='../../outside'
    elif change=='missing':p['inventory'].pop()
    elif change=='duplicate':p['inventory'][0]=p['inventory'][1]
    elif change=='size':p['inventory'][0]['bytes']=2**40
    elif change=='root':r['registration_sha256']='0'*64
    else:r['source_policy']['owner_id']='123'
    with pytest.raises(EvidenceError):m.validate_request(r)

@pytest.mark.parametrize('change',['bytes','size','extra'])
def test_public_download_cannot_trust_manifest_only(tmp_path,change):
    root,r=request(tmp_path);p=r['packet']
    def fetch(url):
        if '/api/datasets/' in url:
            entries=[{'type':'file','path':p['prefix']+'/'+e['path'],'size':e['bytes']} for e in p['inventory']]
            if change=='size':entries[0]['size']+=1
            if change=='extra':entries.append({'type':'file','path':p['prefix']+'/extra','size':1})
            return canonical(entries)
        return (root/url.rsplit('/',1)[1]).read_bytes()+b' '
    with pytest.raises(EvidenceError):m.download_packet(r,tmp_path/'download',fetch=fetch)


def repository(tmp_path):
    def git(*args):return subprocess.check_output(['git','-C',str(tmp_path),*args],stderr=subprocess.DEVNULL).decode().strip()
    git('init');git('config','user.name','Test');git('config','user.email','test@example.invalid')
    (tmp_path/'README').write_text('synthetic fixture');git('add','.');git('commit','-m','initial')
    def commit():
        git('add','.');git('commit','-m','test')
        return {'GITHUB_REPOSITORY':REPOSITORY,'GITHUB_REF':m.REF,'GITHUB_SHA':git('rev-parse','HEAD'),'GITHUB_EVENT_NAME':'push'}
    return git,commit


def test_append_only_commit_selection(tmp_path):
    git,commit=repository(tmp_path)
    path=tmp_path/m.REQUEST_DIRECTORY/'test-run-attempt.json';path.parent.mkdir(parents=True);path.write_text('{}')
    env=commit();assert m.select_request(tmp_path,env)==str(path.relative_to(tmp_path))
    (tmp_path/'README').write_text('unrelated update');env=commit();assert m.select_request(tmp_path,env) is None
    path.write_text('{"changed":true}');env=commit()
    with pytest.raises(EvidenceError):m.select_request(tmp_path,env)
    path.unlink();env=commit()
    with pytest.raises(EvidenceError):m.select_request(tmp_path,env)
    path.write_text('{}');env=commit()
    with pytest.raises(EvidenceError):m.select_request(tmp_path,env)


def test_code_commit_and_dependency_identity(tmp_path,monkeypatch):
    git,commit=repository(tmp_path)
    for name,data in [('src/ovl_pipeline/a.py',b'# synthetic module'),('src/model.py',b'# synthetic model'),('requirements/gpu.lock',b'# synthetic dependency lock')]:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    env=commit();items=[{'path':n,'sha256':file_hash(tmp_path/'src'/n)} for n in ('ovl_pipeline/a.py','model.py')]
    r={'code_revision':env['GITHUB_SHA'],'code_root':digest(items),'runtime':{'dependency_lock_sha256':file_hash(tmp_path/'requirements/gpu.lock')}}
    # Real temporary git source comparisons; substitute only this process's
    # module-root lookup because tests don't execute the synthetic source files.
    monkeypatch.setattr(training,'code_root',lambda:digest(items))
    assert m.verify_code(tmp_path,r)['result']=='PASS'
    (tmp_path/'requirements/gpu.lock').write_text('changed')
    with pytest.raises(EvidenceError):m.verify_code(tmp_path,r)
    git('checkout','--','requirements/gpu.lock');(tmp_path/'src/model.py').write_text('changed')
    with pytest.raises(EvidenceError):m.verify_code(tmp_path,r)
    git('checkout','--','src/model.py')
    with pytest.raises(EvidenceError):m.verify_code(tmp_path,{**r,'code_root':'0'*64})


def test_real_current_source_digest_matches_git_then_rejects_drift(tmp_path):
    import shutil
    from pathlib import Path
    git,commit=repository(tmp_path);source=Path(__file__).parents[1]
    names=[p.relative_to(source).as_posix() for p in (source/'src/ovl_pipeline').glob('*.py')]+['src/model.py','requirements/gpu.lock']
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,path)
    env=commit();r={'code_revision':env['GITHUB_SHA'],'code_root':training.code_root(),
        'runtime':{'dependency_lock_sha256':file_hash(source/'requirements/gpu.lock')}}
    assert m.verify_code(tmp_path,r)['result']=='PASS'  # No code_root double.
    (tmp_path/'src/ovl_pipeline/new.py').write_text('# drift')
    env=commit();r['code_revision']=env['GITHUB_SHA']
    with pytest.raises(EvidenceError,match='code root mismatch'):m.verify_code(tmp_path,r)

@pytest.mark.parametrize('case',['not-ancestor','nested','nonregular','missing'])
def test_source_commit_edge_cases(tmp_path,monkeypatch,case):
    git,commit=repository(tmp_path)
    for name in ('src/ovl_pipeline/a.py','src/model.py','requirements/gpu.lock'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic')
    env=commit();r={'code_revision':env['GITHUB_SHA'],'code_root':'1'*64,'runtime':{'dependency_lock_sha256':'2'*64}}
    if case=='not-ancestor':r['code_revision']='0'*40
    elif case=='nested':
        p=tmp_path/'src/ovl_pipeline/nested/a.py';p.parent.mkdir();p.write_text('nested');r['code_revision']=commit()['GITHUB_SHA']
    elif case=='nonregular':
        (tmp_path/'src/ovl_pipeline/link.txt').symlink_to('a.py');r['code_revision']=commit()['GITHUB_SHA']
    else:(tmp_path/'src/model.py').unlink()
    with pytest.raises(EvidenceError):m.verify_code(tmp_path,r)
