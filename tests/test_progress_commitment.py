"""Public-download and append-only request checks; no remote publication/signing."""
from dataclasses import asdict
import pytest
from test_production_commitment import request as production_request,repository
from test_pipeline import prepared
from ovl_pipeline.production_anchoring import PACKET_FILES
from test_progress_anchoring import packet as progress_packet
from ovl_pipeline import progress_commitment as m
from ovl_pipeline.canonical import EvidenceError,canonical,digest,inventory,sha256,write_json,read_json
from ovl_pipeline.production_identity import ProductionPublisherPolicy,PRODUCTION_WORKFLOW


def request(tmp_path):
    source_dir=tmp_path/'source';source_dir.mkdir();root,rr=production_request(source_dir)
    # Structural request fixture: run/registration consistency belongs to generate.
    progress=tmp_path/'progress';progress.mkdir();r,reg,envs,policies=progress_packet(progress)
    rr['registration_sha256']=reg;rr['packet']['prefix']='production-registration/'+reg
    for e in rr['packet']['inventory']:
        if e['path']=='registration.json':e['sha256']=reg
    policy=ProductionPublisherPolicy(**{**asdict(policies[0]),'workflow':PRODUCTION_WORKFLOW,'statement_sha256':reg})
    def archive(prefix,inv):return {'repo':rr['packet']['repo'],'revision':'1'*40,'prefix':prefix,'inventory':inv}
    return {'schema':'ovl.progress-signing-request.v1','registration_request':rr,'registration_policy':asdict(policy),
        'registration_anchor':archive('production-anchors/'+reg,[{'path':'registration.sigstore.json','bytes':2,'sha256':'1'*64}]),
        'envelopes':envs[:1],'previous_progress':[],
        'checkpoint_archive':read_json(progress/'progress-00000/statement.json')['archive']}


def test_closed_progress_request_shapes(tmp_path):
    r=request(tmp_path);assert m.validate_request(r).statement_sha256==r['registration_request']['registration_sha256']


@pytest.mark.parametrize('change',['tag','outside-repo','missing-checkpoint','extra-checkpoint','oversized','missing-parent','wrong-policy','wrong-root','unknown','empty-chain'])
def test_malformed_progress_request_refused(tmp_path,change):
    r=request(tmp_path);a=r['checkpoint_archive']
    if change=='tag':a['revision']='main'
    elif change=='outside-repo':a['repo']='unapproved/repo'
    elif change=='missing-checkpoint':a['inventory'].pop()
    elif change=='extra-checkpoint':a['inventory'].append(a['inventory'][0])
    elif change=='oversized':a['inventory'][0]['bytes']=2**32
    elif change=='missing-parent':r['envelopes'].append(r['envelopes'][0])
    elif change=='wrong-policy':r['registration_policy']['workflow']='.github/workflows/anchor-pipeline.yml'
    elif change=='wrong-root':r['registration_policy']['statement_sha256']='0'*64
    elif change=='unknown':r['allow_unsigned']=True
    else:r['envelopes']=[]
    with pytest.raises(EvidenceError):m.validate_request(r)


def archive_fixture(tmp_path):
    raw=tmp_path/'raw';raw.mkdir()
    for n in ('checkpoint.json','state.json','state.safetensors'):(raw/n).write_bytes(n.encode())
    a={'repo':'AOSSIE/openverifiable-test-evidence','revision':'1'*40,'prefix':'production-checkpoints/'+'2'*64+'/boundary-00000',
       'inventory':inventory(raw,['checkpoint.json','state.json','state.safetensors'])}
    return raw,a


def test_download_is_anonymous_forced_and_full_bytes_checked(tmp_path):
    raw,a=archive_fixture(tmp_path);calls=[]
    def fetch(url):return canonical([{'type':'file','path':a['prefix']+'/'+e['path'],'size':e['bytes']} for e in a['inventory']])
    def download(**kw):
        calls.append(kw);return raw/kw['filename'].rsplit('/',1)[1]
    v=m.download_archive(a,tmp_path/'fresh',fetch=fetch,download=download)
    assert len(calls)==3
    assert v['result']=='PASS' and calls[0]['token'] is False and calls[0]['force_download'] is True
    assert calls[0]['local_files_only'] is False and calls[0]['revision']==a['revision']
    with pytest.raises(EvidenceError):m.download_archive(a,tmp_path/'fresh',fetch=fetch,download=download)


@pytest.mark.parametrize('change',['bytes','tree','size','failure'])
def test_remote_metadata_or_download_failure_is_not_verification(tmp_path,change):
    raw,a=archive_fixture(tmp_path)
    def fetch(url):
        entries=[{'type':'file','path':a['prefix']+'/'+e['path'],'size':e['bytes']} for e in a['inventory']]
        if change=='tree':entries=[]
        if change=='size':entries[0]['size']+=1
        return canonical(entries)
    def download(**kw):
        if change=='failure':raise RuntimeError('private transport details must not escape')
        p=raw/'checkpoint.json'
        if change=='bytes':p.write_bytes(b'altered')
        return p
    with pytest.raises(EvidenceError) as caught:m.download_archive(a,tmp_path/'fresh',fetch=fetch,download=download)
    assert 'private transport details' not in str(caught.value)


def test_progress_requests_are_one_append_only_file_at_head(tmp_path):
    git,commit=repository(tmp_path);p=tmp_path/m.REQUEST_DIRECTORY/'test-run-attempt-boundary-00000.json'
    p.parent.mkdir(parents=True);p.write_text('{}');env=commit()
    assert m.select_request(tmp_path,env)==str(p.relative_to(tmp_path))
    (tmp_path/'README').write_text('unrelated');env=commit();assert m.select_request(tmp_path,env) is None
    p.write_text('{"changed":true}');env=commit()
    with pytest.raises(EvidenceError):m.select_request(tmp_path,env)
    p.unlink();env=commit()
    with pytest.raises(EvidenceError):m.select_request(tmp_path,env)
    p.write_text('{}');env=commit()
    with pytest.raises(EvidenceError):m.select_request(tmp_path,env)


@pytest.mark.parametrize('damage',[None,'checkpoint-control','prior-signature'])
def test_generate_downloads_real_safe_state_and_checks_prior_ancestry(tmp_path,prepared,monkeypatch,damage):
    """Actual Git/Ed25519/state/download bytes; explicit publisher-adapter doubles."""
    import shutil
    from pathlib import Path
    from test_production_chain import actual_artifacts
    from ovl_pipeline import progress_anchoring as pa
    from ovl_pipeline.anchoring import ISSUER,OWNER_ID,REPOSITORY,REPOSITORY_ID
    fixture=tmp_path/'numerical';fixture.mkdir();r,root,envs,key,prover,streams=actual_artifacts(prepared,fixture)
    pub=tmp_path/'public';pub.mkdir();packet_root,rr=production_request(pub)
    write_json(packet_root/'registration.json',r);rr['registration_sha256']=root
    rr['packet']['prefix']='production-registration/'+root;rr['packet']['inventory']=inventory(packet_root,sorted(PACKET_FILES))
    repo=rr['packet']['repo'];remote={}
    def archived(prefix,directory,names):
        a={'repo':repo,'revision':'1'*40,'prefix':prefix,'inventory':inventory(directory,names)};remote[prefix]=(a,directory);return a
    bundle_dir=tmp_path/'bundle';bundle_dir.mkdir();write_json(bundle_dir/'registration.sigstore.json',{'explicit-test-double':True})
    reg_archive=archived('production-anchors/'+root,bundle_dir,['registration.sigstore.json'])
    base_policy=dict(schema='ovl.publisher-policy.v2',repository=REPOSITORY,workflow=pa.PROGRESS_WORKFLOW,issuer=ISSUER,
        ref=m.REF,source_revision='2'*40,statement_sha256='0'*64,trust_root='sigstore-production-tuf',repository_id=REPOSITORY_ID,owner_id=OWNER_ID,runner_environment='github-hosted')
    first=archived('production-checkpoints/'+root+'/boundary-00000',prover/'boundary-00000',['checkpoint.json','state.json','state.safetensors'])
    previous=pa.statement(r,root,envs[:1],first,root)
    prevdir=tmp_path/'previous';prevdir.mkdir();write_json(prevdir/'statement.json',previous);write_json(prevdir/'statement.sigstore.json',{'explicit-test-double':True})
    prev_archive=archived('production-progress/'+root+'/progress-00000',prevdir,['statement.json','statement.sigstore.json'])
    current=archived('production-checkpoints/'+root+'/boundary-00001',prover/'boundary-00001',['checkpoint.json','state.json','state.safetensors'])
    value={'schema':'ovl.progress-signing-request.v1','registration_request':rr,'registration_anchor':reg_archive,
           'registration_policy':{**base_policy,'workflow':PRODUCTION_WORKFLOW,'statement_sha256':root},'envelopes':envs[:2],
           'previous_progress':[{'archive':prev_archive,'policy':{**base_policy,'statement_sha256':digest(previous)}}],
           'checkpoint_archive':current}
    gitroot=tmp_path/'repo';gitroot.mkdir();git,commit=repository(gitroot)
    path=gitroot/m.REQUEST_DIRECTORY/(r['run_id']+'-'+r['attempt_id']+'-boundary-00001.json');path.parent.mkdir(parents=True)
    write_json(path,value);environ=commit()
    monkeypatch.setattr(m,'download_packet',lambda request,output:shutil.copytree(packet_root,output))
    called=[]
    def endorsement(*a,**kw):called.append('registration');return {'explicit-test-double':True}
    monkeypatch.setattr(m,'verify_packet',endorsement)
    def anchor(statement,bundle,policy,**kwargs):
        called.append('prior-progress')
        if damage=='prior-signature':raise EvidenceError('explicit failed publisher-signature double')
        assert digest(read_json(statement))==policy.statement_sha256
        return {'explicit-test-double':True}
    monkeypatch.setattr(pa,'verify_anchor',anchor)
    original=m.download_archive
    def download(a,output):
        _,directory=remote[a['prefix']]
        def fetch(url):return canonical([{'type':'file','path':a['prefix']+'/'+e['path'],'size':e['bytes']} for e in a['inventory']])
        def file(**kwargs):
            assert kwargs['token'] is False and kwargs['force_download'] is True
            return directory/kwargs['filename'].rsplit('/',1)[1]
        return original(a,output,fetch=fetch,download=file)
    monkeypatch.setattr(m,'download_archive',download)
    if damage=='checkpoint-control':
        original_read=m.read_state
        def altered(*args):
            md,ts=original_read(*args)
            # Safe-state/control comparator must run after actual file identity.
            monkeypatch.setattr(m,'unpack',lambda *a:{'control':{'altered':True}})
            return md,ts
        monkeypatch.setattr(m,'read_state',altered)
    output=tmp_path/'signing'
    if damage:
        with pytest.raises(EvidenceError):m.generate(gitroot,environ,output)
        assert not (output/'current.json').exists()
    else:
        m.generate(gitroot,environ,output)
        got=read_json(output/'progress/progress-00001/statement.json')
        assert got['previous_statement_sha256']==digest(previous)
        assert got['boundary_sha256']==digest(envs[1])
        checks=read_json(output/'ci-input-checks.json')
        assert len(checks['downloads'])==3 and all(d['result']=='PASS' for d in checks['downloads'])
        assert read_json(output/'ci-progress-policies.json')[-1]['source_revision']==environ['GITHUB_SHA']
        assert called==['registration','prior-progress']


@pytest.mark.parametrize('change',['repo','tag','path','missing'])
def test_standalone_downloader_revalidates_before_any_network(tmp_path,change):
    raw,a=archive_fixture(tmp_path)
    if change=='repo':a['repo']='outside/repo'
    elif change=='tag':a['revision']='main'
    elif change=='path':a['prefix']='../private'
    else:a['inventory'].pop()
    with pytest.raises(EvidenceError):m.download_archive(a,tmp_path/'fresh',fetch=lambda *a:pytest.fail('no network for invalid archive'))
