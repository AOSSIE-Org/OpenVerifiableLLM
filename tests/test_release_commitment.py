"""Actual public-byte adapters over local fixtures, with explicit identity doubles."""
from dataclasses import asdict
from pathlib import Path
import shutil
import pytest
from test_preparation import inputs
from test_prepared_verification import prepared
from test_gpu_pilot import cpu_runtime
from test_production_release import setup as release_setup
from test_production_commitment import repository
from ovl_pipeline import release_commitment as m,production_release,progress_anchoring as pa
from ovl_pipeline.canonical import EvidenceError,canonical,digest,inventory,read_json,write_json


def setup(inputs,prepared,tmp_path,monkeypatch,damage=None):
    r,reports,archive,payloads=release_setup(inputs,prepared,tmp_path,monkeypatch)
    context=read_json(reports/'context.json');c=context['closing_request'];root=digest(r);remote={}
    source=read_json(tmp_path/'packet/source-statement.json')
    c['registration_request']['source_policy']['statement_sha256']=digest(source)
    c['registration_request']['source_policy']['source_revision']=source['source_revision']
    def archived(prefix,directory,names):
        item={'repo':archive['repo'],'revision':'1'*40,'prefix':prefix,'inventory':inventory(directory,names)}
        remote[prefix]=directory;return item
    bundle=tmp_path/'reg-anchor';bundle.mkdir();write_json(bundle/'registration.sigstore.json',{'explicit-test-double':True})
    c['registration_anchor']=archived('production-anchors/'+root,bundle,['registration.sigstore.json'])
    previous=root;progress=[]
    for i,env in enumerate(c['envelopes']):
        cp=archived(f'production-checkpoints/{root}/boundary-{i:05d}',tmp_path/'checkpoints'/f'boundary-{i:05d}',
                    ['checkpoint.json','state.json','state.safetensors'])
        statement=pa.statement(r,root,c['envelopes'][:i+1],cp,previous);previous=digest(statement)
        directory=tmp_path/f'public-progress-{i:05d}';directory.mkdir()
        write_json(directory/'statement.json',statement);write_json(directory/'statement.sigstore.json',{'explicit-test-double':True})
        policy=pa.ProgressPublisherPolicy(**{**c['registration_policy'],'workflow':pa.PROGRESS_WORKFLOW,'statement_sha256':previous})
        progress.append({'archive':archived(f'production-progress/{root}/progress-{i:05d}',directory,['statement.json','statement.sigstore.json']),
                         'policy':asdict(policy)})
    c['checkpoint_archive']=cp;c['previous_progress']=progress[:-1];context['final_progress']=progress[-1]
    write_json(reports/'context.json',context);archive['inventory']=inventory(reports,production_release.EVIDENCE)
    archive['prefix']='release-evidence/'+digest(archive['inventory']);remote[archive['prefix']]=reports
    value=production_release.build(r,reports,archive,payloads)
    monkeypatch.setattr(m,'download_packet',lambda req,out:shutil.copytree(tmp_path/'packet',out))
    monkeypatch.setattr(m,'verify_packet',lambda *a,**k:{'explicit-registration-test-double':True})
    def anchor(statement,bundle,policy,**kw):
        if damage=='signature':raise EvidenceError('explicit rejected publisher signature')
        assert digest(read_json(statement))==policy.statement_sha256
        return {'identity':'explicit-publisher-test-double','statement_sha256':policy.statement_sha256}
    monkeypatch.setattr(pa,'verify_anchor',anchor)
    original=m.download_archive;downloads=[]
    def download(a,out):
        downloads.append(a['prefix']);directory=remote[a['prefix']]
        def fetch(url):return canonical([{'type':'file','path':a['prefix']+'/'+e['path'],'size':e['bytes']} for e in a['inventory']])
        def get(**kw):
            assert kw['token'] is False and kw['force_download'] is True
            return directory/kw['filename'].rsplit('/',1)[1]
        return original(a,out,fetch=fetch,download=get)
    monkeypatch.setattr(m,'download_archive',download)
    if damage=='checkpoint':(tmp_path/'checkpoints'/c['envelopes'][-1]['body']['checkpoint_path']/'state.safetensors').write_bytes(b'altered')
    pp,sp,policies=m.policies(context)
    if damage=='external-policy':policies[-1]=pa.ProgressPublisherPolicy(**{**asdict(policies[-1]),'statement_sha256':'0'*64})
    def run(complete=False):return m.verify_parents(value,reports,tmp_path/'checked',tmp_path,pp,sp,policies,complete_checkpoints=complete)
    return r,value,run,downloads,reports


@pytest.mark.parametrize('complete',[False,True])
def test_ancestry_downloads_actual_safe_states_without_claiming_replay(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,complete):
    r,value,run,downloads,reports=setup(inputs,prepared,tmp_path,monkeypatch)
    got,report=run(complete);assert got==r and report['result']=='PASS'
    assert report['raw_reconstruction_performed'] is False and report['training_replay_performed'] is False
    assert set(report['final_models'])=={'base','chat'}
    n=len(read_json(reports/'context.json')['closing_request']['envelopes'])
    assert len(report['checkpoint_indices_downloaded'])==(n if complete else 2)
    assert (tmp_path/'checked/chain/chain.json').exists()==complete


@pytest.mark.parametrize('damage',['signature','checkpoint','external-policy'])
def test_failed_parent_identity_or_bytes_cannot_become_release_verification(cpu_runtime,inputs,prepared,tmp_path,monkeypatch,damage):
    r,value,run,downloads,reports=setup(inputs,prepared,tmp_path,monkeypatch,damage)
    with pytest.raises(EvidenceError):run()
    assert not(tmp_path/'checked/verification.json').exists()
    if damage=='external-policy':assert downloads==[]


def test_release_requests_are_append_only_and_unambiguous(tmp_path):
    git,commit=repository(tmp_path);path=tmp_path/m.REQUEST_DIRECTORY/'test-attempt.json';path.parent.mkdir(parents=True)
    path.write_text('{}');env=commit();assert m.select_request(tmp_path,env)==str(path.relative_to(tmp_path))
    path.write_text('{"different":true}');env=commit()
    with pytest.raises(EvidenceError):m.select_request(tmp_path,env)
    path.unlink();env=commit()
    with pytest.raises(EvidenceError):m.select_request(tmp_path,env)


@pytest.mark.parametrize('module_name',['release_commitment','production_commitment','progress_commitment'])
def test_request_and_workflow_or_dependency_edit_cannot_share_signing_commit(tmp_path,module_name):
    import importlib
    module=importlib.import_module('ovl_pipeline.'+module_name);git,commit=repository(tmp_path)
    name='test-attempt-boundary-00000.json' if module_name=='progress_commitment' else 'test-attempt-release-1.json'
    path=tmp_path/module.REQUEST_DIRECTORY/name;path.parent.mkdir(parents=True);path.write_text('{}')
    (tmp_path/'README').write_text('changed verifier or workflow in request commit')
    with pytest.raises(EvidenceError,match='only the append-only request'):module.select_request(tmp_path,commit())


@pytest.mark.parametrize('damage',[None,'missing','same-commit','unknown-ancestor'])
def test_ci_release_parent_policies_must_predate_request_and_select_ancestors(tmp_path,damage):
    from ovl_pipeline.anchoring import PublisherPolicy,REPOSITORY,WORKFLOW,ISSUER,REPOSITORY_ID,OWNER_ID
    from ovl_pipeline.production_identity import ProductionPublisherPolicy,PRODUCTION_WORKFLOW
    git,commit=repository(tmp_path);revision=git('rev-parse','HEAD')
    source=PublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,WORKFLOW,ISSUER,'refs/heads/feat/verifiable-wikipedia-pipeline',
                           'f'*40 if damage=='unknown-ancestor' else revision,'a'*64,'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    pp=ProductionPublisherPolicy(**{**asdict(source),'workflow':PRODUCTION_WORKFLOW})
    progress=pa.ProgressPublisherPolicy(**{**asdict(source),'workflow':pa.PROGRESS_WORKFLOW})
    value={'schema':'ovl.release-parent-policies.v1','production':asdict(pp),'source':asdict(source),'progress':[asdict(progress)]}
    name='test-attempt-release-1.json';p=tmp_path/'project/release-policies'/name
    if damage!='missing':write_json(p,value)
    if damage not in ('same-commit','missing'):commit()
    path=tmp_path/m.REQUEST_DIRECTORY/name;write_json(path,{});commit()
    if damage is None:
        a,b,c,receipt=m.selected_parent_policies(tmp_path,str(path.relative_to(tmp_path)))
        assert a==pp and b==source and c==[progress] and receipt['sha256']==digest(value)
    else:
        with pytest.raises(EvidenceError):m.selected_parent_policies(tmp_path,str(path.relative_to(tmp_path)))
