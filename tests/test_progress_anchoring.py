"""Progress composition uses an explicit signature-verifier double; no live anchors."""
from dataclasses import replace
import pytest
from test_production_chain import chain
from ovl_pipeline import progress_anchoring as p
from ovl_pipeline.anchoring import PublisherPolicy,REPOSITORY,REPOSITORY_ID,OWNER_ID,ISSUER
from ovl_pipeline.canonical import EvidenceError,canonical,digest,sha256,write_json


def packet(tmp_path):
    r,root,envs,key=chain();envs=envs[:3];policies=[];previous=root
    for i,env in enumerate(envs):
        ck=env['body']['checkpoint'];raw=canonical(ck)
        archive={'repo':'AOSSIE/openverifiable-synthetic-evidence','revision':'1'*40,
                 'prefix':'production-checkpoints/'+root+'/'+env['body']['checkpoint_path'],
                 'inventory':[{'path':'checkpoint.json','bytes':len(raw),'sha256':sha256(raw)},*ck['files']]}
        value=p.statement(r,root,envs[:i+1],archive,previous);previous=digest(value)
        d=tmp_path/f'progress-{i:05d}';d.mkdir();write_json(d/'statement.json',value);write_json(d/'statement.sigstore.json',{'explicit-test-double':True})
        policies.append(p.ProgressPublisherPolicy(schema='ovl.publisher-policy.v2',repository=REPOSITORY,
            workflow=p.PROGRESS_WORKFLOW,issuer=ISSUER,ref='refs/heads/feat/verifiable-wikipedia-pipeline',source_revision='2'*40,
            statement_sha256=digest(value),trust_root='sigstore-production-tuf',repository_id=REPOSITORY_ID,owner_id=OWNER_ID,runner_environment='github-hosted'))
    return r,root,envs,policies


def test_every_anchor_is_reverified_under_separately_selected_policy(tmp_path,monkeypatch):
    r,root,envs,policies=packet(tmp_path);calls=[]
    def verify(statement,bundle,policy):
        calls.append(policy);return {'explicit-test-double':True,'statement_sha256':policy.statement_sha256}
    monkeypatch.setattr(p,'verify_anchor',verify)
    v=p.verify_prefix(r,root,envs,tmp_path,policies,complete=False)
    assert calls==policies and v['boundaries_checked']==3
    assert v['production_admission']=='NOT_RUN' and v['checkpoint_downloads']=='NOT_RUN'
    with pytest.raises(EvidenceError):p.verify_prefix(r,root,envs,tmp_path,policies,complete=True)


@pytest.mark.parametrize('change',['saved-pass','wrong-policy','wrong-root','parent','boundary','archive-revision','archive-prefix','archive-files','missing-anchor','extra-file'])
def test_substituted_anchors_and_availability_claims_fail_closed(tmp_path,monkeypatch,change):
    r,root,envs,policies=packet(tmp_path)
    monkeypatch.setattr(p,'verify_anchor',lambda *a: {'explicit-test-double':True})
    from ovl_pipeline.canonical import read_json
    d=tmp_path/'progress-00001';value=read_json(d/'statement.json')
    if change=='saved-pass':policies[1]={'result':'PASS'}
    elif change=='wrong-policy':policies[1]=PublisherPolicy(**{**policies[1].__dict__,'workflow':'.github/workflows/anchor-pipeline.yml'})
    elif change=='wrong-root':policies[1]=replace(policies[1],statement_sha256='0'*64)
    elif change=='missing-anchor':(d/'statement.sigstore.json').unlink()
    elif change=='extra-file':write_json(d/'policy.json',{'injected':True})
    else:
        if change=='parent':value['previous_statement_sha256']='0'*64
        elif change=='boundary':value['boundary_sha256']='0'*64
        elif change=='archive-revision':value['archive']['revision']='main'
        elif change=='archive-prefix':value['archive']['prefix']='../other'
        else:value['archive']['inventory'].pop()
        write_json(d/'statement.json',value);policies[1]=replace(policies[1],statement_sha256=digest(value))
    with pytest.raises(EvidenceError):p.verify_prefix(r,root,envs,tmp_path,policies,complete=False)


def test_real_sigstore_adapter_rejects_fake_bundle(tmp_path):
    r,root,envs,policies=packet(tmp_path)
    with pytest.raises(EvidenceError,match='bundle'):p.verify_prefix(r,root,envs,tmp_path,policies,complete=False)
