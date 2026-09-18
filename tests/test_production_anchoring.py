"""Signature orchestration tests use explicit doubles; no live-signature credit."""
from dataclasses import asdict,replace
import pytest
from test_production_parents import parents,rebind
from test_production_identity import policy
from ovl_pipeline.anchoring import PublisherPolicy,WORKFLOW
from ovl_pipeline.canonical import EvidenceError,digest,write_json
from ovl_pipeline import production_anchoring as m


def packet(tmp_path):
    r,p=parents();source=PublisherPolicy(**asdict(replace(policy(),workflow=WORKFLOW,statement_sha256=digest(p['source']))))
    p['source_policy']=asdict(source);r['source_policy_sha256']=digest(p['source_policy'])
    root=tmp_path/'packet';root.mkdir()
    for key,name in m.PARENT_FILES.items():write_json(root/name,p[key])
    for phase in ('wikipedia','conversation'):
        for plural in ('records','replays'):write_json(root/f'{phase}-pilot-{plural[:-1]}.json',p['pilot_'+plural][phase])
    r['source_bundle_sha256']=digest({'test':'unsigned'})
    write_json(root/'registration.json',r);write_json(root/'source-statement.sigstore.json',{'test':'unsigned'})
    return root,replace(policy(),statement_sha256=digest(r)),source


def test_both_signatures_recomputed_and_no_execution_claim(tmp_path,monkeypatch):
    root,production,source=packet(tmp_path);calls=[]
    def fake(statement,bundle,policy,**kwargs):
        calls.append((statement.name,bundle.name,policy));return {'result':'PASS','test_double':True}
    monkeypatch.setattr(m,'verify_anchor',fake)
    result=m.verify_packet(root,tmp_path/'external-bundle.json',production,source)
    assert [c[0] for c in calls]==['registration.json','source-statement.json']
    assert calls[0][2] is production and calls[1][2] is source
    assert result['production_admission']=='NOT_RUN' and result['assertion_truth_established'] is False
    assert result['training_replay']=='NOT_RUN'

@pytest.mark.parametrize('which',['production','source'])
def test_saved_pass_cannot_replace_missing_or_invalid_signature(tmp_path,which):
    root,production,source=packet(tmp_path)
    # Actual verifier: neither a PASS string nor empty/fake bundle can verify.
    if which=='source':
        with pytest.raises(EvidenceError,match='only Sigstore bundle v0.3 supported'):m.check_source_parents(root,source)
    else:
        write_json(tmp_path/'saved-pass.json',{'result':'PASS'})
        with pytest.raises(EvidenceError):m.verify_packet(root,tmp_path/'saved-pass.json',production,source)

@pytest.mark.parametrize('mutation',['missing','extra-policy','symlink','directory','oversized','stale-root'])
def test_closed_packet_and_report_parents(tmp_path,monkeypatch,mutation):
    root,production,source=packet(tmp_path)
    monkeypatch.setattr(m,'verify_anchor',lambda *a,**k:{'result':'PASS','test_double':True})
    target=root/'initial-record.json'
    if mutation=='missing':target.unlink()
    elif mutation=='extra-policy':write_json(root/'attacker-policy.json',asdict(source))
    elif mutation=='symlink':target.rename(tmp_path/'outside.json');target.symlink_to(tmp_path/'outside.json')
    elif mutation=='directory':target.unlink();target.mkdir()
    elif mutation=='oversized':target.write_bytes(b' '*(16*1024*1024+1))
    else:write_json(target,{'result':'PASS','injected':True})
    with pytest.raises(EvidenceError):m.verify_packet(root,tmp_path/'bundle',production,source)


def test_packet_cannot_supply_source_trust_policy(tmp_path,monkeypatch):
    root,production,source=packet(tmp_path)
    monkeypatch.setattr(m,'verify_anchor',lambda *a,**k:{'result':'PASS','test_double':True})
    with pytest.raises(EvidenceError):m.verify_packet(root,tmp_path/'bundle',production,replace(source,statement_sha256='0'*64))
    with pytest.raises(EvidenceError):m.verify_packet(root,tmp_path/'bundle',source,source)


def test_optional_local_code_binding_refuses_nonexistent_revision(tmp_path,monkeypatch):
    root,production,source=packet(tmp_path)
    monkeypatch.setattr(m,'verify_anchor',lambda *a,**k:{'result':'PASS','test_double':True})
    basic=m.verify_packet(root,tmp_path/'bundle',production,source)
    assert basic['code_and_dependency_lock_binding']=='NOT_RUN'
    assert all(basic[k]=='NOT_RUN' for k in ('container_identity','installed_runtime_identity','fixed_cost_basis','conversation_split_membership'))
    with pytest.raises(EvidenceError,match='ancestor'):m.verify_packet(root,tmp_path/'bundle',production,source,source_checkout=tmp_path)


def test_source_signature_bundle_identity_is_registered(tmp_path,monkeypatch):
    root,production,source=packet(tmp_path)
    monkeypatch.setattr(m,'verify_anchor',lambda *a,**k:{'result':'PASS','test_double':True})
    write_json(root/'source-statement.sigstore.json',{'different':'even if hypothetically signed'})
    with pytest.raises(EvidenceError,match='bundle differs'):m.verify_packet(root,tmp_path/'bundle',production,source)
