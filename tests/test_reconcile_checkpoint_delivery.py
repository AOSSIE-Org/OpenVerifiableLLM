"""Real tiny CPU checkpoint/transport recovery; explicit publisher/CUDA doubles."""
from pathlib import Path
from dataclasses import asdict,replace
import time
import pytest
import reconcile_checkpoint_delivery as m
import pod_checkpoint_handoff as handoff
from test_pipeline import prepared
from test_pod_checkpoint_handoff import published
from ovl_pipeline.canonical import EvidenceError,read_json,write_json


def test_lost_receipt_after_policy_delivery_recovers_read_only_without_waiting_marker(prepared,tmp_path,monkeypatch):
    t,remote,r,root,snapshot,ack,policies,calls=published(prepared,tmp_path,monkeypatch)
    handoff.deliver(t,r,root,snapshot,ack,policies,tmp_path/'delivery',int(time.time())+60)
    # Simulate coordinator losing its response after atomic policy installation,
    # while the recorder has removed the completed waiting marker.
    (remote/'awaiting-anchor.json').unlink()
    monkeypatch.setattr(t,'put',lambda *a,**k:pytest.fail('reconciliation must never write remote state'))
    result=m.reconcile(t,r,root,snapshot,ack,policies,tmp_path/'reconcile',int(time.time())+60)
    assert result['result']=='DELIVERED' and result['training_replay']=='NOT_RUN'
    assert result['remote_writes'] is False and len(result['transfers'])==2
    assert read_json(remote/'external-progress-policies.json')==[asdict(p) for p in policies]


def test_partial_anchor_copy_without_atomic_policy_is_not_delivered(prepared,tmp_path,monkeypatch):
    t,remote,r,root,snapshot,ack,policies,calls=published(prepared,tmp_path,monkeypatch)
    t.put('anchors/progress-00000/statement.json',Path(ack['anchor_directory'])/'progress-00000/statement.json',int(time.time())+30)
    monkeypatch.setattr(t,'put',lambda *a,**k:pytest.fail('read-only recovery'))
    result=m.reconcile(t,r,root,snapshot,ack,policies,tmp_path/'reconcile',int(time.time())+60)
    assert result['result']=='NOT_DELIVERED' and not(remote/'external-progress-policies.json').exists()


@pytest.mark.parametrize('damage',['policy','peer-policy','peer-chain','peer-anchor','extra-anchor','local-state','local-anchor','race','ack'])
def test_incorrect_or_changed_evidence_never_recovers_delivery(prepared,tmp_path,monkeypatch,damage):
    t,remote,r,root,snapshot,ack,policies,calls=published(prepared,tmp_path,monkeypatch)
    handoff.deliver(t,r,root,snapshot,ack,policies,tmp_path/'delivery',int(time.time())+60)
    if damage=='policy':policies=[replace(policies[0],source_revision='f'*40)]
    elif damage=='peer-policy':write_json(remote/'external-progress-policies.json',[asdict(policies[0])]*2)
    elif damage=='peer-chain':
        v=read_json(remote/'chain.json');v['boundaries'][0]['signature']='e'*128;write_json(remote/'chain.json',v)
    elif damage=='peer-anchor':(remote/'anchors/progress-00000/statement.json').write_bytes(b'changed')
    elif damage=='extra-anchor':(remote/'anchors/extra').write_bytes(b'preserve')
    elif damage=='local-state':(snapshot/'boundary-00000/state.safetensors').write_bytes(b'changed')
    elif damage=='local-anchor':(Path(ack['anchor_directory'])/'progress-00000/statement.json').write_bytes(b'changed')
    elif damage=='ack':ack={**ack,'checkpoint_archive':{'wrong':'archive'}}
    else:
        original=t.get
        def changed(*a,**kw):
            result=original(*a,**kw);write_json(remote/'external-progress-policies.json',[]);return result
        monkeypatch.setattr(t,'get',changed)
    monkeypatch.setattr(t,'put',lambda *a,**k:pytest.fail('recovery must never write'))
    with pytest.raises(EvidenceError):m.reconcile(t,r,root,snapshot,ack,policies,tmp_path/'reconcile',int(time.time())+60)
    assert not(tmp_path/'reconcile/reconciliation.json').exists()
