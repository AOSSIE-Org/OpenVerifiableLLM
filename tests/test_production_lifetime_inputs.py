"""Actual bounded local SSH transfers; no real credentials or provider resources."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import time
from types import SimpleNamespace
import pytest
from ovl_pipeline.canonical import EvidenceError,digest,read_json,write_json
from ovl_pipeline.run_key import create,load
from ovl_pipeline.production_anchoring import PACKET_FILES
from test_pod_transfer import setup as ssh
import run_production_lifetime as m


def fixture(tmp_path):
    transport,remote,calls,_=ssh(tmp_path)
    packet=tmp_path/'packet';packet.mkdir()
    for name in PACKET_FILES:write_json(packet/name,{'explicit-public-fixture':name})
    private=tmp_path/'owner-key';public=create(private,'synthetic-upload-fixture')
    registration={'run_id':public['run_id'],'run_public_key':public['public_key']}
    bundle=tmp_path/'bundle';write_json(bundle,{'explicit-bundle-double':True})
    from dataclasses import dataclass
    @dataclass
    class Policy:value:str='explicit-policy-double'
    out=tmp_path/'coordinator';out.mkdir()
    health=SimpleNamespace(now=lambda:int(time.time()),bytes=lambda *a,**kw:None,write=lambda *a:None)
    run=SimpleNamespace(control=transport,output=out,health=health,health_file=out/'health.json',
                        plan={'request_checkpoint_epoch':int(time.time())+600})
    args={'packet':packet,'bundle':bundle,'registration':registration,'source_policy':Policy(),'production_policy':Policy()}
    return run,{'run_key':str(private)},args,remote,calls


def test_all_public_parents_and_private_seed_delivered_once_without_public_secret_material(tmp_path):
    run,spec,a,remote,calls=fixture(tmp_path)
    m.inputs(run,spec,a)
    private=Path(spec['run_key']);original=load(private,run_id=a['registration']['run_id'],expected_public_key=a['registration']['run_public_key'])
    transported=load(remote/'private/run-key',run_id=a['registration']['run_id'],expected_public_key=a['registration']['run_public_key'])
    assert bytes(original)==bytes(transported)
    assert (remote/'production-inputs/packet/registration.json').read_bytes()==(a['packet']/'registration.json').read_bytes()
    before=len(calls);m.inputs(run,spec,a);assert len(calls)==before
    seed=bytes(original);seed_hash=__import__('hashlib').sha256(seed).hexdigest().encode()
    for path in run.output.rglob('*.json'):
        data=path.read_bytes();assert seed.hex().encode() not in data and seed_hash not in data
    assert list((private/'operator-transfer-receipts').glob('*/*.json'))


def test_uncertain_put_reuses_original_window_and_remote_exact_bytes(tmp_path,monkeypatch):
    run,spec,a,remote,calls=fixture(tmp_path);put=run.control.put;once=[]
    def uncertain(*args,**kw):
        value=put(*args,**kw)
        if not once:once.append(True);raise TimeoutError('explicit response loss after successful write')
        return value
    monkeypatch.setattr(run.control,'put',uncertain)
    with pytest.raises(TimeoutError):m.inputs(run,spec,a)
    windows={p:p.read_bytes() for p in (run.output/'input-delivery').glob('*-intent.json')}
    monkeypatch.setattr(run.control,'put',put);m.inputs(run,spec,a)
    assert windows and all(p.read_bytes()==v for p,v in windows.items())


def test_changed_remote_parent_is_not_overwritten(tmp_path):
    run,spec,a,remote,calls=fixture(tmp_path)
    p=remote/'production-inputs/packet/registration.json';p.parent.mkdir(parents=True);p.write_bytes(b'preserve conflicting remote bytes')
    with pytest.raises(EvidenceError):m.inputs(run,spec,a)
    assert p.read_bytes()==b'preserve conflicting remote bytes'


def test_v2_adoption_rebuilds_every_completed_predecessor_without_rerunning(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(m,'restore_phase_bindings',lambda plan,root,out,bindings,downloads:calls.append((out.name,plan['prior_jobs'])))
    spec={'schema':'ovl.production-lifetime-invocation.v2','selection':{'phases':{}},'initialization':{}}
    for number,name in enumerate(('qualification','optimization','initialization-baseline','initialization-candidate'),1):
        plan={'schema':'explicit-phase-double','prior_jobs':[]};path=tmp_path/(name+'.json');write_json(path,plan)
        output=tmp_path/name
        item={'plan':str(path),'inputs':str(tmp_path),'output':str(output)}
        if name.startswith('initialization-'):spec['initialization'][name.removeprefix('initialization-')]=item
        else:spec[name]=item
        spec['selection']['phases'][name]=digest(plan)
        if number<=2:
            (output/'final').mkdir(parents=True)
            write_json(output/'final/result.json',{'stages':[{'job_sha256':str(number)*64}]})
    result=m.selected_phases(spec,{}, {})
    assert set(result)==set(spec['selection']['phases'])
    assert calls==[('qualification',[]),('optimization',['1'*64]),('initialization-baseline',['1'*64,'2'*64]),('initialization-candidate',['1'*64,'2'*64])]
    output=tmp_path/'initialization-candidate';output.mkdir()
    write_json(output/'selected-plan.json',{'schema':'explicit-phase-double','prior_jobs':['1'*64]})
    with pytest.raises(EvidenceError,match='parent selection'):m.selected_phases(spec,{}, {})
