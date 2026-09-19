"""Real stdlib reader through an explicit local SSH process double."""
import base64
import json
from pathlib import Path
import sys
import time
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from pod_observe import observe_many
from test_pod_transfer import setup
from ovl_pipeline.canonical import EvidenceError,canonical,sha256


def test_complete_selected_metadata_uses_one_exchange_and_retains_exact_bytes(tmp_path):
    t,remote,calls,processes=setup(tmp_path);out=tmp_path/'observed';out.mkdir()
    expected={}
    for i in range(5):
        name=f'record-{i}.json';data=canonical({'counter':i,'text':'é'*100})
        (remote/name).write_bytes(data);expected[name]=out/name
    expected['absent.json']=out/'absent.json'
    result=observe_many(t,expected,65536,int(time.time())+30)
    assert len(calls)==1 and all(p.returncode==0 for p in processes)
    assert result['absent.json'] is None and not expected['absent.json'].exists()
    for name in sorted(expected):
        if name!='absent.json':assert expected[name].read_bytes()==(remote/name).read_bytes()
    assert result['record-4.json']['counter']==4


@pytest.mark.parametrize('damage',['symlink','fifo','oversize','invalid-json'])
def test_unsafe_or_unparseable_peer_metadata_is_rejected(tmp_path,damage):
    import os
    t,remote,calls,processes=setup(tmp_path);p=remote/'record.json';dest=tmp_path/'observed.json'
    if damage=='symlink':p.symlink_to(tmp_path/'outside')
    elif damage=='fifo':os.mkfifo(p)
    else:p.write_bytes(b'x'*65 if damage=='oversize' else b'not JSON')
    with pytest.raises(EvidenceError):observe_many(t,{'record.json':dest},64,int(time.time())+30)
    assert all(p.poll() is not None for p in processes)
    if damage=='invalid-json':assert dest.read_bytes()==b'not JSON'
    else:assert not dest.exists()


@pytest.mark.parametrize('damage',['extra','missing','path','hash','length','boolean','encoding'])
def test_forged_bundle_framing_fails_before_local_publication(tmp_path,damage):
    t,remote,calls,processes=setup(tmp_path);data=b'{}';dest=tmp_path/'observed.json'
    value={'path':'a.json','present':True,'bytes':2,'sha256':sha256(data),'data_b64':base64.b64encode(data).decode()}
    values=[value]
    if damage=='extra':values.append(value.copy())
    elif damage=='missing':values=[]
    elif damage=='path':value['path']='../elsewhere'
    elif damage=='hash':value['sha256']='a'*64
    elif damage=='length':value['bytes']=1
    elif damage=='boolean':values=[{'path':'a.json','present':0}]
    else:value['data_b64']='invalid!'
    def stream(argv,reply,*a,**kw):reply.write(json.dumps(values).encode())
    t.stream=stream
    with pytest.raises(EvidenceError):observe_many(t,{'a.json':dest},64,int(time.time())+30)
    assert not dest.exists()


def test_unconfined_paths_duplicate_destinations_and_expired_deadline_are_rejected(tmp_path):
    t,remote,calls,processes=setup(tmp_path);dest=tmp_path/'observed.json'
    for selection in ({'../escape':dest},{'a':dest,'b':dest}):
        with pytest.raises(EvidenceError):observe_many(t,selection,64,int(time.time())+30)
    with pytest.raises(EvidenceError,match='deadline'):observe_many(t,{'a':dest},64,int(time.time())-1)
    assert not calls and not dest.exists()
