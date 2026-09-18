import itertools
import shutil

import pytest

from test_pipeline import prepared
from ovl_pipeline import data,observed_validation as m,runtime_activity
from ovl_pipeline.canonical import EvidenceError,canonical,digest,inventory,read_json,write_json


@pytest.fixture
def observed(monkeypatch,tmp_path):
    events=[]
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'))
    monkeypatch.setattr(runtime_activity,'stream_validation',lambda *args:events.append(args))
    ticks=itertools.count(step=31)
    monkeypatch.setattr(m.time,'monotonic',lambda:next(ticks))
    return events


@pytest.mark.parametrize('phase',['wikipedia','conversation'])
def test_every_checked_row_is_identical_and_module_iterator_unchanged(prepared,observed,phase,monkeypatch):
    directory,_=prepared;directory=directory/phase;manifest=read_json(directory/'stream.json')
    original=data.rows;original_validator=data.validate_stream
    yielded=[]
    def spy(path):
        for row in original(path):
            yielded.append(row)
            yield row
    monkeypatch.setattr(data,'rows',spy)
    assert m.validate_stream(directory,manifest)==original_validator(directory,manifest)==manifest['targets']
    assert data.rows is spy and data.validate_stream is original_validator
    assert yielded[:manifest['documents']]==yielded[manifest['documents']:]
    assert observed==[(digest(manifest),manifest['documents'],n,False) for n in range(manifest['documents']+1)]+[
        (digest(manifest),manifest['documents'],manifest['documents'],True)]


@pytest.mark.parametrize('damage',['hash','row','duplicate','mask','eos','root','trailing'])
def test_corruption_never_emits_final_success_or_credits_failing_row(prepared,observed,damage,tmp_path):
    source,_=prepared;directory=tmp_path/'damaged';shutil.copytree(source/'wikipedia',directory)
    manifest=read_json(directory/'stream.json')
    docs=list(data.rows(directory/'documents.jsonl'))
    if damage=='hash':
        (directory/'tokens.u16').write_bytes(b'bad')
    elif damage in ('row','duplicate'):
        if damage=='row':docs[1]['offset']+=1
        else:docs[1]['identity']=docs[0]['identity']
        (directory/'documents.jsonl').write_bytes(b''.join(canonical(d)+b'\n' for d in docs))
    elif damage=='mask':
        p=directory/'mask.u8';v=bytearray(p.read_bytes());v[0]=2;p.write_bytes(v)
    elif damage=='eos':
        p=directory/'tokens.u16';v=bytearray(p.read_bytes());v[docs[0]['tokens']*2-2:docs[0]['tokens']*2]=b'\x00\x00';p.write_bytes(v)
    elif damage=='root':manifest['index_root']='f'*64
    else:
        with (directory/'tokens.u16').open('ab') as f:f.write(b'\x02\x00')
        with (directory/'mask.u8').open('ab') as f:f.write(b'\x01')
    if damage!='hash':manifest['files']=inventory(directory,['tokens.u16','mask.u8','documents.jsonl'])
    original=data.rows
    with pytest.raises(EvidenceError):m.validate_stream(directory,manifest)
    assert data.rows is original and not any(e[-1] for e in observed)
    if damage=='hash':assert not observed
    elif damage in ('row','duplicate'):assert observed[-1][2]==1
    elif damage in ('mask','eos'):assert observed[-1][2]==0
    else:assert observed[-1][2]==manifest['documents']


def test_disabled_observation_calls_original_directly(prepared,monkeypatch):
    monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False)
    directory,_=prepared;manifest=read_json(directory/'wikipedia/stream.json')
    seen=[]
    monkeypatch.setattr(data,'validate_stream',lambda *a:seen.append(a) or 42)
    assert m.validate_stream(directory/'wikipedia',manifest)==42 and len(seen)==1


def test_nested_observed_validation_is_rejected_and_original_binding_preserved(prepared,observed,monkeypatch):
    directory,_=prepared;wiki=read_json(directory/'wikipedia/stream.json');chat=read_json(directory/'conversation/stream.json')
    original=data.rows;inside=False
    def nested(path):
        nonlocal inside
        if not inside:
            inside=True
            with pytest.raises(EvidenceError,match='only one observed'):m.validate_stream(directory/'conversation',chat)
        yield from original(path)
    monkeypatch.setattr(data,'rows',nested)
    assert m.validate_stream(directory/'wikipedia',wiki)==wiki['targets']
    assert data.rows is nested
    assert {e[0] for e in observed if e[-1]}=={digest(wiki)}
    # The lock is released after successful invocation.
    assert m.validate_stream(directory/'conversation',chat)==chat['targets']


def test_validation_and_numerical_observations_share_identity_and_sequence(tmp_path,monkeypatch):
    r=runtime_activity
    monkeypatch.setattr(r,'_last',None);monkeypatch.setattr(r,'_sequence',0);monkeypatch.setattr(r,'_process',None)
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'activity.json'))
    r.stream_validation('a'*64,10,0,False,clock=lambda:0);first=read_json(tmp_path/'activity.json')
    r.stream_validation('a'*64,10,10,True,clock=lambda:1);last=read_json(tmp_path/'activity.json')
    r.update({'global_step':1},clock=lambda:31);numeric=read_json(tmp_path/'activity.json')
    assert [v['sequence'] for v in (first,last,numeric)]==[1,2,3]
    assert len({v['process_instance'] for v in (first,last,numeric)})==1
    assert last['complete'] is True and numeric['schema']=='ovl.runtime-activity.v1'
