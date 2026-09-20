"""Actual spawned-worker ordering/equality and failure behavior; synthetic only."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

from ovl_pipeline.data import extract_wikipedia
from ovl_pipeline.extraction_workers import OrderedExtraction


def test_actual_parallel_extraction_matches_serial_all_outputs(tmp_path):
    fixture=ET.parse(Path(__file__).parent/'fixtures/pipeline/wiki.xml').getroot()
    ns=fixture.tag.split('}')[0]+'}' if '}' in fixture.tag else ''
    root=ET.Element(fixture.tag,fixture.attrib)
    empty=copy.deepcopy(fixture.findall(ns+'page')[0])
    empty.find(ns+'id').text='8';empty.find(ns+'revision/'+ns+'id').text='108'
    empty.find(ns+'revision/'+ns+'text').text='{{removed template}}'
    fixture.append(empty)
    for repeat in range(8):
        for page in fixture.findall(ns+'page'):
            p=copy.deepcopy(page)
            for element in [p.find(ns+'id'),p.find(ns+'revision/'+ns+'id')]:
                element.text=str(int(element.text)+repeat*100000)
            root.append(p)
    raw=tmp_path/'wiki.xml';ET.ElementTree(root).write(raw,encoding='utf-8')
    one=extract_wikipedia([raw],tmp_path/'serial',workers=1)
    parallel=extract_wikipedia([raw],tmp_path/'parallel',workers=4)
    assert one==parallel
    assert one['counts']['empty_extracted_text']==8
    for name in ('articles.jsonl','ledger.jsonl','corpus.json'):
        assert (tmp_path/'serial'/name).read_bytes()==(tmp_path/'parallel'/name).read_bytes()


def test_callback_failure_propagates_without_success_or_remaining_pool():
    children=[]
    def broken(*a):
        children.extend(worker.pool._processes.values())
        raise RuntimeError('injected output failure')
    worker=OrderedExtraction(broken,workers=2,pending_limit=2)
    with pytest.raises(RuntimeError,match='injected'):
        with worker:
            worker.submit({'ordinal':0},None,'a [[link]]')
            worker.submit({'ordinal':1},None,'b <ref>citation</ref>')
    assert children and all(not child.is_alive() and child.exitcode is not None for child in children)


def test_tiny_input_does_not_spawn_and_preserves_exclusion_order():
    seen=[]
    with OrderedExtraction(lambda *a:seen.append(a)) as worker:
        worker.submit({'ordinal':0},'redirect',None)
        worker.submit({'ordinal':1},None,'[[text]]')
        assert worker.pool is None
    assert seen==[({'ordinal':0},'redirect',None),({'ordinal':1},None,'text')]


def test_raw_byte_bound_drains_before_more_work_without_dropping_large_page():
    seen=[]
    with OrderedExtraction(lambda r,why,text:seen.append(text),workers=1,pending_bytes_limit=3) as worker:
        worker.submit({},None,'ab');assert worker.pending_bytes==2
        worker.submit({},None,'cd');assert seen==['ab'] and worker.pending_bytes==2
        worker.submit({},None,'large article');assert seen==['ab','cd'] and len(worker.pending)==1
        worker.submit({},None,'z');assert seen==['ab','cd','large article']
    assert seen==['ab','cd','large article','z'] and worker.pending_bytes==0


def test_pool_byte_bound_and_parent_recursion_setting():
    import sys
    seen=[];before=sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(1234)
        with OrderedExtraction(lambda r,why,text:seen.append((r['ordinal'],text)),workers=2,pending_limit=4,pending_bytes_limit=64) as worker:
            for i in range(8):worker.submit({'ordinal':i},None,f'[[a{i}]]')
            assert worker.pool is not None
            assert worker.pool.submit(sys.getrecursionlimit).result()==1234
            worker.submit({'ordinal':8},None,'[[big]]'+'y'*200)
            assert len(worker.pending)==1 and worker.pending_bytes>64
            for i in range(9,13):worker.submit({'ordinal':i},None,f'[[b{i}]]')
        assert [i for i,_ in seen]==list(range(13)) and worker.pending_bytes==0
    finally:sys.setrecursionlimit(before)


@pytest.mark.parametrize('ordinal',[2,6])
def test_parent_detects_injected_dropped_page(tmp_path,monkeypatch,ordinal):
    from ovl_pipeline.canonical import EvidenceError,read_json
    original=OrderedExtraction.submit
    def drop(self,record,reason,raw):
        if record['ordinal']!=ordinal:return original(self,record,reason,raw)
    monkeypatch.setattr(OrderedExtraction,'submit',drop)
    with pytest.raises(EvidenceError,match='ordinal|reconcile'):
        extract_wikipedia([Path(__file__).parent/'fixtures/pipeline/wiki.xml'],tmp_path/'bad')
    assert not (tmp_path/'bad/corpus.json').exists()
    assert read_json(tmp_path/'bad/extraction-failure.json')['result']=='FAIL'


def test_parent_failure_preserves_pending_identities_without_draining(tmp_path):
    from ovl_pipeline.canonical import EvidenceError,read_json
    raw=ET.parse(Path(__file__).parent/'fixtures/pipeline/wiki.xml').getroot()
    raw.append(copy.deepcopy(list(raw)[1]))
    source=tmp_path/'duplicate.xml';ET.ElementTree(raw).write(source,encoding='utf-8')
    with pytest.raises(EvidenceError,match='duplicate'):
        extract_wikipedia([source],tmp_path/'bad')
    failure=read_json(tmp_path/'bad/extraction-failure.json')
    assert len(failure['worker_observation']['pending_records'])==7
    assert failure['parsed_by_source']=={'0':7} and failure['emitted_by_source']=={}
    assert not (tmp_path/'bad/corpus.json').exists()
