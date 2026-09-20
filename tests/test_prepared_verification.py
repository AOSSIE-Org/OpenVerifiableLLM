"""Whole prepared integrity checks reject rehashed accounting mutations."""
import copy
from pathlib import Path
import pytest
from test_preparation import inputs
from ovl_pipeline.preparation import build_prepared
from ovl_pipeline.prepared_verification import verify_prepared
from ovl_pipeline.canonical import EvidenceError,digest,write_json,read_json,file_hash,canonical,Merkle

@pytest.fixture
def prepared(inputs,tmp_path):
    source,wiki,conv=inputs;directory=tmp_path/'prepared'
    value=build_prepared(source,wiki,conv,directory)
    return directory,value,digest(source)

def test_entire_prepared_integrity_is_not_reconstruction(prepared):
    directory,value,source=prepared
    report=verify_prepared(directory,digest(value),source)
    assert report['result']=='PASS' and report['corpus']['included']==3
    assert report['raw_transformations_reconstructed']=='NOT_RUN'
    assert report['production_admission']=='NOT_RUN'

@pytest.mark.parametrize('change',['raw-byte','extra-file','symlink','wrong-root','wrong-source','wrong-split'])
def test_identity_confinement_and_split_fail_closed(prepared,change,tmp_path):
    d,v,s=prepared;expected=digest(v)
    if change=='raw-byte':(d/'wikipedia/tokens.u16').write_bytes(b'changed')
    elif change=='extra-file':(d/'corpus/unregistered').write_bytes(b'x')
    elif change=='symlink':(d/'corpus/unregistered').symlink_to(tmp_path)
    elif change=='wrong-root':expected='0'*64
    elif change=='wrong-source':s='0'*64
    else:
        v['validation_used_for_training']=True;write_json(d/'preparation.json',v);expected=digest(v)
    with pytest.raises(EvidenceError):verify_prepared(d,expected,s)

@pytest.mark.parametrize('change',['ordinal-gap','article-lineage','article-text'])
def test_rehashed_output_still_requires_ledger_accounting(prepared,change):
    d,v,s=prepared
    if change=='ordinal-gap':name='ledger.jsonl';key='ordinal';replacement=99
    elif change=='article-lineage':name='articles.jsonl';key='title';replacement='changed title'
    else:name='articles.jsonl';key='text';replacement='changed text'
    path=d/'corpus'/name;records=[read_json_line(line) for line in path.read_bytes().splitlines()];records[0][key]=replacement
    path.write_bytes(b''.join(canonical(row)+b'\n' for row in records))
    corpus=v['corpus']
    for e in corpus['files']:
        if e['path']==name:e.update(bytes=path.stat().st_size,sha256=file_hash(path))
    if name=='ledger.jsonl':
        m=Merkle()
        for row in records:m.add(canonical(row))
        corpus['ledger_root']=m.root()
    write_json(d/'corpus/corpus.json',corpus);write_json(d/'preparation.json',v)
    with pytest.raises(EvidenceError):verify_prepared(d,digest(v),s)

def read_json_line(raw):
    from ovl_pipeline.canonical import parse_json
    return parse_json(raw)


@pytest.mark.parametrize('phase',['wikipedia','conversation','conversation-validation'])
def test_rehashed_stream_cannot_swap_document_membership(prepared,phase):
    d,v,s=prepared;path=d/phase/'documents.jsonl'
    records=[read_json_line(line) for line in path.read_bytes().splitlines()]
    records[0]['identity'][1]='different-source-member'
    path.write_bytes(b''.join(canonical(row)+b'\n' for row in records))
    stream=v['streams'][phase];root=Merkle()
    for row in records:root.add(canonical(row))
    stream['index_root']=root.root()
    for entry in stream['files']:
        if entry['path']=='documents.jsonl':entry.update(bytes=path.stat().st_size,sha256=file_hash(path))
    write_json(d/phase/'stream.json',stream);write_json(d/'preparation.json',v)
    with pytest.raises(EvidenceError,match='ancestry|membership'):verify_prepared(d,digest(v),s)


def test_stream_tokenizer_must_match_prepared_tokenizer(prepared):
    d,v,s=prepared;v['streams']['wikipedia']['tokenizer_sha256']='0'*64
    write_json(d/'wikipedia/stream.json',v['streams']['wikipedia']);write_json(d/'preparation.json',v)
    with pytest.raises(EvidenceError,match='tokenizer parent'):verify_prepared(d,digest(v),s)
