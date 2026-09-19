import copy
from pathlib import Path
import shutil
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from build_source_request import build
from ovl_pipeline.canonical import EvidenceError,digest,write_json,file_hash,read_json
from test_preparation import inputs

@pytest.fixture
def archive(inputs,tmp_path):
    c,wiki,conv=inputs
    files=c['archive']['inventory']
    # Synthetic notices in the existing fixture are exactly b'raw'.
    prefix='raw/'+digest(files);download=tmp_path/'download';root=download/'downloaded'/prefix
    root.mkdir(parents=True)
    shutil.copytree(wiki,root/'wikipedia');shutil.copytree(conv,root/'conversation')
    for n in ('README.md','LICENSES.md'):(root/n).write_bytes(b'raw')
    observed=read_json(wiki/(c['wikipedia']['spec']['filename']+'.verified.json'))['verified']
    plan={'schema':'ovl.raw-archive-plan.v1','repo':'AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence',
          'prefix':prefix,'files':files,'wikipedia_spec':c['wikipedia']['spec'],'wikipedia_verified':observed}
    path=tmp_path/'plan.json';write_json(path,plan)
    write_json(download/'verification.json',{'schema':'ovl.raw-download-verification.v1','result':'PASS',
        'plan_sha256':file_hash(path),'repo':plan['repo'],'files':files,'wikipedia_verified':observed,
        'total_bytes':sum(e['bytes'] for e in files),'revision':'1'*40})
    write_json(download/'download-intent.json',{'repo':plan['repo'],'revision':'1'*40,'plan_sha256':file_hash(path),
        'token':False,'force_download':True,'xet_disabled':True})
    return path,download,root

def test_builder_binds_complete_download_and_actual_code(archive):
    path,download,root=archive
    value=build(path,download,'synthetic','source-1')
    assert value['source_revision']=='github-actions-head'
    assert value['archive']['revision']=='1'*40 and value['recipe']['tokenizer_vocab_size']==32000
    assert value['recipe']['tokenizer_sample_bytes']==16000000
    assert value['archive']['inventory']==read_json(path)['files']

@pytest.mark.parametrize('change',['missing-receipt','wrong-plan','wrong-revision','partial-download','authenticated','altered-raw'])
def test_builder_fails_closed(archive,change):
    path,download,root=archive
    if change=='missing-receipt':(download/'verification.json').unlink()
    elif change=='altered-raw':
        f=next((root/'wikipedia').glob('*.bz2'));f.write_bytes(b'changed')
    elif change in ('wrong-revision','authenticated'):
        p=download/'download-intent.json';v=read_json(p)
        if change=='wrong-revision':v['revision']='2'*40
        else:v['token']=True
        write_json(p,v)
    else:
        p=download/'verification.json';v=read_json(p)
        if change=='wrong-plan':v['plan_sha256']='0'*64
        else:v['files']=v['files'][:-1]
        write_json(p,v)
    with pytest.raises((EvidenceError,FileNotFoundError)):build(path,download,'synthetic','source-1')
