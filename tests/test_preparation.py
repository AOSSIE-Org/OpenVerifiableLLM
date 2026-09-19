"""Full-data orchestration on synthetic raw files; no production gate credit."""
import bz2
import copy
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ovl_pipeline.acquisition import Source, verify_source
from ovl_pipeline.anchoring import PublisherPolicy, REPOSITORY, REPOSITORY_ID, OWNER_ID, WORKFLOW, ISSUER
from ovl_pipeline.canonical import EvidenceError, digest, file_hash, inventory, read_json, write_json
from ovl_pipeline.preparation import build_prepared, preparation_code, preparation_environment, prepare_committed, validate_contract


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    wiki = tmp_path / "raw-wiki";wiki.mkdir()
    conv = tmp_path / "raw-conversation"; (conv / "data").mkdir(parents=True)
    date = "20260901";filename = f"enwiki-{date}-pages-articles.xml.bz2"
    raw = bz2.compress((Path(__file__).parent / "fixtures/pipeline/wiki.xml").read_bytes())
    (wiki / filename).write_bytes(raw)
    spec = Source(f"https://dumps.wikimedia.org/enwiki/{date}/{filename}", filename, len(raw),
                  {n: hashlib.new(n,raw).hexdigest() for n in ("md5", "sha1")}, ("dumps.wikimedia.org",), "bz2")
    splits = {}
    for split in ("train", "validation"):
        name = f"data/{split}-00000-of-00001-abcdef.parquet";splits[split]=name
        messages = [{"message_id": split+"-root", "message_tree_id": split+"-root", "parent_id": None,
                     "role": "prompter", "text": "Question?", "lang": "en", "review_result": True,
                     "deleted": False, "rank": None},
                    {"message_id": split+"-answer", "message_tree_id": split+"-root", "parent_id": split+"-root",
                     "role": "assistant", "text": "A synthetic answer.", "lang": "en", "review_result": True,
                     "deleted": False, "rank": 0}]
        pq.write_table(pa.Table.from_pylist(messages), conv/name)
    (conv/"LICENSE").write_text('synthetic fixture only')
    (conv/"README.md").write_text('synthetic fixture only')
    status = {"jobs": {"articlesdumprecombine": {"status": "done", "files": {filename: {
        "url": f"/enwiki/{date}/{filename}", "size": len(raw), **spec.upstream_checksums}}}}}
    write_json(wiki/"dumpstatus.json",status)
    metadata_names = []
    for suffix in ("index.html", "md5sums.txt", "sha1sums.txt"):
        name = f"enwiki-{date}-{suffix}";metadata_names.append(name)
        (wiki/name).write_text("synthetic index" if suffix=="index.html" else spec.upstream_checksums[suffix.split('sums')[0]]+"  "+filename+"\n")
    observed=verify_source(wiki/filename,spec)
    receipt={"schema":"ovl.acquisition-receipt.v1","result":"PASS","requested_url":spec.url,
             "spec_root":digest(spec.object()),"verified":observed,
             "downloader_code_sha256":file_hash(Path(__file__).parents[1]/"src/ovl_pipeline/acquisition.py")}
    write_json(wiki/"receipt.json",receipt)
    write_json(wiki/(filename+".verified.json"),{"schema":"ovl.acquisition-result.v1","spec_root":digest(spec.object()),
               "verified":observed,"network_performed":False,"receipt":"receipt.json","receipt_sha256":digest(receipt)})
    inv=inventory(conv,["LICENSE","README.md",*splits.values()])
    recorded=[]
    for e in inv:
        b=(conv/e["path"]).read_bytes()
        recorded.append({**e,"upstream_blob_id":hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest(),
                         "upstream_lfs_sha256":e["sha256"] if e["path"].endswith(".parquet") else None})
    write_json(conv/"acquisition.json",{"schema":"ovl.oasst-acquisition-survey.v1","repo":"OpenAssistant/oasst1","revision":"0"*40,"files":recorded})
    archive = ([{**e, "path": "wikipedia/"+e["path"]} for e in inventory(wiki,[filename,"dumpstatus.json",filename+".verified.json","receipt.json",*metadata_names])]
               + [{**e, "path": "conversation/"+e["path"]} for e in inventory(conv,["LICENSE","README.md",*splits.values(),"acquisition.json"])]
               + [{"path": n, "bytes": 3, "sha256": hashlib.sha256(b"raw").hexdigest()} for n in ("README.md","LICENSES.md")])
    contract = {"schema": "ovl.source-preparation.v2", "scope": "production-source-preparation",
                "run_id": "synthetic-test", "attempt_id": "test-1", "source_revision": "0"*40,
                "wikipedia": {"date": date, "spec": spec.object(), "inventory": inventory(wiki,[filename]), "official_status_sha256": file_hash(wiki/"dumpstatus.json"), "metadata_inventory":inventory(wiki,metadata_names)},
                "conversation": {"repo": "OpenAssistant/oasst1", "revision": "0"*40,
                                 "inventory": inventory(conv,["LICENSE","README.md",*splits.values()]), "splits": splits},
                "recipe": {"extractor": "main-nonredirect-stripcode-v1", "tokenizer_vocab_size": 320,
                           "tokenizer_sample_bytes": 100000, "conversation_policy": "oasst-en-preferred-path-v1"},
                "code": preparation_code(), "environment": preparation_environment(),
                "acquisition_receipts": {"wikipedia": file_hash(wiki/(filename+".verified.json")), "conversation":file_hash(conv/"acquisition.json")},
                "archive": {"repo":"AOSSIE/openverifiable-synthetic-evidence", "revision":"1"*40,"prefix":"raw/synthetic",
                            "inventory":sorted(archive,key=lambda e:e["path"]),"retention_days_target":90,
                            "retention_policy":"owner-preserve-best-effort-public-host-v1"}}
    return contract,wiki,conv


def test_complete_preparation_reconstructs_all_synthetic_outputs(inputs,tmp_path):
    contract,wiki,conv=inputs
    a=build_prepared(contract,wiki,conv,tmp_path/"first")
    b=build_prepared(contract,wiki,conv,tmp_path/"clean-reconstruction")
    assert a==b
    assert a["source_commitment_sha256"]==digest(contract)
    assert a["corpus"]["record_count"]==7 and a["corpus"]["counts"]["included"]==3
    assert set(a["streams"])=={"wikipedia","conversation","conversation-validation"}
    assert a["conversation_selection"]["messages"]==4
    assert a["validation_used_for_training"] is False


def test_raw_source_changed_or_incomplete_cannot_prepare(inputs,tmp_path):
    contract,wiki,conv=inputs
    p=wiki/contract["wikipedia"]["spec"]["filename"]
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(EvidenceError):build_prepared(contract,wiki,conv,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_contract_code_environment_policy_and_inventory_fail_closed(inputs):
    contract,_,_=inputs
    mutations=[]
    for key,value in [("scope","development-identity-test-only"),("code",[]),("environment",{}),("schema","future")]:
        c=copy.deepcopy(contract);c[key]=value;mutations.append(c)
    c=copy.deepcopy(contract);c["recipe"]["filter_articles"]="short-only";mutations.append(c)
    c=copy.deepcopy(contract);c["wikipedia"]["inventory"]*=2;mutations.append(c)
    c=copy.deepcopy(contract);c["conversation"]["inventory"].pop();mutations.append(c)
    c=copy.deepcopy(contract);c["recipe"]["tokenizer_sample_bytes"]=0;mutations.append(c)
    for c in mutations:
        with pytest.raises(EvidenceError):validate_contract(c)


def test_missing_real_anchor_refuses_before_output(inputs,tmp_path):
    contract,wiki,conv=inputs
    statement=tmp_path/"source.json";write_json(statement,contract)
    policy=PublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,WORKFLOW,ISSUER,
                           'refs/heads/feat/verifiable-wikipedia-pipeline','0'*40,digest(contract),
                           'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    with pytest.raises(EvidenceError):
        prepare_committed(statement,tmp_path/"missing.sigstore.json",policy,wiki,conv,tmp_path/"production")
    assert not (tmp_path/"production").exists()


@pytest.mark.parametrize("parent", ["status", "wiki-result", "network-receipt", "conversation-receipt"])
def test_broken_parent_metadata_rejected_before_preparation(inputs,tmp_path,parent):
    contract,wiki,conv=inputs
    target={"status":wiki/"dumpstatus.json", "wiki-result":wiki/(contract["wikipedia"]["spec"]["filename"]+".verified.json"),
            "network-receipt":wiki/"receipt.json", "conversation-receipt":conv/"acquisition.json"}[parent]
    target.write_bytes(b'{}')
    with pytest.raises(EvidenceError):build_prepared(contract,wiki,conv,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_self_consistent_wrong_official_inventory_still_fails(inputs,tmp_path):
    contract,wiki,conv=inputs
    status=read_json(wiki/"dumpstatus.json")
    status["jobs"]["articlesdumprecombine"]["status"]="in-progress"
    write_json(wiki/"dumpstatus.json",status)
    contract["wikipedia"]["official_status_sha256"]=file_hash(wiki/"dumpstatus.json")
    for e in contract["archive"]["inventory"]:
        if e["path"] == "wikipedia/dumpstatus.json":
            e.update(bytes=(wiki/"dumpstatus.json").stat().st_size,sha256=file_hash(wiki/"dumpstatus.json"))
    with pytest.raises(EvidenceError,match="completed"):
        build_prepared(contract,wiki,conv,tmp_path/"bad")


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong-checksum"])
def test_checksum_listing_must_agree_even_if_all_metadata_hashes_are_rebound(inputs,tmp_path,change):
    contract,wiki,conv=inputs
    name=f"enwiki-{contract['wikipedia']['date']}-sha1sums.txt"
    path=wiki/name
    line=path.read_text()
    path.write_text("unrelated\n" if change=="missing" else line+line if change=="duplicate" else "0"*40+line[40:])
    replacement=inventory(wiki,[name])[0]
    for inv,prefix in ((contract["wikipedia"]["metadata_inventory"],""),(contract["archive"]["inventory"],"wikipedia/")):
        for e in inv:
            if e["path"]==prefix+name:e.update({**replacement,"path":prefix+name})
    with pytest.raises(EvidenceError,match="checksum file disagrees"):
        build_prepared(contract,wiki,conv,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_resume_reuses_verified_stages_and_preserves_incomplete_bytes(inputs,tmp_path,monkeypatch):
    import ovl_pipeline.preparation as prep
    contract,wiki,conv=inputs
    baseline=build_prepared(contract,wiki,conv,tmp_path/'baseline')
    def fail(documents,tokenizer,output,**kwargs):
        output.mkdir();(output/'partial.bin').write_bytes(b'interrupted stream')
        raise OSError('injected stage interruption')
    with monkeypatch.context() as m:
        m.setattr(prep,'prepare_stream',fail)
        with pytest.raises(OSError,match='injected'):
            build_prepared(contract,wiki,conv,tmp_path/'resumed')
    with monkeypatch.context() as m:
        m.setattr(prep,'extract_wikipedia',lambda *a,**k:pytest.fail('complete corpus must be adopted'))
        m.setattr(prep,'train_tokenizer',lambda *a,**k:pytest.fail('complete tokenizer must be adopted'))
        actual=build_prepared(contract,wiki,conv,tmp_path/'resumed',resume=True)
    assert actual==baseline
    kept=list((tmp_path/'resumed-progress/incomplete').rglob('partial.bin'))
    assert len(kept)==1 and kept[0].read_bytes()==b'interrupted stream'
    assert not list((tmp_path/'resumed').rglob('partial.bin'))


@pytest.mark.parametrize('change',['bytes','extra-file','receipt-parent','source-recipe','context-link'])
def test_resume_refuses_altered_completed_stages_and_parent_links(inputs,tmp_path,change):
    contract,wiki,conv=inputs;out=tmp_path/'prepared';build_prepared(contract,wiki,conv,out)
    progress=tmp_path/'prepared-progress'
    if change=='bytes':(out/'tokenizer/tokenizer.json').write_bytes(b'{}')
    elif change=='extra-file':(out/'corpus/extra').write_bytes(b'not registered')
    elif change=='receipt-parent':
        r=read_json(progress/'tokenizer.json');r['inputs']['parents']['corpus']='0'*64;write_json(progress/'tokenizer.json',r)
    elif change=='source-recipe':contract['recipe']['tokenizer_vocab_size']+=1
    else:
        (progress/'context.json').rename(progress/'original-context.json')
        (progress/'context.json').symlink_to(progress/'original-context.json')
    with pytest.raises(EvidenceError):build_prepared(contract,wiki,conv,out,resume=True)


def test_resume_cannot_count_as_clean_reconstruction(inputs,tmp_path):
    contract,wiki,conv=inputs
    with pytest.raises(EvidenceError,match='fresh preparation'):
        prepare_committed(tmp_path/'missing',tmp_path/'missing',None,wiki,conv,tmp_path/'out',
                          expected_preparation={},resume=True)
    assert not (tmp_path/'out').exists()


def test_preparation_lease_refuses_concurrent_writer(inputs,tmp_path):
    from ovl_pipeline.preparation_stages import Stages
    contract,_,_=inputs;output=tmp_path/'out'
    with Stages(output,digest(contract)).lease(resume=False):
        with pytest.raises(EvidenceError,match='active writer'):
            with Stages(output,digest(contract)).lease(resume=True):pass


def test_anchor_statement_replacement_refused_before_transformation(inputs,tmp_path,monkeypatch):
    import ovl_pipeline.preparation as prep
    contract,wiki,conv=inputs
    statement=tmp_path/'source.json';write_json(statement,contract)
    policy=PublisherPolicy('ovl.publisher-policy.v2',REPOSITORY,WORKFLOW,ISSUER,
                           'refs/heads/feat/verifiable-wikipedia-pipeline','0'*40,digest(contract),
                           'sigstore-production-tuf',REPOSITORY_ID,OWNER_ID,'github-hosted')
    def swap(*a):
        changed=copy.deepcopy(contract);changed['recipe']['tokenizer_vocab_size']+=1
        write_json(statement,changed)
        return {'statement_sha256':digest(contract)}
    monkeypatch.setattr(prep,'verify_anchor',swap)
    monkeypatch.setattr(prep,'build_prepared',lambda *a,**k:pytest.fail('unverified source used'))
    with pytest.raises(EvidenceError,match='changed after anchor'):
        prepare_committed(statement,tmp_path/'bundle',policy,wiki,conv,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_execution_observations_distinguish_cache_from_fresh_work(inputs,tmp_path):
    contract,wiki,conv=inputs
    first={};resumed={};fresh={};out=tmp_path/'out'
    a=build_prepared(contract,wiki,conv,out,execution_observation=first)
    b=build_prepared(contract,wiki,conv,out,resume=True,execution_observation=resumed)
    c=build_prepared(contract,wiki,conv,tmp_path/'fresh',execution_observation=fresh)
    assert a==b==c
    assert len(first['stages_executed_this_run'])==6 and not first['stages_adopted_from_local_cache']
    assert resumed['stages_adopted_from_local_cache']==first['stages_executed_this_run']
    assert resumed['stages_executed_this_run']==[]
    assert fresh['stages_executed_this_run']==first['stages_executed_this_run']
    assert first['invocation_id']!=resumed['invocation_id']
    assert read_json(tmp_path/'out-progress/observations'/f'{digest(first)}.json')==first
    assert read_json(tmp_path/'out-progress/observations'/f'{digest(resumed)}.json')==resumed


def test_removing_lock_filename_does_not_admit_second_writer(tmp_path):
    from ovl_pipeline.preparation_stages import Stages
    out=tmp_path/'out'
    with Stages(out,'0'*64).lease(resume=False):
        lock=tmp_path/'out-progress/lease.lock';lock.touch();lock.unlink();lock.touch()
        with pytest.raises(EvidenceError,match='active writer'):
            with Stages(out,'0'*64).lease(resume=True):pass


def test_parent_directory_synced_before_stage_receipt(tmp_path,monkeypatch):
    import os
    import ovl_pipeline.preparation_stages as stages
    out=tmp_path/'out';events=[];real_sync=os.fsync;real_write=stages.write_json
    def sync(fd):
        events.append(('sync',os.fstat(fd).st_ino));real_sync(fd)
    def write(path,value):
        if path.name=='corpus.json' and path.parent.name=='out-progress':
            assert ('sync',out.stat().st_ino) in events
            assert ('sync',out.parent.stat().st_ino) in events
            events.append(('receipt',None))
        real_write(path,value)
    monkeypatch.setattr(os,'fsync',sync);monkeypatch.setattr(stages,'write_json',write)
    def producer(path):
        path.mkdir();write_json(path/'corpus.json',{'ok':True});return {'ok':True}
    with stages.Stages(out,'0'*64).lease(resume=False) as s:
        events.clear();s.run('corpus',{},producer)
    assert ('receipt',None) in events


def test_partial_preservation_is_bounded_and_unindexed_bytes_accounted(tmp_path,monkeypatch):
    import ovl_pipeline.preparation_stages as stages
    out=tmp_path/'out';monkeypatch.setattr(stages,'MAX_PRESERVED_STAGES',2)
    def interrupted(path):
        path.mkdir();(path/'partial').write_bytes(b'abc');raise OSError('interrupted')
    for i in range(3):
        with stages.Stages(out,'0'*64).lease(resume=i>0) as s:
            with pytest.raises(OSError,match='interrupted'):s.run('corpus',{},interrupted)
    with stages.Stages(out,'0'*64).lease(resume=True) as s:
        # A crash after rename and before completion receipt must not hide bytes.
        for receipt in (s.progress/'incomplete').glob('*.json'):receipt.unlink()
        observation=s.storage_observation()
        assert observation['preserved_bytes']==6 and len(observation['preserved_directories'])==2
        with pytest.raises(EvidenceError,match='count bound'):
            s.run('corpus',{},lambda p:pytest.fail('must not run'))
    assert (out/'corpus/partial').read_bytes()==b'abc'


def test_headroom_failure_preserves_partial_and_refuses_producer(tmp_path,monkeypatch):
    import collections
    import ovl_pipeline.preparation_stages as stages
    out=tmp_path/'out'
    with stages.Stages(out,'0'*64).lease(resume=False) as s:
        (out/'corpus').mkdir();(out/'corpus/partial').write_bytes(b'abc')
        Usage=collections.namedtuple('Usage','total used free')
        monkeypatch.setattr(stages.shutil,'disk_usage',lambda p:Usage(1000,801,199))
        with pytest.raises(EvidenceError,match='headroom'):
            s.run('corpus',{},lambda p:pytest.fail('must not run'))
        assert (out/'corpus/partial').read_bytes()==b'abc'


def test_cross_filesystem_rename_error_never_copies_or_deletes(tmp_path,monkeypatch):
    import errno
    import ovl_pipeline.preparation_stages as stages
    out=tmp_path/'out'
    with stages.Stages(out,'0'*64).lease(resume=False) as s:
        (out/'corpus').mkdir();(out/'corpus/partial').write_bytes(b'abc')
        def cross(*a):raise OSError(errno.EXDEV,'cross-device')
        monkeypatch.setattr(stages.os,'rename',cross)
        with pytest.raises(OSError):s.run('corpus',{},lambda p:pytest.fail('must not run'))
        assert (out/'corpus/partial').read_bytes()==b'abc'
        intents=list((s.progress/'incomplete').glob('*-intent.json'))
        assert len(intents)==1 and read_json(intents[0])['files'][0]['bytes']==3
