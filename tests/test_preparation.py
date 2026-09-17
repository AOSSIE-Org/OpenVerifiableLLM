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
