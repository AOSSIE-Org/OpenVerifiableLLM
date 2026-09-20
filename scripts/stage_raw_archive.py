"""Stage unchanged, verified public raw inputs; no corpus transformations or signing."""
import argparse
import os
from pathlib import Path
import shutil

from ovl_pipeline.acquisition import wikipedia_source
from ovl_pipeline.canonical import EvidenceError, confined, digest, file_hash, inventory, read_json, verify_inventory, write_json
from publish_raw_archive import REPO, validate_plan


def stage(wiki, conversation, metadata, notices, date, output, plan_path):
    if output.exists() or plan_path.exists() or any(p.is_symlink() for p in (wiki,conversation,metadata,notices)):
        raise EvidenceError("fresh stage/plan and regular input roots required")
    spec=wikipedia_source(read_json(metadata/"dumpstatus.json",canonical_required=False),date)
    result_path=confined(wiki,spec.filename+".verified.json")
    acquired=read_json(result_path)
    if acquired.get("schema")!="ovl.acquisition-result.v1" or acquired.get("spec_root")!=digest(spec.object()):
        raise EvidenceError("complete original acquisition receipt required")
    receipt_path=confined(wiki,acquired["receipt"])
    receipt=read_json(receipt_path)
    if (file_hash(receipt_path)!=acquired["receipt_sha256"] or receipt.get("result")!="PASS"
            or receipt.get("verified")!=acquired["verified"] or receipt.get("spec_root")!=digest(spec.object())
            or receipt.get("requested_url")!=spec.url):
        raise EvidenceError("raw acquisition parent mismatch")
    if any(acquired["verified"]["hashes"].get(k)!=v for k,v in spec.upstream_checksums.items()):
        raise EvidenceError("acquisition disagrees with completed upstream metadata")
    raw=confined(wiki,spec.filename)
    if not raw.is_file() or raw.stat().st_size!=spec.bytes:
        raise EvidenceError("promoted complete raw source missing")
    conv=read_json(metadata/"oasst-acquisition.json",canonical_required=False)
    if conv.get("repo")!="OpenAssistant/oasst1" or conv.get("revision")!=conversation.name:
        raise EvidenceError("conversation source revision mismatch")
    conv_inv=[{k:e[k] for k in ("path","bytes","sha256")} for e in conv["files"]]
    verify_inventory(conversation,conv_inv)
    mapping={"wikipedia/"+spec.filename:raw,
             "wikipedia/"+result_path.name:result_path,
             "wikipedia/"+receipt_path.name:receipt_path,
             "wikipedia/"+spec.filename+".source.json":confined(wiki,spec.filename+".source.json"),
             "conversation/acquisition.json":metadata/"oasst-acquisition.json"}
    for e in conv_inv:mapping["conversation/"+e["path"]]=confined(conversation,e["path"])
    for name in ("dumpstatus.json","dumpstatus.headers.txt","dumpstatus-receipt.json",
                 "upstream-metadata-acquisition.json","upstream-metadata-response-receipts.json",
                 *(f"enwiki-{date}-{suffix}" for suffix in ("index.html","md5sums.txt","sha1sums.txt"))):
        mapping["wikipedia/"+name]=confined(metadata,name)
    for name in ("README.md","LICENSES.md"):mapping[name]=confined(notices,name)
    for source in mapping.values():
        if not source.is_file() or source.is_symlink():raise EvidenceError("raw staging parent missing or symlink")
    output.mkdir(parents=True,exist_ok=False)
    for name,source in mapping.items():
        dest=confined(output,name);dest.parent.mkdir(parents=True,exist_ok=True)
        if source==raw:os.link(source,dest)  # immutable local bytes, no extra 25GB copy
        else:shutil.copyfile(source,dest)
    files=inventory(output,list(mapping))  # hashes every byte, including the complete dump
    plan={"schema":"ovl.raw-archive-plan.v1","repo":REPO,"prefix":"raw/"+digest(files),
          "files":files,"wikipedia_spec":spec.object(),"wikipedia_verified":acquired["verified"]}
    validate_plan(plan)
    write_json(plan_path,plan)
    return {"plan_sha256":digest(plan),"prefix":plan["prefix"],"files":len(files),"bytes":sum(e["bytes"] for e in files)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("wikipedia-raw","conversation-raw","metadata","notices","output","plan"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--date",required=True);a=p.parse_args()
    print(stage(a.wikipedia_raw,a.conversation_raw,a.metadata,a.notices,a.date,a.output,a.plan))
