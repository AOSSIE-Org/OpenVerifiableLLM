"""Complete corpus preparation, admitted only by an externally anchored contract."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import re

from .acquisition import Source, verify_source, wikipedia_source
from .anchoring import PublisherPolicy, verify_anchor
from .canonical import EvidenceError, canonical, confined, digest, file_hash, inventory, read_json, require_digest, verify_inventory, write_json
from .conversations import POLICY as CONVERSATION_POLICY, prepare_oasst
from .data import extract_wikipedia, prepare_stream, rows, train_tokenizer, validate_stream, wikipedia_documents
from .schema import fields, integer

PREPARATION_FILES = ["__init__.py", "acquisition.py", "anchoring.py", "canonical.py", "conversations.py", "data.py", "preparation.py", "schema.py", "source_commitment.py"]
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def preparation_code():
    return inventory(SOURCE_ROOT, ["src/ovl_pipeline/" + n for n in PREPARATION_FILES]
                     + ["requirements/preparation.in", "requirements/preparation.lock",
                        ".github/workflows/anchor-pipeline.yml"])


def preparation_environment():
    lock = SOURCE_ROOT / "requirements/preparation.lock"
    pinned = dict(re.findall(r"^([a-zA-Z0-9_.-]+)==([^\s]+)", lock.read_text(), re.M))
    if not pinned or not {"torch", "sigstore", "pyarrow", "tokenizers", "mwparserfromhell"} <= pinned.keys():
        raise EvidenceError("missing preparation dependency lock")
    installed = {n: importlib.metadata.version(n) for n in pinned}
    if installed != pinned:
        raise EvidenceError("installed preparation packages differ from the hash-locked environment")
    return {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
            "packages": installed, "dependency_lock_sha256": file_hash(lock),
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM")}


def validate_contract(value):
    fields(value, "schema scope run_id attempt_id source_revision wikipedia conversation recipe code environment acquisition_receipts archive", "source contract")
    if value["schema"] != "ovl.source-preparation.v2" or value["scope"] != "production-source-preparation":
        raise EvidenceError("not a production source/preparation commitment")
    for name in ("run_id", "attempt_id"):
        if type(value[name]) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,100}", value[name]):
            raise EvidenceError("invalid source attempt identity")
    if type(value["source_revision"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", value["source_revision"]):
        raise EvidenceError("invalid source revision")
    wiki = value["wikipedia"]
    fields(wiki, "date spec inventory official_status_sha256 metadata_inventory", "Wikipedia source")
    if type(wiki["date"]) is not str or not re.fullmatch(r"[0-9]{8}", wiki["date"]):
        raise EvidenceError("invalid Wikipedia dump date")
    fields(wiki["spec"], "schema url filename bytes upstream_checksums allowed_hosts compression max_uncompressed_bytes", "download spec")
    if wiki["spec"]["schema"] != "ovl.download-spec.v1":
        raise EvidenceError("unsupported source download spec")
    spec = Source(**{k: v for k, v in wiki["spec"].items() if k != "schema"})
    spec.validate()
    expected_name = f"enwiki-{wiki['date']}-pages-articles.xml.bz2"
    if (spec.filename != expected_name or spec.url != f"https://dumps.wikimedia.org/enwiki/{wiki['date']}/{expected_name}"
            or spec.allowed_hosts != ["dumps.wikimedia.org"] or spec.compression != "bz2"
            or set(spec.upstream_checksums) != {"md5", "sha1"}):
        raise EvidenceError("source must be the complete dated monolithic official article dump")
    if type(wiki["inventory"]) is not list or len(wiki["inventory"]) != 1:
        raise EvidenceError("exactly one Wikipedia raw source required")
    fields(wiki["inventory"][0], "path bytes sha256", "Wikipedia inventory entry")
    if wiki["inventory"][0]["path"] != expected_name or wiki["inventory"][0]["bytes"] != spec.bytes:
        raise EvidenceError("Wikipedia inventory/spec mismatch")
    require_digest(wiki["inventory"][0]["sha256"])
    require_digest(wiki["official_status_sha256"])
    metadata_names = {f"enwiki-{wiki['date']}-{n}" for n in ("index.html", "md5sums.txt", "sha1sums.txt")}
    if type(wiki["metadata_inventory"]) is not list or len(wiki["metadata_inventory"]) != 3:
        raise EvidenceError("complete official listing/checksum metadata required")
    if {e.get("path") for e in wiki["metadata_inventory"]} != metadata_names:
        raise EvidenceError("unexpected official metadata inventory")
    for e in wiki["metadata_inventory"]:
        fields(e, "path bytes sha256", "official metadata entry")
        integer(e["bytes"], 1, 16 * 1024 * 1024, "official metadata length")
        require_digest(e["sha256"])
    conv = value["conversation"]
    fields(conv, "repo revision inventory splits", "conversation source")
    if conv["repo"] != "OpenAssistant/oasst1" or not re.fullmatch(r"[0-9a-f]{40}", conv["revision"]):
        raise EvidenceError("unsupported pinned conversation source")
    if type(conv["inventory"]) is not list or not conv["inventory"]:
        raise EvidenceError("missing conversation inventory")
    names = []
    for e in conv["inventory"]:
        fields(e, "path bytes sha256", "conversation inventory entry")
        if type(e["path"]) is not str:
            raise EvidenceError("invalid conversation inventory path")
        integer(e["bytes"], 1, 2**40, "conversation file size")
        require_digest(e["sha256"])
        names.append(e["path"])
    fields(conv["splits"], "train validation", "conversation split map")
    for split, name in conv["splits"].items():
        if type(name) is not str or not re.fullmatch(rf"data/{split}-00000-of-00001-[0-9a-f]+\.parquet", name):
            raise EvidenceError("unexpected conversation split filename")
    if len(names) != 4 or set(names) != {"README.md", "LICENSE", *conv["splits"].values()}:
        raise EvidenceError("conversation inventory must cover both full splits and license/card")
    recipe = value["recipe"]
    fields(recipe, "extractor tokenizer_vocab_size tokenizer_sample_bytes conversation_policy", "preparation recipe")
    if recipe["extractor"] != "main-nonredirect-stripcode-v1" or recipe["conversation_policy"] != CONVERSATION_POLICY:
        raise EvidenceError("unsupported transformation policy")
    integer(recipe["tokenizer_vocab_size"], 261, 65536, "vocabulary budget")
    integer(recipe["tokenizer_sample_bytes"], 1, 2**30, "tokenizer sample budget")
    if value["code"] != preparation_code() or value["environment"] != preparation_environment():
        raise EvidenceError("preparation source/environment differs from commitment")
    if value["environment"]["tokenizers_parallelism"] != "false":
        raise EvidenceError("TOKENIZERS_PARALLELISM=false required")
    fields(value["acquisition_receipts"], "wikipedia conversation", "acquisition receipt roots")
    for root in value["acquisition_receipts"].values():
        require_digest(root)
    archive = value["archive"]
    fields(archive, "repo revision prefix inventory retention_days_target retention_policy", "raw archive")
    if (type(archive["repo"]) is not str or not re.fullmatch(r"AOSSIE/openverifiable-[a-z0-9-]+-evidence", archive["repo"])
            or type(archive["revision"]) is not str or not re.fullmatch(r"[0-9a-f]{40}", archive["revision"])
            or type(archive["prefix"]) is not str or not re.fullmatch(r"raw/[a-z0-9-]+", archive["prefix"])):
        raise EvidenceError("raw archive must be a pinned approved public dataset")
    integer(archive["retention_days_target"], 90, 36500, "raw retention target")
    if archive["retention_policy"] != "owner-preserve-best-effort-public-host-v1":
        raise EvidenceError("unsupported public retention policy")
    if type(archive["inventory"]) is not list or not 10 <= len(archive["inventory"]) <= 32:
        raise EvidenceError("missing raw archive inventory")
    names = []
    for e in archive["inventory"]:
        fields(e, "path bytes sha256", "archive inventory entry")
        confined(Path("."), e["path"])
        integer(e["bytes"], 1, 2**40, "archive file length")
        require_digest(e["sha256"])
        names.append(e["path"])
    if names != sorted(set(names)):
        raise EvidenceError("raw archive inventory must be sorted and unique")
    recorded = {e["path"]: e for e in archive["inventory"]}
    for prefix, inv in (("wikipedia/", wiki["inventory"] + wiki["metadata_inventory"]), ("conversation/", conv["inventory"])):
        for e in inv:
            if recorded.get(prefix + e["path"]) != {**e, "path": prefix + e["path"]}:
                raise EvidenceError("archive does not bind the complete raw sources")
    for name, root in (("wikipedia/dumpstatus.json", wiki["official_status_sha256"]),
                       ("wikipedia/" + spec.filename + ".verified.json", value["acquisition_receipts"]["wikipedia"]),
                       ("conversation/acquisition.json", value["acquisition_receipts"]["conversation"])):
        if recorded.get(name, {}).get("sha256") != root:
            raise EvidenceError("archive does not bind source metadata")
    if not {"README.md", "LICENSES.md"} <= recorded.keys():
        raise EvidenceError("raw archive requires attribution and retention notices")
    return spec


def validate_source_metadata(value, wiki_raw: Path, conversation_raw: Path):
    """Validate metadata parent links, not the complete raw bytes or network history."""
    spec = validate_contract(value)
    # Follow every raw-input metadata/receipt parent before transformation. The
    # receipts remain operator acquisition evidence; their claims are also
    # checked against the actual complete raw bytes below.
    status_path = confined(wiki_raw, "dumpstatus.json")
    if file_hash(status_path) != value["wikipedia"]["official_status_sha256"]:
        raise EvidenceError("official status parent hash mismatch")
    if wikipedia_source(read_json(status_path, canonical_required=False), value["wikipedia"]["date"]).object() != spec.object():
        raise EvidenceError("complete official inventory differs from source contract")
    verify_inventory(wiki_raw, value["wikipedia"]["metadata_inventory"], max_bytes=48 * 1024 * 1024)
    for kind in ("md5", "sha1"):
        path = confined(wiki_raw, f"enwiki-{value['wikipedia']['date']}-{kind}sums.txt")
        matches = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] in (spec.filename, "*" + spec.filename):
                matches.append(parts[0])
        if matches != [spec.upstream_checksums[kind]]:
            raise EvidenceError("official checksum file disagrees with completed inventory")
    result_path = confined(wiki_raw, spec.filename + ".verified.json")
    if file_hash(result_path) != value["acquisition_receipts"]["wikipedia"]:
        raise EvidenceError("Wikipedia acquisition parent hash mismatch")
    acquired = read_json(result_path)
    fields(acquired, "schema spec_root verified network_performed receipt receipt_sha256", "Wikipedia acquisition result")
    if acquired["schema"] != "ovl.acquisition-result.v1" or acquired["spec_root"] != digest(spec.object()):
        raise EvidenceError("Wikipedia acquisition inventory parent mismatch")
    receipt_path = confined(wiki_raw, acquired["receipt"])
    if file_hash(receipt_path) != acquired["receipt_sha256"]:
        raise EvidenceError("Wikipedia network receipt parent mismatch")
    receipt = read_json(receipt_path)
    if (receipt.get("schema") != "ovl.acquisition-receipt.v1" or receipt.get("result") != "PASS"
            or receipt.get("requested_url") != spec.url or receipt.get("spec_root") != digest(spec.object())
            or receipt.get("verified") != acquired["verified"]
            or receipt.get("downloader_code_sha256") != file_hash(Path(__file__).with_name("acquisition.py"))):
        raise EvidenceError("Wikipedia acquisition receipt does not bind declared source/code")
    verified = acquired["verified"]
    fields(verified, "bytes hashes decompressed_bytes", "acquired source verification")
    if (verified["bytes"] != spec.bytes or verified["hashes"] != {
            **spec.upstream_checksums, "sha256": value["wikipedia"]["inventory"][0]["sha256"]}):
        raise EvidenceError("acquisition assertion differs from committed source")
    integer(verified["decompressed_bytes"], 1, spec.max_uncompressed_bytes, "decompressed size assertion")
    archive = {e["path"]: e for e in value["archive"]["inventory"]}
    if archive.get("wikipedia/" + acquired["receipt"], {}).get("sha256") != acquired["receipt_sha256"]:
        raise EvidenceError("archive omits the network receipt parent")
    conversation_receipt = confined(conversation_raw, "acquisition.json")
    if file_hash(conversation_receipt) != value["acquisition_receipts"]["conversation"]:
        raise EvidenceError("conversation acquisition parent hash mismatch")
    acquired_conv = read_json(conversation_receipt, canonical_required=False)
    conv = value["conversation"]
    if (acquired_conv.get("schema") != "ovl.oasst-acquisition-survey.v1"
            or acquired_conv.get("repo") != conv["repo"] or acquired_conv.get("revision") != conv["revision"]):
        raise EvidenceError("conversation acquisition source identity mismatch")
    recorded = [{k: e[k] for k in ("path", "bytes", "sha256")} for e in acquired_conv["files"]]
    if recorded != conv["inventory"]:
        raise EvidenceError("conversation acquisition inventory mismatch")
    for entry in acquired_conv["files"]:
        if entry["path"].endswith(".parquet"):
            if entry.get("upstream_lfs_sha256") != entry["sha256"]:
                raise EvidenceError("conversation LFS parent mismatch")
        else:
            path = confined(conversation_raw, entry["path"])
            blob = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
            with path.open("rb") as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    blob.update(block)
            if blob.hexdigest() != entry.get("upstream_blob_id"):
                raise EvidenceError("conversation Git blob parent mismatch")
    return spec, acquired, acquired_conv


def build_prepared(value, wiki_raw: Path, conversation_raw: Path, output: Path):
    """Internal transformation kernel. Production uses prepare_committed.

    Metadata checks never substitute for complete raw hashing and decompression.
    """
    if wiki_raw.is_symlink() or conversation_raw.is_symlink() or output.exists():
        raise EvidenceError("raw roots must not be symlinks; output must be fresh")
    spec, acquired, _ = validate_source_metadata(value, wiki_raw, conversation_raw)
    verify_inventory(wiki_raw, value["wikipedia"]["inventory"])
    verify_inventory(conversation_raw, value["conversation"]["inventory"])
    if verify_source(confined(wiki_raw, spec.filename), spec) != acquired["verified"]:
        raise EvidenceError("raw acquisition differs from committed verification")
    conv = value["conversation"]
    output.mkdir(parents=True, exist_ok=False)
    corpus = extract_wikipedia([confined(wiki_raw, spec.filename)], output / "corpus")
    recipe = value["recipe"]
    tokenizer = train_tokenizer(output / "corpus/articles.jsonl", output / "tokenizer",
                                vocab_size=recipe["tokenizer_vocab_size"], sample_bytes=recipe["tokenizer_sample_bytes"])
    token_path = output / "tokenizer/tokenizer.json"
    wiki = prepare_stream(wikipedia_documents(output / "corpus/articles.jsonl"), token_path,
                          output / "wikipedia", phase="wikipedia")
    selection = prepare_oasst(conversation_raw, conv["inventory"], conv["splits"], output / "conversation-selection")
    streams = {"wikipedia": wiki}
    for split in ("train", "validation"):
        name = "conversation" if split == "train" else "conversation-validation"
        streams[name] = prepare_stream(rows(output / f"conversation-selection/{split}.jsonl"),
                                       token_path, output / name, phase="conversation")
    for name, stream in streams.items():
        validate_stream(output / name, stream)
    result = {"schema": "ovl.complete-preparation.v1", "source_commitment_sha256": digest(value),
              "corpus": corpus, "tokenizer": tokenizer, "conversation_selection": selection,
              "streams": streams, "code": value["code"], "environment": value["environment"],
              "validation_used_for_training": False}
    write_json(output / "preparation.json", result)
    return result


def prepare_committed(statement_path, bundle_path, policy, wiki_raw, conversation_raw, output, *, expected_preparation=None):
    # Always recompute cryptographic admission. A cached local receipt is insufficient.
    admission = verify_anchor(statement_path, bundle_path, policy)
    statement = read_json(statement_path)
    validate_contract(statement)
    if statement["source_revision"] != policy.source_revision:
        raise EvidenceError("source commitment revision differs from publisher certificate policy")
    result = build_prepared(statement, wiki_raw, conversation_raw, output)
    # Admission observations may change with TUF; keep them outside deterministic
    # prepared roots. Reconstruction compares only actual transformation outputs.
    write_json(output / "admission-observation.json", admission)
    if expected_preparation is not None and result != expected_preparation:
        raise EvidenceError("full preparation reconstruction mismatch; fresh output preserved")
    return {"result": "PASS", "scope": "complete-source-preparation",
            "preparation_sha256": digest(result), "source_commitment_sha256": digest(statement),
            "full_reconstruction_compared": expected_preparation is not None,
            "training_replay": "NOT_RUN", "production_training_admission": "NOT_RUN"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("statement", "bundle", "trust-policy", "wikipedia-raw", "conversation-raw", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--compare-preparation", type=Path)
    args = parser.parse_args()
    try:
        result = prepare_committed(args.statement, args.bundle, PublisherPolicy(**read_json(args.trust_policy)),
                                   args.wikipedia_raw, args.conversation_raw, args.output,
                                   expected_preparation=read_json(args.compare_preparation) if args.compare_preparation else None)
    except Exception as e:
        print(canonical({"result": "FAIL", "reason": str(e)}).decode())
        return 1
    print(canonical(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
