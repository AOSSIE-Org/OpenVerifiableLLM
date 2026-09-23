"""Deterministic streaming preparation and bounded-memory target scheduling."""
from __future__ import annotations

import bz2
from collections import Counter
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import torch

from . import schema
from .extraction_workers import OrderedExtraction
from .canonical import EvidenceError, Merkle, canonical, digest, file_hash, inventory, parse_json, sha256, write_json

PAD, BOS, EOS, USER, ASSISTANT, OFFSET = range(6)


def rows(path):
    with Path(path).open("rb") as f:
        while line := f.readline(16 * 1024 * 1024 + 1):
            if not line.endswith(b"\n"):
                raise EvidenceError("truncated or oversized JSONL record")
            yield parse_json(line)


def write_row(f, row):
    f.write(canonical(row) + b"\n")


def extract_wikipedia(sources: list[Path], output: Path, *, workers=8):
    """Page order is source-list order, then XML order. No network expansion."""
    output.mkdir(parents=True, exist_ok=False)
    counts, root = Counter(), Merkle()
    parsed_by_source, emitted_by_source = Counter(), Counter()
    extraction = None
    db = sqlite3.connect(output / "seen.sqlite")
    db.execute("CREATE TABLE ids (page TEXT PRIMARY KEY, revision TEXT UNIQUE)")
    try:
        with (output / "articles.jsonl").open("wb") as articles, (output / "ledger.jsonl").open("wb") as ledger:
            def emit(record, reason, text):
                source_no = record['source_order']
                if record['ordinal'] != emitted_by_source[source_no]:
                    raise EvidenceError('extraction ledger ordinal gap or duplicate')
                if reason is None:
                    reason = "included" if text else "empty_extracted_text"
                    if reason == "included":
                        article = {**record, "text": text, "text_sha256": sha256(text.encode("utf-8"))}
                        write_row(articles, article)
                        record["text_sha256"] = article["text_sha256"]
                record["reason"] = reason
                write_row(ledger, record)
                root.add(canonical(record))
                counts[reason] += 1
                emitted_by_source[source_no] += 1
            with OrderedExtraction(emit, workers=workers) as extraction:
                for source_no, source in enumerate(sources):
                    opener = bz2.open if source.suffix == ".bz2" else open
                    source_digest = file_hash(source)
                    with opener(source, "rb") as f:
                        # Entity declarations are forbidden. Wikimedia dumps have no DTD.
                        from defusedxml.ElementTree import iterparse
                        parser = iterparse(f, events=("start", "end"), forbid_dtd=True, forbid_entities=True, forbid_external=True)
                        _, docroot = next(parser)
                        ns = docroot.tag.split("}")[0] + "}" if "}" in docroot.tag else ""
                        ordinal = 0
                        for event, page in parser:
                            if event != "end" or page.tag != ns + "page":
                                continue
                            def val(node, key, required=True):
                                child = node.find(ns + key)
                                if child is None or child.text is None:
                                    if required:
                                        raise EvidenceError(f"missing XML field: {key}")
                                    return ""
                                return child.text
                            revisions = page.findall(ns + "revision")
                            if len(revisions) != 1:
                                raise EvidenceError("expected exactly one current revision per page")
                            rev = revisions[0]
                            pid, rid = val(page, "id"), val(rev, "id")
                            if any(not re.fullmatch(r"[1-9][0-9]{0,31}", i) for i in (pid, rid)):
                                raise EvidenceError("invalid Wikipedia page/revision ID")
                            try:
                                db.execute("INSERT INTO ids VALUES (?,?)", (pid, rid))
                            except sqlite3.IntegrityError as e:
                                raise EvidenceError("duplicate page/revision identity") from e
                            raw = val(rev, "text", required=False)
                            record = {"source": source_digest, "source_order": source_no, "ordinal": ordinal,
                                      "page_id": pid, "revision_id": rid, "title": val(page, "title"),
                                      "namespace": int(val(page, "ns")), "timestamp": val(rev, "timestamp"),
                                      "attribution_url": f"https://en.wikipedia.org/w/index.php?curid={pid}",
                                      "revision_url": f"https://en.wikipedia.org/w/index.php?oldid={rid}",
                                      "history_url": f"https://en.wikipedia.org/w/index.php?curid={pid}&action=history",
                                      "raw_text_sha256": sha256(raw.encode("utf-8"))}
                            model = val(rev, "model", required=False)
                            if record["namespace"] != 0:
                                reason = "non_main_namespace"
                            elif page.find(ns + "redirect") is not None:
                                reason = "redirect"
                            elif model != "wikitext":
                                reason = "non_wikitext_model"
                            elif not raw:
                                reason = "empty_revision_text"
                            else:
                                reason = None
                            parsed_by_source[source_no] += 1
                            extraction.submit(record, reason, raw if reason is None else None)
                            ordinal += 1
                            page.clear()
                            docroot.clear()
                    db.commit()
            if parsed_by_source != emitted_by_source or extraction.pending or extraction.pending_bytes:
                raise EvidenceError('parsed pages and emitted ledger do not reconcile')
    except Exception as error:
        # Failed partial bytes are diagnostic, never a reconstructible corpus.
        # Do not drain after parser/worker/output failure or hide the first error.
        try:
            write_json(output / 'extraction-failure.json', {
                'schema': 'ovl.extraction-failure.v1', 'result': 'FAIL',
                'error_type': type(error).__name__,
                'parsed_by_source': {str(k):v for k,v in parsed_by_source.items()},
                'emitted_by_source': {str(k):v for k,v in emitted_by_source.items()},
                'worker_observation': extraction.observation() if extraction else None,
                'scope': 'operator diagnostic; pending pages discarded on abort, no successful corpus'})
        except Exception:
            pass  # Evidence-storage failure must not suppress the original error.
        raise
    finally:
        db.close()
        (output / "seen.sqlite").unlink()
    manifest = {"schema": "ovl.corpus.v1", "policy": "main-nonredirect-stripcode-v1",
                "sources": [file_hash(p) for p in sources], "counts": dict(counts),
                "record_count": sum(counts.values()), "ledger_root": root.root(),
                "files": inventory(output, ["articles.jsonl", "ledger.jsonl"])}
    if not counts["included"]:
        raise EvidenceError("empty eligible corpus")
    write_json(output / "corpus.json", manifest)
    return manifest


def train_tokenizer(articles: Path, output: Path, *, vocab_size=32000, sample_bytes=16_000_000):
    if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
        raise EvidenceError("tokenizer reconstruction requires TOKENIZERS_PARALLELISM=false")
    if not OFFSET + 256 <= vocab_size <= 65536 or sample_bytes < 1:
        raise EvidenceError("invalid tokenizer bounds")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size - OFFSET, min_frequency=2, show_progress=False,
                                  initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()), special_tokens=[])
    members, count = Merkle(), 0
    def sample():
        nonlocal count
        for row in rows(articles):
            n = len(row["text"].encode("utf-8"))
            # Take the longest prefix of complete articles fitting the bound, with
            # one first article even if it exceeds the target (record actual bytes).
            if count and count + n > sample_bytes:
                break
            members.add(canonical({"page_id": row["page_id"], "text_sha256": row["text_sha256"]}))
            count += n
            yield row["text"]
    tokenizer.train_from_iterator(sample(), trainer)
    # Canonicalize JSON object ordering from Rust's maps. Merge array order remains bound.
    encoded = json.loads(tokenizer.to_str())
    write_json(output / "tokenizer.json", encoded)
    manifest = {"schema": "ovl.tokenizer.v1", "backend": "tokenizers-bytelevel-bpe",
                "sample_policy": "ordered-whole-article-prefix-v1", "sample_byte_budget": sample_bytes,
                "sample_actual_bytes": count, "sample_articles": members.count, "sample_root": members.root(),
                "vocab_size": tokenizer.get_vocab_size() + OFFSET, "text_id_offset": OFFSET,
                "controls": {"pad": PAD, "bos": BOS, "eos": EOS, "user": USER, "assistant": ASSISTANT},
                "files": inventory(output, ["tokenizer.json"])}
    write_json(output / "tokenizer-manifest.json", manifest)
    return manifest


def text_ids(tokenizer, text):
    raw_ids = tokenizer.encode(text, add_special_tokens=False).ids
    if tokenizer.decode(raw_ids, skip_special_tokens=False) != text:
        raise EvidenceError("tokenizer roundtrip mismatch")
    ids = [i + OFFSET for i in raw_ids]
    if any(i >= 65536 for i in ids):
        raise EvidenceError("uint16 token overflow")
    return ids


def prepare_stream(documents, tokenizer_path: Path, output: Path, *, phase):
    """Documents provide identity, text (wiki) or explicit role messages (chat)."""
    if phase not in ("wikipedia", "conversation"):
        raise EvidenceError("unknown phase")
    output.mkdir(parents=True, exist_ok=False)
    tok = Tokenizer.from_file(str(tokenizer_path))
    offset = targets = count = 0
    root = Merkle()
    with (output / "tokens.u16").open("wb") as tf, (output / "mask.u8").open("wb") as mf, (output / "documents.jsonl").open("wb") as index:
        for d in documents:
            if phase == "wikipedia":
                tokens = text_ids(tok, d["text"]) + [EOS]
                mask = [1] * len(tokens)
            else:
                tokens, mask = [], []
                messages = d["messages"]
                if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant":
                    raise EvidenceError("conversation must start user and end assistant")
                for i, msg in enumerate(messages):
                    role = "user" if i % 2 == 0 else "assistant"
                    if msg["role"] != role:
                        raise EvidenceError("roles must alternate")
                    body = text_ids(tok, msg["text"]) + [EOS]
                    tokens.extend([USER if role == "user" else ASSISTANT] + body)
                    mask.extend([0] + [int(role == "assistant")] * len(body))
            n = sum(mask)
            if not n:
                raise EvidenceError("document without targets")
            row = {"identity": d["identity"], "offset": offset, "tokens": len(tokens),
                   "target_start": targets, "targets": n}
            write_row(index, row)
            root.add(canonical(row))
            tf.write(np.array(tokens, dtype="<u2").tobytes())
            mf.write(bytes(mask))
            offset += len(tokens)
            targets += n
            count += 1
    if not count:
        raise EvidenceError("empty token stream")
    manifest = {"schema": "ovl.stream.v1", "phase": phase, "token_dtype": "uint16-le",
                "tokenizer_sha256": file_hash(tokenizer_path), "documents": count, "tokens": offset,
                "targets": targets, "index_root": root.root(), "window_policy": "per-document-overlap-one-v1",
                "files": inventory(output, ["tokens.u16", "mask.u8", "documents.jsonl"])}
    write_json(output / "stream.json", manifest)
    return manifest


def wikipedia_documents(articles):
    for row in rows(articles):
        yield {"identity": [row["source"], row["page_id"], row["revision_id"]], "text": row["text"]}


def stream_arrays(directory):
    return (np.memmap(directory / "tokens.u16", dtype="<u2", mode="r"),
            np.memmap(directory / "mask.u8", dtype=np.uint8, mode="r"))


def validate_stream(directory: Path, manifest, *, inventory_progress=None):
    """Independent count/layout validator; never calls the trainer's window code."""
    from .canonical import verify_inventory
    schema.stream(manifest)
    if inventory_progress is None:verify_inventory(directory, manifest["files"])
    else:verify_inventory(directory, manifest["files"],progress=inventory_progress)
    tok, masks = stream_arrays(directory)
    cursor = target = docs = 0
    with tempfile.TemporaryDirectory(prefix="ovl-index-check-") as temp:
        seen = sqlite3.connect(str(Path(temp) / "identities.sqlite"))
        seen.execute("PRAGMA cache_size=-8192")
        seen.execute("CREATE TABLE ids (identity TEXT PRIMARY KEY)")
        try:
            tree = Merkle()
            for row in rows(directory / "documents.jsonl"):
                identity = digest(row["identity"])
                try:
                    seen.execute("INSERT INTO ids VALUES (?)", (identity,))
                except sqlite3.IntegrityError as e:
                    raise EvidenceError("duplicate document") from e
                schema.fields(row, "identity offset tokens target_start targets", "document index")
                for field in ("offset", "target_start", "tokens", "targets"):
                    schema.integer(row[field], 0 if field in ("offset", "target_start") else 1, 2**53 - 1, field)
                if row["offset"] != cursor or row["target_start"] != target or row["tokens"] <= 0:
                    raise EvidenceError("noncontiguous document index")
                n = row["tokens"]
                mask = masks[cursor:cursor+n]
                if len(mask) != n or not np.all((mask == 0) | (mask == 1)) or int(mask.sum()) != row["targets"]:
                    raise EvidenceError("loss-mask accounting mismatch")
                if manifest["phase"] == "wikipedia" and (row["targets"] != n or tok[cursor+n-1] != EOS):
                    raise EvidenceError("Wikipedia target/EOS mismatch")
                tree.add(canonical(row))
                cursor += n
                target += row["targets"]
                docs += 1
        finally:
            seen.close()
    if (cursor, target, docs, tree.root()) != (manifest["tokens"], manifest["targets"], manifest["documents"], manifest["index_root"]):
        raise EvidenceError("stream totals/root mismatch")
    if cursor != len(tok) or cursor != len(masks) or target <= 0:
        raise EvidenceError("empty or trailing token data")
    return target


def batches(directory: Path, context: int, batch_size: int):
    if context < 1 or batch_size < 1:
        raise EvidenceError("invalid batch dimensions")
    tok, mask = stream_arrays(directory)
    pending = []
    for d in rows(directory / "documents.jsonl"):
        target = d["target_start"]
        for start in range(0, d["tokens"], context):
            n = min(context, d["tokens"] - start)
            absolute = d["offset"] + start
            y = np.array(tok[absolute:absolute+n], dtype=np.int64)
            valid = np.array(mask[absolute:absolute+n], dtype=np.bool_)
            ids = np.full(n, -1, dtype=np.int64)
            ids[valid] = np.arange(target, target + int(valid.sum()))
            target += int(valid.sum())
            if not valid.any():
                continue
            previous = BOS if start == 0 else int(tok[absolute - 1])
            x = np.concatenate(([previous], y[:-1]))
            pending.append((x, y, valid, ids))
            if len(pending) == batch_size:
                yield _collate(pending, context)
                pending = []
    if pending:
        yield _collate(pending, context)


def _collate(windows, context):
    shape = (len(windows), context)
    out = {"inputs": torch.full(shape, PAD, dtype=torch.int64), "targets": torch.full(shape, PAD, dtype=torch.int64),
           "mask": torch.zeros(shape, dtype=torch.bool), "target_ids": torch.full(shape, -1, dtype=torch.int64)}
    for row, window in enumerate(windows):
        for name, arr in zip(out, window):
            out[name][row, :len(arr)] = torch.from_numpy(arr.copy())
    return out


def check_coverage(batch, cursor, total):
    ids = batch["target_ids"][batch["mask"]]
    if ids.numel() == 0 or cursor + ids.numel() > total:
        raise EvidenceError("empty/excess coverage")
    if not torch.equal(ids, torch.arange(cursor, cursor + ids.numel(), dtype=torch.int64)):
        raise EvidenceError("target gap, repetition or reordering")
    if not torch.all(batch["target_ids"][~batch["mask"]] == -1):
        raise EvidenceError("masked positions carry target IDs")
    return cursor + ids.numel()
