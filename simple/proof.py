"""Per-article membership proofs against OpenVerifiableLLM chain.jsonl.

Given a Wikipedia article (by title or page ID), proves which training segment
trained on it, or proves that an exclusion fork never saw it, tied to the
cryptographic chain commitments.

Usage:
    python simple/proof.py --data PREP --run RUN --article "Albert Einstein"
    python simple/proof.py --data PREP --run RUN --article 736
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def read_chain(run_dir):
    path = Path(run_dir) / "chain.jsonl"
    if not path.is_file():
        sys.exit(f"{path} not found")
    chain = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            chain.append(json.loads(line))
        except json.JSONDecodeError as err:
            sys.exit(
                f"{path}: line {n} is not valid JSON ({err.msg}); the file is cut off or damaged"
            )
    if not chain:
        sys.exit(f"{path} has no chain entries")
    # Verify sha256 chain links
    for i in range(1, len(chain)):
        want = hashlib.sha256(
            json.dumps(chain[i - 1], sort_keys=True).encode()
        ).hexdigest()
        if chain[i].get("previous") != want:
            sys.exit(f"chain link broken at entry {i}")
    return chain


def find_article(data_dir, query):
    if not query or not query.strip():
        return None
    query = query.strip()
    path = Path(data_dir) / "wikipedia" / "documents.jsonl"
    if not path.is_file():
        sys.exit(f"{path} not found")
    is_id = query.isdigit()
    query_lower = query.lower()
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # identity format: [page_id, title]
            page_id, title = str(d["identity"][0]), str(d["identity"][1])
            if is_id and page_id == query:
                return d
            if title.lower() == query_lower:
                return d
    return None


def verify_data_commitment(data_dir, chain_entry_0):
    """Check that stream.json matches genesis commitment in chain.jsonl entry 0."""
    if chain_entry_0.get("kind") != "initial":
        return False, "chain genesis entry kind is not 'initial'"
    stream_path = Path(data_dir) / "wikipedia" / "stream.json"
    if not stream_path.is_file():
        return False, "stream.json not found"
    stream_bytes = stream_path.read_bytes()
    stream_hash = hashlib.sha256(stream_bytes).hexdigest()
    expected = chain_entry_0.get("data", {}).get("wikipedia")
    if not expected:
        return False, "chain genesis entry has no wikipedia data commitment"
    if stream_hash != expected:
        return (
            False,
            f"stream.json hash {stream_hash} differs from chain genesis {expected}",
        )

    # Verify documents.jsonl matches the hash listed in stream.json
    try:
        stream_meta = json.loads(stream_bytes.decode("utf-8"))
        doc_files = [
            f
            for f in stream_meta.get("files", [])
            if f.get("path") == "documents.jsonl" or f.get("name") == "documents.jsonl"
        ]
        if not doc_files:
            return False, "documents.jsonl not listed in stream.json files table"
        expected_doc_hash = doc_files[0]["sha256"]
        actual_doc_hash = hashlib.sha256(
            (Path(data_dir) / "wikipedia" / "documents.jsonl").read_bytes()
        ).hexdigest()
        if actual_doc_hash != expected_doc_hash:
            return (
                False,
                f"documents.jsonl hash {actual_doc_hash} differs from stream.json {expected_doc_hash}",
            )
    except Exception as e:
        return False, f"failed checking stream.json file table: {e}"

    return True, stream_hash


def build_membership_proof(doc, chain, data_bound, data_hash):
    target_start = doc["target_start"]
    target_count = doc["targets"]
    target_end = target_start + target_count
    page_id, title = doc["identity"][0], doc["identity"][1]

    # Find the Wikipedia phase boundaries
    segments = []
    is_excluded = False
    exclusion_detail = None

    # Scan every entry for exclusions (including if the fork entry is the last entry)
    for k, entry in enumerate(chain):
        exclude_ranges = (
            entry.get("control", {}).get("exclude", {}).get("wikipedia", [])
        )
        for ex_start, ex_len in exclude_ranges:
            if ex_start == target_start and ex_len == target_count and not is_excluded:
                is_excluded = True
                exclusion_detail = {
                    "at_boundary": k,
                    "excluded_range": [ex_start, ex_start + ex_len],
                    "fork_entry_state": entry.get("state"),
                }

    for k in range(len(chain) - 1):
        c_k = chain[k].get("control", {})
        c_next = chain[k + 1].get("control", {})

        if c_k.get("phase") != "wikipedia":
            continue

        cur_start = c_k.get("cursor", 0)
        cur_end = c_next.get("cursor", cur_start)

        # Check interval overlap
        overlap_start = max(target_start, cur_start)
        overlap_end = min(target_end, cur_end)
        if overlap_start < overlap_end:
            segments.append(
                {
                    "segment": k,
                    "boundary_start": k,
                    "boundary_end": k + 1,
                    "cursor_interval": [cur_start, cur_end],
                    "article_overlap_targets": [overlap_start, overlap_end],
                    "boundary_start_state": chain[k].get("state"),
                    "boundary_end_state": chain[k + 1].get("state"),
                    "verify_cmd": f"python simple/ovl.py verify --data PREP --run RUN --segment {k}",
                }
            )

    status = "excluded" if is_excluded else ("trained" if segments else "unreached")

    proof = {
        "article": {
            "title": title,
            "page_id": page_id,
            "targets": {
                "start": target_start,
                "end": target_end,
                "count": target_count,
            },
            "tokens": doc.get("tokens"),
        },
        "membership_status": status,
        "data_provenance": {
            "bound_to_chain": data_bound,
            "stream_genesis_hash": data_hash,
        },
        "segments": [] if is_excluded else [s["segment"] for s in segments],
        "segment_details": [] if is_excluded else segments,
    }

    if is_excluded:
        proof["exclusion_proof"] = exclusion_detail

    return proof


def main():
    p = argparse.ArgumentParser(
        description="Prove Wikipedia article membership in OpenVerifiableLLM training"
    )
    p.add_argument(
        "--data",
        required=True,
        help="Prepared data directory (containing wikipedia/documents.jsonl)",
    )
    p.add_argument(
        "--run", required=True, help="Run directory (containing chain.jsonl)"
    )
    p.add_argument(
        "--article", required=True, help="Article title or Wikipedia page_id"
    )
    a = p.parse_args()

    chain = read_chain(a.run)
    doc = find_article(a.data, a.article)
    if not doc:
        sys.exit(
            f"Article '{a.article}' not found in {Path(a.data) / 'wikipedia' / 'documents.jsonl'}"
        )

    bound, data_hash = verify_data_commitment(a.data, chain[0])
    if not bound:
        sys.exit(f"data commitment check failed: {data_hash}")
    proof = build_membership_proof(doc, chain, bound, data_hash)

    print(json.dumps(proof, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
