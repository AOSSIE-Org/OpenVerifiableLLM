"""Versioned English OASST tree selection with per-message exclusion accounting."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .canonical import EvidenceError, Merkle, canonical, digest, file_hash, inventory, sha256, verify_inventory, write_json
from .data import write_row

COLUMNS = ["message_id", "parent_id", "text", "role", "lang", "review_result", "deleted", "rank", "message_tree_id"]
POLICY = "oasst-en-preferred-path-v1"


def parquet_messages(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if not set(COLUMNS) <= set(parquet.schema.names):
        raise EvidenceError("required OASST columns absent")
    for batch in parquet.iter_batches(batch_size=4096, columns=COLUMNS):
        yield from batch.to_pylist()


def select_conversations(splits, output: Path):
    """One deterministic path per usable root; train and validation stay separate.

    Rank is ascending with missing rank last, then ID; prompter branches use ID.
    Only approved, undeleted English messages with valid alternating ancestry are
    candidates. A final unanswered prompter is not included. All records have one
    reason code, including branches not selected. Input ordering binds source ordinal;
    selected output order is root-ID order within official split.
    """
    if set(splits) != {"train", "validation"}:
        raise EvidenceError("official train and validation splits required")
    output.mkdir(parents=True, exist_ok=False)
    messages, split_trees = {}, {s: set() for s in splits}
    for split in ("train", "validation"):
        for ordinal, message in enumerate(splits[split]):
            if set(message) != set(COLUMNS):
                raise EvidenceError("unexpected OASST projection")
            for field in ("message_id", "message_tree_id", "text", "role", "lang"):
                if type(message[field]) is not str or not message[field]:
                    raise EvidenceError(f"invalid message field {field}")
            mid = message["message_id"]
            if mid in messages:
                raise EvidenceError("duplicate message identity across source splits")
            if message["parent_id"] is not None and type(message["parent_id"]) is not str:
                raise EvidenceError("invalid parent identity")
            if message["role"] not in ("prompter", "assistant"):
                raise EvidenceError("unknown conversation role")
            if message["rank"] is not None and (type(message["rank"]) is not int or message["rank"] < 0):
                raise EvidenceError("invalid branch rank")
            if type(message["deleted"]) is not bool or (message["review_result"] is not None and type(message["review_result"]) is not bool):
                raise EvidenceError("invalid review/deletion state")
            messages[mid] = {**message, "split": split, "ordinal": ordinal}
            split_trees[split].add(message["message_tree_id"])
            if len(messages) > 2_000_000:
                raise EvidenceError("conversation projection exceeds memory budget")
    if split_trees["train"] & split_trees["validation"]:
        raise EvidenceError("tree identity leakage across official splits")
    children = defaultdict(list)
    intrinsic = {}
    for mid, m in messages.items():
        parent = m["parent_id"]
        if m["deleted"]:
            reason = "deleted"
        elif m["review_result"] is not True:
            reason = "not_approved"
        elif m["lang"] != "en":
            reason = "non_english"
        elif parent is None and (m["role"] != "prompter" or m["message_tree_id"] != mid):
            reason = "invalid_root"
        elif parent is not None and parent not in messages:
            reason = "missing_parent"
        elif parent is not None and (messages[parent]["split"] != m["split"] or messages[parent]["message_tree_id"] != m["message_tree_id"]):
            reason = "parent_tree_mismatch"
        elif parent is not None and messages[parent]["role"] == m["role"]:
            reason = "nonalternating_parent"
        else:
            reason = None
        intrinsic[mid] = reason
        if parent is not None:
            children[parent].append(mid)
    # Iterative ancestry resolution is bounded by message count and detects cycles.
    validity = {}
    for opening in messages:
        if opening in validity:
            continue
        path, seen, current = [], set(), opening
        while current is not None and current not in validity:
            if current in seen:
                raise EvidenceError("cycle in conversation parent graph")
            seen.add(current)
            path.append(current)
            if intrinsic[current] is not None:
                validity[current] = False
                break
            current = messages[current]["parent_id"]
        okay = current is None or validity.get(current, False)
        for mid in reversed(path):
            okay = okay and intrinsic[mid] is None
            validity[mid] = okay
    selected, counts = set(), {s: Counter() for s in splits}
    conversation_counts = Counter()
    for split in ("train", "validation"):
        with (output / f"{split}.jsonl").open("wb") as f:
            roots = sorted(mid for mid, m in messages.items() if m["split"] == split and m["parent_id"] is None and validity[mid])
            for root in roots:
                path, current = [root], root
                while True:
                    candidates = [mid for mid in children[current] if validity[mid]]
                    if not candidates:
                        break
                    if messages[current]["role"] == "prompter":
                        candidates.sort(key=lambda mid: (messages[mid]["rank"] is None, messages[mid]["rank"] or 0, mid))
                    else:
                        candidates.sort()
                    current = candidates[0]
                    path.append(current)
                if messages[path[-1]]["role"] == "prompter":
                    path.pop()
                if not path:
                    continue
                selected.update(path)
                obj = {"identity": [split, root, path], "messages": [
                    {"role": "user" if messages[mid]["role"] == "prompter" else "assistant", "text": messages[mid]["text"]}
                    for mid in path]}
                write_row(f, obj)
                conversation_counts[split] += 1
    ledger = Merkle()
    with (output / "selection.jsonl").open("wb") as f:
        for mid, m in sorted(messages.items(), key=lambda item: (item[1]["split"], item[1]["ordinal"])):
            reason = "selected" if mid in selected else intrinsic[mid] or ("unusable_ancestor" if not validity[mid] else "unselected_branch_or_unanswered_prompt")
            row = {"split": m["split"], "ordinal": m["ordinal"], "message_id": mid,
                   "parent_id": m["parent_id"], "tree_id": m["message_tree_id"], "role": m["role"],
                   "text_sha256": sha256(m["text"].encode("utf-8")), "projection_sha256": digest({k: m[k] for k in COLUMNS}),
                   "reason": reason}
            write_row(f, row)
            ledger.add(canonical(row))
            counts[m["split"]][reason] += 1
    if not conversation_counts["train"] or not conversation_counts["validation"]:
        raise EvidenceError("both official splits must retain nonempty selected conversations")
    manifest = {"schema": "ovl.conversation-selection.v1", "policy": POLICY, "projection_columns": COLUMNS,
                "messages": len(messages), "conversations": dict(conversation_counts),
                "counts": {s: dict(c) for s, c in counts.items()}, "selection_root": ledger.root(),
                "files": inventory(output, ["train.jsonl", "validation.jsonl", "selection.jsonl"])}
    write_json(output / "selection.json", manifest)
    return manifest


def prepare_oasst(raw: Path, source_inventory, split_filenames, output: Path):
    from .canonical import confined
    verify_inventory(raw, source_inventory)
    if set(split_filenames) != {"train", "validation"} or not set(split_filenames.values()) <= {e["path"] for e in source_inventory}:
        raise EvidenceError("split filenames must be covered by source inventory")
    result = select_conversations({s: parquet_messages(confined(raw, filename)) for s, filename in split_filenames.items()}, output)
    result["source_inventory_root"] = digest(source_inventory)
    result["split_filenames"] = split_filenames
    write_json(output / "selection.json", result)
    return result
