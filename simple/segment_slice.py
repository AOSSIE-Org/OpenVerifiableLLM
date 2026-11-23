#!/usr/bin/env python3
"""CPU-only verification of a prepared OpenVerifiableLLM Wikipedia segment slice."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

CONTEXT, BATCH_SIZE = 512, 32
PAD, BOS, EOS = 0, 1, 2

def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def make_inputs(windows, tok, mask):
    cols = np.arange(CONTEXT, dtype=np.int64)
    a, n, base, prev = (np.array(c, dtype=np.int64) for c in zip(*windows, strict=True))
    inside = cols < n[:, None]
    idx = np.where(inside, a[:, None] + cols, a[:, None])
    y = np.where(inside, tok[idx].astype(np.int64), PAD)
    valid = inside & (mask[idx] != 0)
    x = np.empty_like(y)
    x[:, 0] = prev
    x[:, 1:] = y[:, :-1]
    x = np.where(inside, x, PAD)
    ids = np.where(valid, base[:, None] + np.cumsum(valid, axis=1) - 1, -1)
    return np.ascontiguousarray(x), np.ascontiguousarray(ids)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify a prepared Wikipedia segment against the published chain."
    )
    parser.add_argument("--start", type=int, required=True, help="Starting chain entry")
    parser.add_argument("--end", type=int, required=True, help="Ending chain entry")
    parser.add_argument(
        "--data", type=Path, required=True,
        help="Directory containing chain.jsonl and segment slice files",
    )
    args = parser.parse_args(argv)

    check(args.start >= 0, "--start must be non-negative.")
    check(args.end == args.start + 1, "--end must immediately follow --start.")

    data_dir = args.data.resolve()
    docs = data_dir / f"segment-{args.start}-{args.end}-documents.jsonl"
    tokens = data_dir / f"segment-{args.start}-{args.end}-tokens.u16"
    mask_file = data_dir / f"segment-{args.start}-{args.end}-mask.u8"
    chain_path = data_dir / "chain.jsonl"
    stream_path = data_dir / "source" / "wikipedia-stream.json"

    for p in (docs, tokens, mask_file, chain_path, stream_path):
        check(p.is_file(), f"Missing required file: {p}")
    check(sys.byteorder == "little", "Expected little-endian host.")

    chain = []
    for line_number, line in enumerate(
        chain_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            chain.append(json.loads(line))
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"Invalid JSON in chain.jsonl at line {line_number}: {err.msg}"
            ) from err

    check(bool(chain), "chain.jsonl contains no entries.")

    # Validate the complete hash-linked chain using the production rule.
    for i, entry in enumerate(chain):
        expected_previous = (
            hashlib.sha256(
                json.dumps(chain[i - 1], sort_keys=True).encode()
            ).hexdigest()
            if i > 0 else None
        )
        check(entry.get("index") == i, f"Invalid chain index at entry {i}.")
        check(
            entry.get("previous") == expected_previous,
            f"Chain link broken at entry {i}.",
        )

    check(args.end < len(chain), f"Chain has no entry {args.end}.")

    manifest_bytes = stream_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    initial_data = chain[0].get("data")
    expected_manifest_hash = (
        initial_data.get("wikipedia") if isinstance(initial_data, dict) else None
    )
    check(
        manifest_hash == expected_manifest_hash,
        "Wikipedia stream manifest hash does not match chain[0].data.wikipedia.",
    )
    meta = json.loads(manifest_bytes.decode("utf-8"))
    full_bytes = sum(int(x["bytes"]) for x in meta["files"])

    start_entry = chain[args.start]
    end_entry = chain[args.end]
    start = start_entry["control"]
    end = end_entry["control"]

    doc_start, first = map(int, start["pos"])
    doc_end, end_window = map(int, end["pos"])
    token_start, token_end = int(start["cursor"]), int(end["cursor"])

    check(start["phase"] == end["phase"] == "wikipedia", "Wrong chain phase.")
    check(first == 0 and end_window == 0, "Expected document-window boundary positions.")
    check(token_end > token_start, "Segment has no token range.")
    check(doc_end > doc_start, "Segment has no document range.")
    check(
        docs.stat().st_size == doc_end - doc_start,
        "Document slice byte size mismatch.",
    )

    tok = np.memmap(tokens, dtype="<u2", mode="r")
    mask = np.memmap(mask_file, dtype=np.uint8, mode="r")
    expected_tokens = token_end - token_start
    check(len(tok) == expected_tokens, "Token slice length mismatch.")
    check(len(mask) == expected_tokens, "mask slice length mismatch.")

    running = start["batches"]
    cursor = token_start
    batches = documents = mask_targets = 0
    expected_offset = expected_target = token_start
    pending = []
    last_after = [0, 0]

    def consume():
        nonlocal running, cursor, batches, pending
        inputs, ids = make_inputs(pending, tok, mask)
        selected = ids[ids != -1]
        check(selected.size > 0, f"Batch {batches + 1} is empty.")
        check(int(selected[0]) == cursor, f"Batch {batches + 1}: first target ID mismatch.")
        check(int(selected[-1]) == cursor + selected.size - 1,
              f"Batch {batches + 1}: last target ID mismatch.")
        check(np.all(np.diff(selected) == 1), f"Batch {batches + 1}: target IDs are not contiguous.")
        bh = hashlib.sha256(inputs.tobytes(order="C")).digest()
        running = hashlib.sha256(bytes.fromhex(running) + bh).hexdigest()
        cursor += int(selected.size)
        batches += 1
        pending = []

    with docs.open("rb") as f:
        first_window = first
        while True:
            row = f.tell()
            line = f.readline()
            if not line:
                break
            d = json.loads(line)
            n_tok = int(d["tokens"])
            targets = int(d["targets"])
            source_offset = int(d["offset"])
            target_start = int(d["target_start"])
            check(n_tok > 0, f"Empty document row {documents}.")
            check(source_offset == expected_offset, f"Token offset gap at row {documents}.")
            check(target_start == expected_target, f"Target cursor gap at row {documents}.")
            check(targets == n_tok, f"Wikipedia targets != tokens at row {documents}.")

            local = source_offset - token_start
            check(local >= 0 and local + n_tok <= len(tok), f"Row {documents} outside token slice.")
            rm = mask[local:local + n_tok]
            check(not np.any((rm != 0) & (rm != 1)), f"Non-binary mask at row {documents}.")
            count = int(np.count_nonzero(rm))
            check(count == targets, f"mask target count mismatch at row {documents}.")
            check(int(tok[local + n_tok - 1]) == EOS, f"EOS missing at row {documents}.")
            mask_targets += count

            starts = np.arange(0, n_tok, CONTEXT, dtype=np.int64)
            counts = np.add.reduceat(rm != 0, starts, dtype=np.int64)
            bases = target_start + np.concatenate(
                (np.array([0], dtype=np.int64), np.cumsum(counts)[:-1])
            )
            for s, c, base in zip(starts.tolist(), counts.tolist(), bases.tolist(), strict=True):
                if s < first_window or c == 0:
                    continue
                absolute = local + s
                previous = BOS if s == 0 else int(tok[absolute - 1])
                pending.append((absolute, min(CONTEXT, n_tok - s), int(base), previous))
                if len(pending) == BATCH_SIZE:
                    last_after = [row, s + CONTEXT] if s + CONTEXT < n_tok else [f.tell(), 0]
                    consume()
            first_window = 0
            expected_offset += n_tok
            expected_target += targets
            documents += 1

    if pending:
        consume()
        last_after = [docs.stat().st_size, 0]
    check(documents > 0, "No document rows processed.")
    computed_pos = [doc_start + int(last_after[0]), int(last_after[1])]
    check(expected_offset == token_end, f"Document token range does not end at chain[{args.end}].")
    check(expected_target == token_end and cursor == token_end, f"Target cursor does not end at chain[{args.end}].")
    check(computed_pos == [doc_end, end_window], f"Final position mismatch: {computed_pos}.")
    expected_batches = int(end["global_step"]) - int(start["global_step"])
    check(batches == expected_batches, f"Batch count {batches} != chain delta {expected_batches}.")
    check(running == end["batches"], f"ROLLING BATCH HASH DOES NOT MATCH chain[{args.end}].")

    slice_bytes = docs.stat().st_size + tokens.stat().st_size + mask_file.stat().st_size

    print("REAL SEGMENT CPU REPLAY: PASS")
    print(f"Segment: {args.start} -> {args.end} ({start['phase']})")
    print(f"Documents: {documents:,}")
    print(f"Targets covered: {cursor - token_start:,}")
    print(f"Batches reproduced: {batches:,}")
    print(f"Final cursor: {cursor:,} (expected {token_end:,})")
    print(f"Final position: {computed_pos}")
    print(f"Computed batches hash: {running}")
    print(f"Expected batches hash: {end['batches']}")
    print()
    print("DOCUMENT / mask VALIDATION: PASS")
    print(f"mask targets: {mask_targets:,}")
    print(f"All mask bytes are 1: {bool(np.all(mask == 1))}")
    print("Every row: contiguous offsets, mask count matches targets, EOS present")
    print()
    print("SLICE SIZE")
    print(f"documents.jsonl: {docs.stat().st_size:,} bytes")
    print(f"tokens.u16:      {tokens.stat().st_size:,} bytes")
    print(f"mask.u8:         {mask_file.stat().st_size:,} bytes")
    print(f"Total slice:     {slice_bytes:,} bytes ({slice_bytes / 2**20:,.2f} MiB)")
    print(f"Full Wikipedia prepared stream: {full_bytes:,} bytes ({full_bytes / 1e9:,.3f} GB)")
    print(f"Slice fraction: {100 * slice_bytes / full_bytes:.2f}%")
    print(f"Download reduction: {100 * (1 - slice_bytes / full_bytes):.2f}%")
    print()
    print("All checks passed. No model, optimizer, or GPU was used.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, KeyError, ValueError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
