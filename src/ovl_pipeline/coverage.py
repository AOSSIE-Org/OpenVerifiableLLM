"""Complete update-count census independent of the trainer's batch generator."""
from pathlib import Path

import numpy as np

from . import schema
from .canonical import EvidenceError, digest, read_json
from .data import rows, validate_stream


def schedule_counts(directory: Path, recipe):
    schema.recipe(recipe, gpu=True)
    stream = read_json(directory / "stream.json")
    validate_stream(directory, stream)  # Includes every file hash and every index row.
    masks = np.memmap(directory / "mask.u8", dtype=np.uint8, mode="r")
    context, batch = recipe["context"], recipe["batch_size"]
    windows = targets = documents = occupied = 0
    # Separate implementation: reshape full context blocks and handle the tail.
    # Do not call batches() or count a prefix of the corpus.
    for doc in rows(directory / "documents.jsonl"):
        documents += 1
        mask = masks[doc["offset"]:doc["offset"] + doc["tokens"]]
        full, tail = divmod(len(mask), context)
        if full:
            sums = mask[:full * context].reshape(full, context).sum(axis=1)
            windows += int(np.count_nonzero(sums))
            occupied += int(np.count_nonzero(sums)) * context
            targets += int(sums.sum())
        if tail:
            count = int(mask[full * context:].sum())
            windows += int(count != 0)
            occupied += tail if count else 0
            targets += count
    if targets != stream["targets"] or documents != stream["documents"] or not windows:
        raise EvidenceError("complete schedule census disagrees with stream")
    updates = (windows + batch - 1) // batch
    return {"schema": "ovl.complete-schedule-counts.v1", "scope": "complete-stream-census",
            "phase": stream["phase"], "stream_sha256": digest(stream), "recipe_sha256": digest(recipe),
            "documents": documents, "targets": targets, "target_bearing_windows": windows,
            "updates": updates, "full_batch_updates": windows // batch,
            "final_batch_rows": (windows - 1) % batch + 1,
            "context": context, "batch_size": batch, "padded_positions": windows * context - occupied,
            "masked_context_positions": occupied - targets,
            "training_coverage": "NOT_RUN"}


def main():
    import argparse
    from .canonical import canonical, write_json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stream", type=Path, required=True)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():raise EvidenceError("preserve existing schedule census")
    result = schedule_counts(a.stream, read_json(a.recipe))
    write_json(a.output, result)
    print(canonical({"schedule_sha256": digest(result), "updates": result["updates"], "targets": result["targets"]}).decode())


if __name__ == "__main__":main()
