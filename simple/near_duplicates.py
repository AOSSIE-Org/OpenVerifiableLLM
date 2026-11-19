"""Near-duplicate and topic-level information leakage analysis for exclusion forks.

Measures whether an exclusion fork (which removed an article as a document) still
learned the entity's facts through other training articles that mention it.
Computes passage-level next-token cross-entropy loss, token-by-token loss attribution,
rarity-matched astronomical control comparisons, and generates a cryptographically
verifiable topic exclusion filter specification.

Usage:
    python simple/near_duplicates.py --data PREP --original RUN --fork FORK [--entity "3I/ATLAS"]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ovl  # noqa: E402
from data import BOS, OFFSET, stream_arrays  # noqa: E402


def find_token_subsequences(arr, sub):
    """Find all starting indices of a 1D token subsequence within a 1D array."""
    if len(sub) == 0 or len(sub) > len(arr):
        return []
    if len(sub) == 1:
        return [int(i) for i in np.where(arr == sub[0])[0]]

    # Vectorized search on candidate starts matching the first token
    candidates = np.where(arr[: -len(sub) + 1] == sub[0])[0]
    matches = []
    for idx in candidates:
        if np.array_equal(arr[idx : idx + len(sub)], sub):
            matches.append(int(idx))
    return matches


def build_entity_patterns(tokenizer, entity, extra_aliases=None):
    """Build token ID sequences for an entity and its aliases, including BPE leading spaces."""
    names = [entity] + list(extra_aliases or [])

    seen_names = []
    for name in names:
        if not name:
            continue
        for variant in (name, " " + name, name.lower(), " " + name.lower()):
            if variant not in seen_names:
                seen_names.append(variant)

    patterns = []
    seen_tokens = set()
    for name in seen_names:
        ids = [
            i + OFFSET
            for i in tokenizer.encode(name, add_special_tokens=False).ids
        ]
        tok_tuple = tuple(ids)
        if tok_tuple and tok_tuple not in seen_tokens:
            seen_tokens.add(tok_tuple)
            patterns.append(
                {"text": name.strip(), "tokens": np.array(ids, dtype=np.uint16)}
            )
    return patterns


def scan_entity_mentions(data_dir, patterns, excluded_pages=None, ctx=None):
    """Scan documents.jsonl and tokens.u16 for training windows containing entity mentions.

    Excludes the primary removed document(s) by page ID so only secondary mentions are measured.
    """
    ctx = ctx or ovl.RECIPE["context"]
    data_path = Path(data_dir) / "wikipedia"
    doc_path = data_path / "documents.jsonl"
    if not doc_path.is_file():
        sys.exit(f"documents file {doc_path} not found")

    tok, mask = stream_arrays(data_path)
    excluded_set = {str(p) for p in (excluded_pages or [])}

    results = []
    with doc_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # documents.jsonl identity format: [source_sha256, page_id, revision_id]
            ident = d.get("identity", [])
            page_id = str(ident[1]) if len(ident) >= 2 else str(ident[0])
            if page_id in excluded_set:
                continue

            doc_offset = d["offset"]
            doc_len = d["tokens"]
            doc_tok = tok[doc_offset : doc_offset + doc_len]

            for pat in patterns:
                matches = find_token_subsequences(doc_tok, pat["tokens"])
                for match_idx in matches:
                    k = match_idx // ctx
                    win_start = doc_offset + k * ctx
                    win_len = min(ctx, doc_len - k * ctx)
                    span_start = match_idx - (k * ctx)
                    span_end = span_start + len(pat["tokens"])

                    # Target offset within target token stream
                    target_start = d["target_start"] + min(
                        k * ctx, d["targets"]
                    )
                    target_len = min(ctx, max(0, d["targets"] - k * ctx))

                    results.append(
                        {
                            "doc_identity": ident,
                            "page_id": page_id,
                            "doc_offset": doc_offset,
                            "window_idx": k,
                            "window_start": win_start,
                            "window_len": win_len,
                            "target_start": target_start,
                            "target_len": target_len,
                            "pattern": pat["text"],
                            "mention_span": (span_start, span_end),
                        }
                    )
    return results


def deduplicate_windows(windows):
    """Keep unique training windows if multiple aliases matched in the same window."""
    unique, seen = [], set()
    for w in windows:
        key = (w["window_start"], w["window_len"])
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return unique


@torch.no_grad()
def eval_passage_loss_and_tokens(
    model_orig, model_fork, tok, mask, win, tokenizer=None
):
    """Evaluate token-level cross-entropy loss under both models on one training window."""
    a = win["window_start"]
    n = win["window_len"]
    y = torch.from_numpy(tok[a : a + n].astype("int64")).to(ovl.DEV)
    prev = BOS if win["window_idx"] == 0 else int(tok[a - 1])
    x = torch.cat((torch.tensor([prev], device=ovl.DEV), y[:-1]))[None]
    m = torch.from_numpy(mask[a : a + n] != 0).to(ovl.DEV)

    with torch.autocast(ovl.DEV, dtype=torch.bfloat16, cache_enabled=False):
        logits_orig = model_orig(x)[0].float()
        logits_fork = model_fork(x)[0].float()

        loss_orig = F.cross_entropy(logits_orig, y, reduction="none")
        loss_fork = F.cross_entropy(logits_fork, y, reduction="none")

    loss_orig_np = loss_orig.cpu().numpy()
    loss_fork_np = loss_fork.cpu().numpy()
    m_np = m.cpu().numpy()

    # Masked mean for the full window
    total_valid = int(m_np.sum())
    if total_valid == 0:
        return None

    win_loss_orig = float(loss_orig_np[m_np].mean())
    win_loss_fork = float(loss_fork_np[m_np].mean())

    # Mention span loss
    s, e = win["mention_span"]
    s = max(0, min(s, n))
    e = max(s, min(e, n))
    span_m = m_np[s:e]

    if span_m.any():
        mention_loss_orig = float(loss_orig_np[s:e][span_m].mean())
        mention_loss_fork = float(loss_fork_np[s:e][span_m].mean())
    else:
        mention_loss_orig = win_loss_orig
        mention_loss_fork = win_loss_fork

    # Token attribution breakdown around the mention
    token_attributions = []
    if tokenizer is not None:
        view_start = max(0, s - 5)
        view_end = min(n, e + 5)
        for i in range(view_start, view_end):
            raw_id = int(y[i].item())
            tok_text = (
                tokenizer.decode([raw_id - OFFSET])
                if raw_id >= OFFSET
                else f"<{raw_id}>"
            )
            token_attributions.append(
                {
                    "token_id": raw_id,
                    "token_text": tok_text,
                    "is_mention": (s <= i < e),
                    "is_target": bool(m_np[i]),
                    "loss_orig": round(float(loss_orig_np[i]), 4),
                    "loss_fork": round(float(loss_fork_np[i]), 4),
                    "delta": round(float(loss_fork_np[i] - loss_orig_np[i]), 4),
                }
            )

    return {
        "window_loss_orig": win_loss_orig,
        "window_loss_fork": win_loss_fork,
        "window_delta": win_loss_fork - win_loss_orig,
        "mention_loss_orig": mention_loss_orig,
        "mention_loss_fork": mention_loss_fork,
        "mention_delta": mention_loss_fork - mention_loss_orig,
        "token_attributions": token_attributions,
    }


def compute_bootstrap_significance(
    target_deltas, control_deltas, n_resamples=1000, seed=1234
):
    """Compute 95% bootstrap confidence interval and permutation p-value.

    Tests null hypothesis H0: target unlearning delta <= control drift delta.
    """
    t_arr = np.asarray(target_deltas, dtype=np.float64)
    c_arr = np.asarray(control_deltas, dtype=np.float64)

    if len(t_arr) == 0:
        return {"error": "no target deltas"}
    if len(c_arr) == 0:
        c_arr = np.zeros(1)

    obs_target = float(t_arr.mean())
    obs_control = float(c_arr.mean())
    obs_diff = obs_target - obs_control

    rng = np.random.default_rng(seed)
    diffs = []
    n_t, n_c = len(t_arr), len(c_arr)
    for _ in range(n_resamples):
        sample_t = rng.choice(t_arr, size=n_t, replace=True)
        sample_c = rng.choice(c_arr, size=n_c, replace=True)
        diffs.append(sample_t.mean() - sample_c.mean())

    diffs = np.array(diffs)
    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    # Permutation test under H0: both samples drawn from the same pooled distribution
    pooled = np.concatenate([t_arr, c_arr])
    perm_diffs = []
    for _ in range(n_resamples):
        permuted = rng.permutation(pooled)
        p_t = permuted[:n_t]
        p_c = permuted[n_t:]
        perm_diffs.append(p_t.mean() - p_c.mean())

    perm_diffs = np.array(perm_diffs)
    p_value = float((perm_diffs >= obs_diff).mean())

    return {
        "target_delta_mean": round(obs_target, 4),
        "control_delta_mean": round(obs_control, 4),
        "net_unlearning": round(obs_diff, 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        "p_value": round(p_value, 4),
        "statistically_significant": bool(p_value < 0.05 and ci_lo > 0),
    }


def compute_filter_code_sha256():
    """Compute deterministic SHA-256 digest of this script file, normalized to LF."""
    source_bytes = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(source_bytes).hexdigest()


def make_topic_filter_spec(
    entity,
    patterns,
    mention_windows,
    stream_targets=None,
    tokenizer_sha256=None,
    aliases=None,
    sample_check_path=None,
    sample_check_data=None,
):
    """Generate a verifiable filter specification for chain.jsonl topic exclusion.

    Merges contiguous or overlapping mention windows so target ranges are
    strictly sorted and non-overlapping. Normalizes script file source to LF line
    endings to guarantee a deterministic SHA-256 across all operating systems.
    """
    code_sha256 = compute_filter_code_sha256()

    raw_ranges = sorted(
        [
            [w["target_start"], w["target_len"]]
            for w in mention_windows
            if w.get("target_len", 0) > 0
        ]
    )

    merged = []
    for start, length in raw_ranges:
        if not merged:
            merged.append([start, length])
        else:
            prev_start, prev_len = merged[-1]
            prev_end = prev_start + prev_len
            if start <= prev_end:
                merged_end = max(prev_end, start + length)
                merged[-1][1] = merged_end - prev_start
            else:
                merged.append([start, length])

    sample_miss_rate = None
    sample_file_sha256 = None
    if sample_check_path is not None:
        sample_path = Path(sample_check_path)
        if not sample_path.is_file():
            sys.exit(f"sample check file not found: {sample_check_path}")
        sample_bytes = sample_path.read_bytes()
        sample_file_sha256 = hashlib.sha256(sample_bytes).hexdigest()
        sample_miss_rate = json.loads(sample_bytes.decode("utf-8"))
    elif sample_check_data is not None:
        sample_miss_rate = sample_check_data

    spec = {
        "filter_name": "topic_near_duplicate_filter",
        "version": "1.0.0",
        "filter_code_sha256": code_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "target_entity": entity,
        "aliases": list(aliases or []),
        "matching_patterns": [p["text"] for p in patterns],
        "excluded_ranges_count": len(merged),
        "total_excluded_targets": sum(l for _, l in merged),
        "ranges": merged,
        "sample_miss_rate": sample_miss_rate,
        "sample_file_sha256": sample_file_sha256,
    }
    validate_topic_filter_spec(spec, stream_targets=stream_targets)
    return spec


def validate_topic_filter_spec(spec, stream_targets=None):
    """Validate that a topic filter spec satisfies all cryptographic and integrity constraints.

    Checks:
    - Required fields: filter_name, version, filter_code_sha256, tokenizer_sha256, target_entity, ranges
    - filter_code_sha256 matches actual sha256 of near_duplicates.py source (normalized to LF)
    - tokenizer_sha256 is present and, if set, is a 64-char hex string
    - sample_miss_rate is null or a dict
    - ranges are sorted, non-overlapping, have positive count, and lie within stream_targets
    """
    required_fields = [
        "filter_name",
        "version",
        "filter_code_sha256",
        "tokenizer_sha256",
        "target_entity",
        "ranges",
    ]
    for f in required_fields:
        if f not in spec:
            raise ValueError(f"missing required field '{f}' in filter spec")

    expected_code_sha = compute_filter_code_sha256()
    if spec["filter_code_sha256"] != expected_code_sha:
        raise ValueError(
            f"filter_code_sha256 mismatch: spec has {spec['filter_code_sha256']}, "
            f"actual code is {expected_code_sha}"
        )

    if spec["tokenizer_sha256"] is not None:
        if (
            not isinstance(spec["tokenizer_sha256"], str)
            or len(spec["tokenizer_sha256"]) != 64
        ):
            raise ValueError(
                f"invalid tokenizer_sha256: {spec['tokenizer_sha256']}"
            )

    if spec.get("sample_miss_rate") is not None and not isinstance(
        spec["sample_miss_rate"], dict
    ):
        raise ValueError(
            f"sample_miss_rate must be null or dict, got {type(spec['sample_miss_rate'])}"
        )

    ranges = spec["ranges"]
    for i, r in enumerate(ranges):
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            raise ValueError(f"range {i} must be [start, length], got {r}")
        start, count = r
        if count <= 0:
            raise ValueError(f"range {i} has non-positive count {count}")
        if start < 0:
            raise ValueError(f"range {i} has negative start {start}")
        if stream_targets is not None and start + count > stream_targets:
            raise ValueError(
                f"range {i} [{start}, {start + count}] exceeds stream targets {stream_targets}"
            )
        if i > 0:
            prev_start, prev_count = ranges[i - 1]
            if start <= prev_start:
                raise ValueError(
                    f"ranges not strictly sorted: range {i-1} start {prev_start} >= range {i} start {start}"
                )
            if prev_start + prev_count > start:
                raise ValueError(
                    f"ranges overlap: range {i-1} end {prev_start + prev_count} > range {i} start {start}"
                )
    return True


def spec_sha256(spec_dict_or_path):
    """Compute deterministic SHA-256 digest of a topic filter spec."""
    if isinstance(spec_dict_or_path, (str, Path)):
        data = json.loads(Path(spec_dict_or_path).read_text(encoding="utf-8"))
    else:
        data = spec_dict_or_path
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_model(path):
    """Load TinyGPT model state dict into evaluation mode."""
    recipe_file = Path(path).parent / "recipe.json"
    if recipe_file.is_file():
        try:
            recipe = json.loads(recipe_file.read_text(encoding="utf-8")).get(
                "model", ovl.RECIPE["model"]
            )
        except Exception:
            recipe = ovl.RECIPE["model"]
    else:
        recipe = ovl.RECIPE["model"]
    model = ovl.TinyGPT(**recipe).to(ovl.DEV)
    model.load_state_dict(
        torch.load(path, map_location=ovl.DEV, weights_only=True)
    )
    model.lm_head.weight = model.transformer["wte"].weight
    return model.eval()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--data", required=True, help="prepared Wikipedia data directory"
    )
    p.add_argument(
        "--original",
        required=True,
        help="original run directory with base-model.pt",
    )
    p.add_argument(
        "--fork",
        required=True,
        help="fork directory with base-model.pt and chain.jsonl",
    )
    p.add_argument(
        "--entity",
        default="3I/ATLAS",
        help="entity to evaluate for near-duplicate leakage",
    )
    p.add_argument(
        "--alias",
        nargs="*",
        default=[],
        help="explicit entity aliases to search for",
    )
    p.add_argument(
        "--control-entities",
        nargs="*",
        default=["1I/'Oumuamua", "2I/Borisov"],
        help="rarity-matched astronomy control entities",
    )
    p.add_argument(
        "--sample-check",
        help="optional path to hand-checked validation sample JSON file",
    )
    p.add_argument(
        "--out-filter",
        help="optional output path to save verifiable topic_filter.json",
    )
    a = p.parse_args()

    from tokenizers import Tokenizer

    tok_path = Path(a.data) / "tokenizer" / "tokenizer.json"
    if not tok_path.is_file():
        sys.exit(f"tokenizer not found at {tok_path}")
    tok_bytes = tok_path.read_bytes()
    tokenizer_sha256 = hashlib.sha256(tok_bytes).hexdigest()
    tokenizer = Tokenizer.from_file(str(tok_path))

    # Read fork chain to identify primary document exclusion
    fork_chain = ovl.read_chain(a.fork)
    fork_entry = next((e for e in fork_chain if e["kind"] == "fork"), None)
    if not fork_entry:
        sys.exit("no fork entry found in fork chain.jsonl")

    excluded_pages = set()
    for doc in fork_entry.get("fork", {}).get("documents", []):
        if isinstance(doc, list) and len(doc) >= 2:
            excluded_pages.add(str(doc[1]))
        elif isinstance(doc, str):
            excluded_pages.add(doc)

    # 1. Scan entity mentions
    patterns = build_entity_patterns(
        tokenizer, a.entity, extra_aliases=a.alias
    )
    raw_windows = scan_entity_mentions(
        a.data, patterns, excluded_pages=excluded_pages
    )
    windows = deduplicate_windows(raw_windows)

    # 2. Scan rarity-matched control entities
    control_windows = []
    for c_entity in a.control_entities:
        c_patterns = build_entity_patterns(tokenizer, c_entity)
        c_raw = scan_entity_mentions(
            a.data, c_patterns, excluded_pages=excluded_pages
        )
        control_windows.extend(deduplicate_windows(c_raw))

    # 3. Load models
    model_orig = load_model(Path(a.original) / "base-model.pt")
    model_fork = load_model(Path(a.fork) / "base-model.pt")
    tok_stream, mask_stream = stream_arrays(Path(a.data) / "wikipedia")

    # 4. Evaluate target passage losses
    target_results = []
    for w in windows:
        res = eval_passage_loss_and_tokens(
            model_orig, model_fork, tok_stream, mask_stream, w, tokenizer
        )
        if res:
            target_results.append({**w, **res})

    # 5. Evaluate control passage losses
    control_results = []
    for w in control_windows:
        res = eval_passage_loss_and_tokens(
            model_orig, model_fork, tok_stream, mask_stream, w
        )
        if res:
            control_results.append({**w, **res})

    # 6. Statistical significance
    target_deltas = [r["mention_delta"] for r in target_results]
    control_deltas = [r["mention_delta"] for r in control_results]
    stats = compute_bootstrap_significance(target_deltas, control_deltas)

    # 7. Generate filter specification
    filter_spec = make_topic_filter_spec(
        a.entity,
        patterns,
        windows,
        stream_targets=tok_stream.shape[0],
        tokenizer_sha256=tokenizer_sha256,
        aliases=a.alias,
        sample_check_path=a.sample_check,
    )
    filter_hash = spec_sha256(filter_spec)

    if a.out_filter:
        Path(a.out_filter).write_text(
            json.dumps(filter_spec, indent=2, sort_keys=True), encoding="utf-8"
        )

    # Print Report
    report = {
        "entity": a.entity,
        "aliases": a.alias,
        "primary_documents_excluded": sorted(list(excluded_pages)),
        "near_duplicate_mentions_found": len(target_results),
        "articles_with_mentions": sorted(
            list({w["page_id"] for w in target_results})
        ),
        "control_mentions_found": len(control_results),
        "statistics": stats,
        "topic_filter_ranges": filter_spec["excluded_ranges_count"],
        "topic_filter_spec_sha256": filter_hash,
        "sample_attribution_map": target_results[0]["token_attributions"]
        if target_results
        else [],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
