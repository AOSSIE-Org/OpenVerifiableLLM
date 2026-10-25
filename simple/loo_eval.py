"""Leave-one-out comparison of an original run and a fork that removed Wikipedia documents.

Loss on the removed documents' own text (every window, exactly as training would see it) versus
control documents from the same segment that both models trained on, the weight distance, and
greedy completions for a few prompts. Prints one JSON object.
    python simple/loo_eval.py --data PREP --original RUN --fork FORK [--prompt "3I/ATLAS is"]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ovl  # noqa: E402
from ovl_pipeline.data import BOS, EOS, USER, ASSISTANT, OFFSET, stream_arrays  # noqa: E402


def load(path):
    model = ovl.TinyGPT(**ovl.RECIPE["model"]).to(ovl.DEV)
    model.load_state_dict(torch.load(path, map_location=ovl.DEV, weights_only=True))
    model.lm_head.weight = model.transformer["wte"].weight
    return model.eval()


@torch.no_grad()
def doc_loss(model, tok, mask, d):
    """Mean next-token loss over one document's targets, using the training windows."""
    ctx, total, count = ovl.RECIPE["context"], 0.0, 0
    for start in range(0, d["tokens"], ctx):
        n = min(ctx, d["tokens"] - start)
        a = d["offset"] + start
        y = torch.from_numpy(tok[a:a + n].astype("int64")).to(ovl.DEV)
        prev = BOS if start == 0 else int(tok[a - 1])
        x = torch.cat((torch.tensor([prev], device=ovl.DEV), y[:-1]))[None]
        m = torch.from_numpy(mask[a:a + n] != 0).to(ovl.DEV)
        with torch.autocast(ovl.DEV, dtype=torch.bfloat16, cache_enabled=False):
            losses = F.cross_entropy(model(x)[0].float(), y, reduction="none")
        total += losses[m].sum().item()
        count += int(m.sum())
    return total / max(1, count), count


@torch.no_grad()
def complete(model, tokenizer, prompt, chat, tokens=40):
    ids = [i + OFFSET for i in tokenizer.encode(prompt, add_special_tokens=False).ids]
    ids = ([USER] + ids + [EOS, ASSISTANT]) if chat else ([BOS] + ids)
    out = []
    for _ in range(tokens):
        x = torch.tensor([ids[-ovl.RECIPE["context"]:]], device=ovl.DEV)
        nxt = int(model(x)[0, -1].float().argmax())  # greedy: deterministic, comparable
        if nxt == EOS:
            break
        ids.append(nxt)
        out.append(nxt)
    return tokenizer.decode([i - OFFSET for i in out if i >= OFFSET])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--original", required=True); p.add_argument("--fork", required=True)
    p.add_argument("--controls", type=int, default=50)
    p.add_argument("--prompt", nargs="*", default=["3I/ATLAS is", "The interstellar comet 3I/ATLAS",
                                                   "Oumuamua was the first interstellar object"])
    a = p.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(a.data) / "tokenizer" / "tokenizer.json"))
    fork_chain = ovl.read_chain(a.fork)
    entry = next(e for e in fork_chain if e["kind"] == "fork")
    removed = {s for s, _ in entry["control"]["exclude"]["wikipedia"]}
    lo = entry["control"]["cursor"]
    hi = next(e for e in fork_chain if e["kind"] == "base")["control"]["cursor"]
    tok, mask = stream_arrays(Path(a.data) / "wikipedia")
    gone, pool = [], []
    with (Path(a.data) / "wikipedia" / "documents.jsonl").open("rb") as f:
        for line in f:
            d = json.loads(line)
            if d["target_start"] in removed:
                gone.append(d)
            elif lo <= d["target_start"] and d["target_start"] + d["targets"] <= hi and 3000 <= d["targets"] <= 15000:
                pool.append(d)
    controls = random.Random(1234).sample(pool, a.controls)
    result = {"removed_documents": [d["identity"] for d in gone], "control_documents": len(controls)}
    models = {"original_base": Path(a.original) / "base-model.pt", "fork_base": Path(a.fork) / "base-model.pt",
              "original_conversation": Path(a.original) / "conversation-model.pt",
              "fork_conversation": Path(a.fork) / "conversation-model.pt"}
    for name in ("original_base", "fork_base"):
        model = load(models[name])
        result[name] = {"removed_loss": [round(doc_loss(model, tok, mask, d)[0], 4) for d in gone],
                        "control_loss_mean": round(sum(doc_loss(model, tok, mask, d)[0] for d in controls) / len(controls), 4),
                        "completions": {q: complete(model, tokenizer, q, chat=False) for q in a.prompt}}
        del model
    for name in ("original_conversation", "fork_conversation"):
        if models[name].exists():
            model = load(models[name])
            result[name] = {"completions": {q: complete(model, tokenizer, "What is 3I/ATLAS?", chat=True) for q in ["chat"]}}
            del model
    ob, fb = (torch.load(models[n], map_location="cpu", weights_only=True) for n in ("original_base", "fork_base"))
    diff = sum(float((ob[k].float() - fb[k].float()).norm() ** 2) for k in ob) ** 0.5
    norm = sum(float(ob[k].float().norm() ** 2) for k in ob) ** 0.5
    result["base_weight_relative_distance"] = round(diff / norm, 6)
    for name in ("original_base", "fork_base"):
        result[name]["removed_minus_control"] = [round(x - result[name]["control_loss_mean"], 4) for x in result[name]["removed_loss"]]
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
