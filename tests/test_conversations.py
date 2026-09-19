import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovl_pipeline.canonical import EvidenceError
from ovl_pipeline.conversations import select_conversations
from ovl_pipeline.data import rows


def message(mid, parent, role, tree="root", rank=None, **kw):
    return {"message_id": mid, "parent_id": parent, "text": "Text " + mid, "role": role,
            "lang": "en", "review_result": True, "deleted": False, "rank": rank,
            "message_tree_id": tree, **kw}


def sources():
    return {"train": [message("root", None, "prompter"),
                      message("missing-rank", "root", "assistant"),
                      message("ranked", "root", "assistant", rank=0),
                      message("next", "ranked", "prompter"),
                      message("final", "next", "assistant"),
                      message("bad", "absent", "assistant"),
                      message("non-en", "root", "assistant", lang="fr"),
                      message("removed", "root", "assistant", deleted=True)],
            "validation": [message("val", None, "prompter", tree="val"),
                           message("val-answer", "val", "assistant", tree="val")]}


def test_preferred_branch_missing_rank_accounting_and_regeneration(tmp_path):
    data = sources()
    first = select_conversations(data, tmp_path / "one")
    assert first == select_conversations(data, tmp_path / "two")
    selected = list(rows(tmp_path / "one/train.jsonl"))
    assert selected[0]["identity"][2] == ["root", "ranked", "next", "final"]
    assert first["conversations"] == {"train": 1, "validation": 1}
    assert sum(sum(c.values()) for c in first["counts"].values()) == 10
    assert first["counts"]["train"]["missing_parent"] == 1
    assert first["counts"]["train"]["non_english"] == 1


def test_leakage_duplicate_and_cycle_rejected(tmp_path):
    data = sources();data["validation"][0]["message_tree_id"] = "root"
    with pytest.raises(EvidenceError, match="leakage"):
        select_conversations(data, tmp_path / "leak")
    data = sources();data["validation"][0]["message_id"] = "root"
    with pytest.raises(EvidenceError, match="duplicate"):
        select_conversations(data, tmp_path / "duplicate")
    data=sources();data["train"] += [message("cycle1", "cycle2", "prompter"), message("cycle2", "cycle1", "assistant")]
    with pytest.raises(EvidenceError, match="cycle"):
        select_conversations(data, tmp_path / "cycle")


def test_rejected_ancestor_and_no_validation_training(tmp_path):
    data=sources();data["train"][2]["review_result"]=False
    manifest=select_conversations(data,tmp_path / "out")
    selected=list(rows(tmp_path / "out/train.jsonl"))
    assert selected[0]["identity"][2] == ["root", "missing-rank"]
    assert manifest["counts"]["train"]["unusable_ancestor"] == 2
    assert all(row["identity"][0] == "train" for row in selected)
