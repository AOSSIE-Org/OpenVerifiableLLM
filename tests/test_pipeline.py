"""Trust/coverage/replay regression tests for the new pipeline, not production evidence."""
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import shutil
import struct
import sys

import pytest
import torch
from nacl.signing import SigningKey

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ovl_pipeline.canonical import EvidenceError, Merkle, canonical, confined, digest, inventory, parse_json, verify_inventory, write_json
from ovl_pipeline.data import BOS, EOS, OFFSET, batches, check_coverage, extract_wikipedia, prepare_stream, rows, text_ids, train_tokenizer, validate_stream
from ovl_pipeline.fixture import prepare, recipe, run_fixture, verify_fixture
from ovl_pipeline.state import capture, read_state, restore, save_state, state_root, tensor_digest
from ovl_pipeline.training import TrustPolicy, full_replay, initialize, make_registration, signed, train, update, validate_chain
from tokenizers import Tokenizer

SOURCE = Path(__file__).parent / "fixtures/pipeline"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp("prepare") / "data"
    manifest = prepare(SOURCE, out)
    return out, manifest


@pytest.fixture(scope="module")
def registered(prepared):
    out, manifest = prepared
    key = SigningKey.generate()
    registration = make_registration(recipe(manifest["tokenizer"]["vocab_size"]), manifest["streams"], key,
                                     digest(inventory(SOURCE, ["wiki.xml", "conversations.json"])), digest(manifest))
    policy = TrustPolicy(digest(registration), bytes(key.verify_key).hex(), "local-synthetic-fixture")
    return registration, policy, {p: out / p for p in manifest["streams"]}, key


def test_canonical_independent_vectors():
    assert canonical({"z": "é", "a": [True, None, -9]}) == b'{"a":[true,null,-9],"z":"\xc3\xa9"}'
    # RFC 8785 sorts UTF-16 code units (astral character before U+E000).
    assert canonical({"\ue000": 1, "😀": 2}).decode() == '{"😀":2,"\ue000":1}'
    for bad in [b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1.0}', b'{"x":9007199254740992}', b'{"x":"\\ud800"}']:
        with pytest.raises(EvidenceError):
            parse_json(bad)
    with pytest.raises(EvidenceError):
        parse_json(b'{ "a":1}', canonical_required=True)


def test_merkle_recursive_independent_vector():
    def H(data):
        return hashlib.sha256(data).digest()
    def tree(items):
        if not items:
            return H(b"")
        if len(items) == 1:
            return H(b"\x00" + items[0])
        split = 1 << ((len(items) - 1).bit_length() - 1)
        return H(b"\x01" + tree(items[:split]) + tree(items[split:]))
    for n in range(18):
        leaves = [str(i).encode() for i in range(n)]
        accumulator = Merkle()
        for leaf in leaves:
            accumulator.add(leaf)
        assert accumulator.root() == H(b"ovl.merkle.v1\x00" + struct.pack(">Q", n) + tree(leaves)).hex()


def test_tensor_hash_binds_names_dtype_shape_bits():
    t = torch.tensor([0., 1.], dtype=torch.float32)
    root = tensor_digest({"a": t})
    alternatives = [{"b": t}, {"a": t.double()}, {"a": t.reshape(1, 2)}, {"a": torch.tensor([-0., 1.])}]
    assert all(tensor_digest(x) != root for x in alternatives)
    assert tensor_digest({"a": torch.tensor([[0., 2.], [1., 3.]]).T}) == tensor_digest({"a": torch.tensor([[0., 1.], [2., 3.]])})


def test_inventory_rejects_paths_and_symlinks(tmp_path):
    (tmp_path / "good").write_bytes(b"safe")
    entries = inventory(tmp_path, ["good"])
    verify_inventory(tmp_path, entries)
    for name in ["../outside", "/etc/passwd", "a/../good", "a//b", "./good", "a\\b", ""]:
        with pytest.raises(EvidenceError):
            confined(tmp_path, name)
    (tmp_path / "link").symlink_to(tmp_path / "good")
    with pytest.raises(EvidenceError):
        confined(tmp_path, "link")
    with pytest.raises(EvidenceError):
        verify_inventory(tmp_path, entries * 2)
    (tmp_path / "good").write_bytes(b"evil")
    with pytest.raises(EvidenceError):
        verify_inventory(tmp_path, entries)


def test_reconstruction_and_tokenizer_marker_roundtrip(prepared, tmp_path):
    out, manifest = prepared
    assert prepare(SOURCE, tmp_path / "again") == manifest
    assert manifest["corpus"]["record_count"] == 7
    assert manifest["corpus"]["counts"] == {"included": 3, "redirect": 1, "non_main_namespace": 1, "empty_revision_text": 1, "non_wikitext_model": 1}
    first=next(rows(out / "corpus/articles.jsonl"))
    assert first["attribution_url"]=="https://en.wikipedia.org/w/index.php?curid=1"
    assert first["revision_url"]=="https://en.wikipedia.org/w/index.php?oldid=101"
    assert first["history_url"]=="https://en.wikipedia.org/w/index.php?curid=1&action=history"
    tok = Tokenizer.from_file(str(out / "tokenizer/tokenizer.json"))
    for text in ["<|bos|><|assistant|>", "café 東京 😀\n", "\x00end"]:
        ids = text_ids(tok, text)
        assert all(i >= OFFSET for i in ids)
        assert tok.decode([i - OFFSET for i in ids]) == text


def test_malformed_duplicate_and_entities_abort(tmp_path):
    good = (SOURCE / "wiki.xml").read_text()
    cases = [good[:-30], good.replace("<id>2</id>", "<id>1</id>"),
             good.replace("<id>1</id>", "<id>1&amp;other=value</id>"),
             '<!DOCTYPE mediawiki [<!ENTITY x SYSTEM "file:///etc/passwd">]><mediawiki>&x;</mediawiki>']
    for i, text in enumerate(cases):
        raw = tmp_path / f"raw{i}.xml"
        raw.write_text(text)
        from xml.etree.ElementTree import ParseError
        from defusedxml.common import DefusedXmlException
        with pytest.raises((EvidenceError, ParseError, DefusedXmlException)):
            extract_wikipedia([raw], tmp_path / f"out{i}")


def test_actual_batch_targets_match_independent_documents(prepared):
    out, manifest = prepared
    import numpy as np
    for phase, stream in manifest["streams"].items():
        directory = out / phase
        assert validate_stream(directory, stream) == stream["targets"]
        tokens = np.memmap(directory / "tokens.u16", dtype="<u2", mode="r")
        mask = np.memmap(directory / "mask.u8", dtype="u1", mode="r")
        expected = tokens[mask.astype(bool)].tolist()
        actual, cursor = [], 0
        all_batches = list(batches(directory, 17, 4))
        for batch in all_batches:
            cursor = check_coverage(batch, cursor, stream["targets"])
            actual.extend(batch["targets"][batch["mask"]].tolist())
        assert actual == expected and cursor == len(expected)
        assert sum(int(b["mask"].sum()) for b in all_batches) == stream["targets"]
        # First article target is predicted from BOS, not silently omitted.
        if phase == "wikipedia":
            assert all_batches[0]["inputs"][0, 0] == BOS
        bad = copy.deepcopy(all_batches[0])
        bad["target_ids"][bad["mask"]] += 1
        with pytest.raises(EvidenceError):
            check_coverage(bad, 0, stream["targets"])


def test_optimizer_rng_control_and_alias_state_are_bound(registered):
    reg, _, dirs, _ = registered
    model, opt, control = initialize(reg["recipe"])
    batch = next(batches(dirs["wikipedia"], reg["recipe"]["context"], reg["recipe"]["batch_size"]))
    control = update(model, opt, batch, control, reg["streams"]["wikipedia"]["targets"])
    md, tensors = capture(model, opt, control)
    root = state_root(md, tensors)
    model_root = tensor_digest(dict(model.state_dict()))
    first = next(iter(opt.state.values()))
    first["exp_avg"].flatten()[0] += 1
    assert state_root(*capture(model, opt, control)) != root
    assert tensor_digest(dict(model.state_dict())) == model_root
    restore(model, opt, md, tensors)
    assert state_root(*capture(model, opt, control)) == root
    torch.rand(1)
    assert state_root(*capture(model, opt, control)) != root
    restore(model, opt, md, tensors)
    assert state_root(*capture(model, opt, {**control, "cursor": control["cursor"] + 1})) != root
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
    with pytest.raises(EvidenceError, match="alias"):
        restore(model, opt, md, tensors)


def test_checkpoint_safe_load_rejects_byte_flip_and_incomplete(registered, tmp_path):
    reg, _, _, _ = registered
    model, opt, ctrl = initialize(reg["recipe"])
    ck = save_state(tmp_path / "ck", model, opt, ctrl)
    read_state(tmp_path / "ck", ck)
    p = tmp_path / "ck/state.safetensors"
    data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
    with pytest.raises(EvidenceError):
        read_state(tmp_path / "ck", ck)
    with pytest.raises(EvidenceError):
        read_state(tmp_path / "absent", ck)


def test_continuous_replay_and_resume(registered, tmp_path):
    reg, policy, dirs, key = registered
    original = train(reg, policy, dirs, tmp_path / "original", key)
    report = full_replay(reg, policy, dirs, tmp_path / "original")
    train(reg, policy, dirs, tmp_path / "resumed", key, stop_after=3)
    resumed = train(reg, policy, dirs, tmp_path / "resumed", key, resume=True)
    assert original == resumed
    assert full_replay(reg, policy, dirs, tmp_path / "resumed") == report
    assert report["locally_recomputed"] and not report["independent_third_party"]
    # A valid signature on altered optimizer/control must still fail continuous replay.
    chain_path = tmp_path / "original/chain.json"
    chain = parse_json(chain_path.read_bytes())
    chain["boundaries"][1]["body"]["control"]["transcript"] = "0" * 64
    chain["boundaries"][1] = signed(chain["boundaries"][1]["body"], key)
    for i in range(2, len(chain["boundaries"])):
        chain["boundaries"][i]["body"]["previous"] = digest(chain["boundaries"][i-1]["body"])
        chain["boundaries"][i] = signed(chain["boundaries"][i]["body"], key)
    write_json(chain_path, chain)
    with pytest.raises(EvidenceError):
        full_replay(reg, policy, dirs, tmp_path / "original")


def test_chain_omission_order_and_untrusted_identity(registered, tmp_path):
    reg, policy, dirs, key = registered
    chain = train(reg, policy, dirs, tmp_path / "run", key)
    for modified in [[], chain[:-1], chain[1:], chain[::-1], chain[:1] + chain[2:]]:
        with pytest.raises(EvidenceError):
            validate_chain(reg, policy, modified, complete=True)
    with pytest.raises(EvidenceError):
        replace(policy, run_public_key_hex=bytes(SigningKey.generate().verify_key).hex()).validate(reg)
    with pytest.raises(EvidenceError):
        replace(policy, scope="production").validate(reg)
    changed = copy.deepcopy(reg); changed["recipe"]["seed"] += 1
    with pytest.raises(EvidenceError):
        policy.validate(changed)


def test_complete_fixture_and_clean_copy_tampering(tmp_path):
    out = tmp_path / "bundle"
    policy_path = tmp_path / "policy.json"
    result = run_fixture(SOURCE, out, policy_path)
    assert result["result"] == "PASS"
    assert result["production_acceptance"] == "NOT_RUN"
    assert result["public_anchoring"] == "NOT_RUN"
    policy = TrustPolicy(**parse_json(policy_path.read_bytes()))
    target = out / "raw/wiki.xml"
    target.write_bytes(target.read_bytes().replace(b"Alpha", b"Omega"))
    with pytest.raises(EvidenceError):
        verify_fixture(out, policy)


def test_resume_rejects_signed_path_escape_before_loading(registered, tmp_path):
    reg, policy, dirs, key = registered
    run = tmp_path / "run"
    train(reg, policy, dirs, run, key, stop_after=3)
    log = parse_json((run / "chain.json").read_bytes())
    last = log["boundaries"][-1]["body"]
    shutil.copytree(run / last["checkpoint_path"], tmp_path / "outside")
    last["checkpoint_path"] = "../outside"
    log["boundaries"][-1] = signed(last, key)
    write_json(run / "chain.json", log)
    with pytest.raises(EvidenceError, match="path|escaping"):
        train(reg, policy, dirs, run, key, resume=True)


def test_restore_rejects_inconsistent_tied_payload(registered):
    from ovl_pipeline.state import pack, unpack
    reg, _, _, _ = registered
    model, opt, control = initialize(reg["recipe"])
    md, tensors = capture(model, opt, control)
    decoded = unpack(md["tree"], tensors)
    decoded["model"]["lm_head.weight"][0, 0] += 1
    changed = {}
    metadata = {"schema": "ovl.state.v1", "tree": pack(decoded, changed), "tensor_root": tensor_digest(changed)}
    with pytest.raises(EvidenceError, match="alias|state|tied"):
        restore(model, opt, metadata, changed)


def test_recipe_unknown_init_and_missing_required_fields_fail(prepared):
    _, manifest = prepared
    invalid = recipe(manifest["tokenizer"]["vocab_size"])
    invalid["init"] = "pretrained-from-private-cache"
    with pytest.raises(EvidenceError, match="recipe|init"):
        initialize(invalid)


def test_untrusted_entrypoint_symlink_rejected_before_read(tmp_path):
    from unittest.mock import patch
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    bundle = tmp_path / "bundle";bundle.mkdir()
    (bundle / "registration.json").symlink_to(outside)
    policy = TrustPolicy("0" * 64, "0" * 64, "local-synthetic-fixture")
    # Reject the path itself, before parsing any attacker-selected outside bytes.
    with pytest.raises(EvidenceError, match="symlink"):
        verify_fixture(bundle, policy)


def test_checkpoint_completion_marker_is_required(registered, tmp_path):
    reg, _, _, _ = registered
    model, opt, ctrl = initialize(reg["recipe"])
    ck = save_state(tmp_path / "ck", model, opt, ctrl)
    (tmp_path / "ck/checkpoint.json").unlink()
    with pytest.raises(EvidenceError, match="completion"):
        read_state(tmp_path / "ck", ck)


def test_unknown_critical_stream_fields_and_invalid_masks(prepared, tmp_path):
    out, manifest = prepared
    stream = copy.deepcopy(manifest["streams"]["wikipedia"])
    stream["secret-extra-data"] = True
    with pytest.raises(EvidenceError, match="fields"):
        validate_stream(out / "wikipedia", stream)
    copied = tmp_path / "wiki"; shutil.copytree(out / "wikipedia", copied)
    mask = copied / "mask.u8"; data = bytearray(mask.read_bytes()); data[0] = 2; mask.write_bytes(data)
    stream = copy.deepcopy(manifest["streams"]["wikipedia"])
    stream["files"] = inventory(copied, ["tokens.u16", "mask.u8", "documents.jsonl"])
    with pytest.raises(EvidenceError, match="mask"):
        validate_stream(copied, stream)


def test_json_read_is_bounded(tmp_path):
    from ovl_pipeline.canonical import read_json
    p = tmp_path / "large.json";p.write_bytes(b" " * 1025)
    with pytest.raises(EvidenceError, match="size"):
        read_json(p, limit=1024)


@pytest.fixture(scope="module")
def signer_known_bundle(tmp_path_factory):
    from unittest.mock import patch
    tmp = tmp_path_factory.mktemp("known-signer")
    key = SigningKey.generate()
    with patch("ovl_pipeline.fixture.SigningKey.generate", return_value=key):
        run_fixture(SOURCE, tmp / "bundle", tmp / "policy.json")
    return tmp / "bundle", TrustPolicy(**parse_json((tmp / "policy.json").read_bytes())), key


def resign_release(bundle, key):
    body = parse_json((bundle / "release.json").read_bytes())["body"]
    names = [p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file() and p.name != "release.json"]
    body["files"] = inventory(bundle, names)
    write_json(bundle / "release.json", signed(body, key))


def test_resigned_published_tokenizer_must_match_registration(signer_known_bundle, tmp_path):
    bundle, policy, key = signer_known_bundle
    out = tmp_path / "copy";shutil.copytree(bundle, out)
    (out / "prepared/tokenizer/tokenizer.json").write_bytes(b"changed published tokenizer")
    resign_release(out, key)
    with pytest.raises(EvidenceError, match="prepar|reconstruct|tokenizer"):
        verify_fixture(out, policy)


def test_resigned_export_architecture_must_match_registration(signer_known_bundle, tmp_path):
    from ovl_pipeline.fixture import infer
    bundle, policy, key = signer_known_bundle
    out = tmp_path / "copy";shutil.copytree(bundle, out)
    path = out / "base/config.json"
    config = parse_json(path.read_bytes())
    config["recipe"]["model"]["num_heads"] = 1  # Same tensor shapes and bits; different computation.
    write_json(path, config)
    write_json(out / "base/inference.json", infer(out / "base", out / "prepared/tokenizer/tokenizer.json"))
    resign_release(out, key)
    with pytest.raises(EvidenceError, match="recipe|config"):
        verify_fixture(out, policy)


def test_pickle_payload_is_never_executed(registered, tmp_path):
    import pickle
    from safetensors import SafetensorError
    reg, _, _, _ = registered
    model, opt, control = initialize(reg["recipe"])
    directory = tmp_path / "checkpoint"
    ck = save_state(directory, model, opt, control)
    sentinel = tmp_path / "pickle-was-executed"
    class Payload:
        def __reduce__(self):
            return (eval, (f"__import__('pathlib').Path({str(sentinel)!r}).write_text('bad')",))
    (directory / "state.safetensors").write_bytes(pickle.dumps(Payload()))
    ck["files"] = inventory(directory, ["state.json", "state.safetensors"])
    write_json(directory / "checkpoint.json", ck)
    with pytest.raises((EvidenceError, SafetensorError)):
        read_state(directory, ck)
    assert not sentinel.exists()


def test_symlinked_evidence_subtrees_are_rejected(signer_known_bundle, tmp_path):
    bundle, policy, key = signer_known_bundle
    out=tmp_path/'copy';shutil.copytree(bundle,out)
    for name in ('raw','training'):
        outside=tmp_path/('external-'+name);shutil.move(str(out/name),outside)
        (out/name).symlink_to(outside,target_is_directory=True)
    resign_release(out,key)
    with pytest.raises(EvidenceError, match='symlink'):
        verify_fixture(out,policy)


@pytest.mark.parametrize("mutation", ["dtype", "metadata"])
def test_dtype_cast_or_extra_export_metadata_rejected(signer_known_bundle, tmp_path, mutation):
    from safetensors.torch import load_file, save_file
    bundle,policy,key=signer_known_bundle
    out=tmp_path/'copy';shutil.copytree(bundle,out)
    p=out/'base/model.safetensors'
    state=load_file(str(p))
    if mutation == "dtype":
        save_file({k:v.double() for k,v in state.items()},str(p))
    else:
        save_file(state,str(p),metadata={"ignored":"unbound"})
    resign_release(out,key)
    with pytest.raises(EvidenceError, match='dtype|export|metadata'):
        verify_fixture(out,policy)


def test_extra_signed_artifacts_are_not_silently_accepted(signer_known_bundle,tmp_path):
    bundle,policy,key=signer_known_bundle
    out=tmp_path/'copy';shutil.copytree(bundle,out)
    (out/'base/another-weights.bin').write_bytes(b'unrelated artifact')
    resign_release(out,key)
    with pytest.raises(EvidenceError,match='unlisted|unexpected|file set'):
        verify_fixture(out,policy)


def test_crash_after_checkpoint_before_chain_publish_resumes(registered,tmp_path):
    from unittest.mock import patch
    import ovl_pipeline.training as training
    reg,policy,dirs,key=registered
    baseline=train(reg,policy,dirs,tmp_path/'baseline',key)
    real=training.write_json
    def crash(path,value):
        if path.name=='chain.json' and len(value['boundaries'])==3:
            raise OSError('simulated crash after durable checkpoint before chain commit')
        return real(path,value)
    with patch('ovl_pipeline.training.write_json',side_effect=crash):
        with pytest.raises(OSError,match='simulated crash'):
            train(reg,policy,dirs,tmp_path/'crashed',key)
    resumed=train(reg,policy,dirs,tmp_path/'crashed',key,resume=True)
    assert resumed==baseline
    assert full_replay(reg,policy,dirs,tmp_path/'crashed')['result']=='PASS'


@pytest.mark.parametrize('failure_call',[1,3])
def test_partial_checkpoint_preserved_and_recomputed(registered,tmp_path,failure_call):
    from unittest.mock import patch
    import ovl_pipeline.state as state_module
    reg,policy,dirs,key=registered
    baseline=train(reg,policy,dirs,tmp_path/'baseline',key)
    original=state_module.save_file;calls=0
    def crash(tensors,path):
        nonlocal calls
        calls+=1
        if calls==failure_call:
            Path(path).write_bytes(b'incomplete-checkpoint-evidence')
            raise OSError('simulated mid-write interruption')
        return original(tensors,path)
    with patch('ovl_pipeline.state.save_file',side_effect=crash):
        with pytest.raises(OSError,match='simulated'):
            train(reg,policy,dirs,tmp_path/'interrupted',key)
    assert train(reg,policy,dirs,tmp_path/'interrupted',key,resume=True)==baseline
    preserved=list((tmp_path/'interrupted-recovery').rglob('state.safetensors'))
    assert len(preserved)==1 and preserved[0].read_bytes()==b'incomplete-checkpoint-evidence'


def test_per_model_export_is_self_contained(signer_known_bundle,tmp_path):
    from ovl_pipeline.fixture import infer
    bundle,policy,key=signer_known_bundle
    for phase in ('base','chat'):
        out=tmp_path/phase;shutil.copytree(bundle/phase,out)
        assert infer(out)==parse_json((bundle/phase/'inference.json').read_bytes())


def test_chain_validation_is_independent_of_working_directory(registered,tmp_path,monkeypatch):
    reg,policy,dirs,key=registered
    chain=train(reg,policy,dirs,tmp_path/'training',key)
    cwd=tmp_path/'cwd';cwd.mkdir();(cwd/'boundary-00000').symlink_to(tmp_path/'training')
    monkeypatch.chdir(cwd)
    validate_chain(reg,policy,chain,complete=True)


def test_oversized_boundary_plan_fails_before_any_update(registered,tmp_path,monkeypatch):
    reg,policy,dirs,key=registered
    monkeypatch.setattr('ovl_pipeline.training.MAX_BOUNDARIES',2)
    with pytest.raises(EvidenceError,match='boundary schedule'):
        train(reg,policy,dirs,tmp_path/'must-not-start',key)
    assert not (tmp_path/'must-not-start').exists()
