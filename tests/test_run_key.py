import os

from nacl.signing import VerifyKey
import pytest

from ovl_pipeline.canonical import EvidenceError, canonical, write_json
from ovl_pipeline.run_key import create, descriptor, load


def test_persistent_key_adoption_signs_under_external_pin(tmp_path):
    path = tmp_path / "private";public = create(path, "test-run")
    key = load(path, run_id="test-run", expected_public_key=public["public_key"])
    message = canonical({"boundary": 3})
    VerifyKey(bytes.fromhex(public["public_key"])).verify(message, key.sign(message).signature)
    assert path.stat().st_mode & 0o777 == 0o700
    assert (path / "seed.key").stat().st_mode & 0o777 == 0o600
    before = (path / "seed.key").read_bytes()
    with pytest.raises(FileExistsError):create(path, "another-run")
    assert (path / "seed.key").read_bytes() == before


def test_standard_ed25519_vector_adopted_without_serialized_code(tmp_path):
    # RFC8032 section7.1 TEST1, publicly known test seed (not a production key).
    path = tmp_path / "private";path.mkdir(mode=0o700)
    public = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    (path / "seed.key").write_bytes(seed);(path / "seed.key").chmod(0o600)
    write_json(path / "public.json", descriptor("vector", public))
    key = load(path, run_id="vector", expected_public_key=public)
    assert key.sign(b"").signature.hex() == ("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb"
                                           "8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")


@pytest.mark.parametrize("mutation", ["directory-mode", "seed-mode", "symlink", "hardlink", "fifo", "truncated", "different-seed", "missing-public", "wrong-run"])
def test_unsafe_or_mismatched_key_fails_closed(tmp_path, mutation):
    path = tmp_path / "private";public = create(path, "test-run");seed = path / "seed.key"
    if mutation == "directory-mode":path.chmod(0o755)
    elif mutation == "seed-mode":seed.chmod(0o644)
    elif mutation == "symlink":
        seed.rename(path / "other");seed.symlink_to(path / "other")
    elif mutation == "hardlink":os.link(seed, path / "alias")
    elif mutation == "fifo":seed.unlink();os.mkfifo(seed, mode=0o600)
    elif mutation == "truncated":seed.write_bytes(b"a")
    elif mutation == "different-seed":seed.write_bytes(b"a" * 32)
    elif mutation == "missing-public":(path / "public.json").unlink()
    with pytest.raises((EvidenceError, OSError)):
        load(path, run_id="wrong" if mutation == "wrong-run" else "test-run", expected_public_key=public["public_key"])


def test_bundle_supplied_identity_cannot_replace_external_pin(tmp_path):
    path = tmp_path / "private";first = create(path, "run")
    other = create(tmp_path / "other", "run")
    write_json(path / "public.json", other)
    with pytest.raises(EvidenceError, match="external identity"):
        load(path, run_id="run", expected_public_key=first["public_key"])


def test_partial_creation_and_symlink_parent_are_never_replaced(tmp_path):
    path = tmp_path / "partial";path.mkdir(mode=0o700)
    with pytest.raises(FileExistsError):create(path, "run")
    alias = tmp_path / "alias";alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(EvidenceError, match="symlink"):create(alias / "new", "run")
    assert not (tmp_path / "new").exists()
