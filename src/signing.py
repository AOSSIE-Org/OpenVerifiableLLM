"""Ed25519 signing for checkpoint artifacts (Phase 2 security fix).

THE HOLE THIS CLOSES
--------------------
The original audit loaded checkpoints with ``torch.load(path, weights_only=False)``
and only afterwards compared a SHA-256 of the *deserialized* weights to an
embedded hash. ``weights_only=False`` runs a pickle reducer, so a malicious
checkpoint executes arbitrary code AT DESERIALIZATION TIME -- i.e. before the
integrity check ever runs. The hash check is security theatre against a tampered
file: by the time it fails, the payload has already executed.

THE FIX
-------
1. The prover signs the raw artifact bytes with an ed25519 *private* key and
   writes a detached ``<path>.sig``.
2. The auditor verifies that signature against the *public* key over the raw
   bytes BEFORE any deserialization. A tampered file fails verification and is
   rejected without ever being unpickled.

This also makes the project's "cryptographically signed" claim true: previously
there was only a SHA-256 (a checksum, not a signature -- anyone can recompute it).

Keys live in ``<repo>/keys``: the private key is git-ignored; the public key is
committed so anyone can verify. ``pip install pynacl`` provides the primitive.
"""

import os
from pathlib import Path

from nacl import signing as _nacl_signing
from nacl.exceptions import BadSignatureError

KEYS_DIR = Path(__file__).resolve().parents[1] / "keys"
PRIVATE_KEY_PATH = KEYS_DIR / "ovl_ed25519.key"   # secret, git-ignored
PUBLIC_KEY_PATH = KEYS_DIR / "ovl_ed25519.pub"    # public, committed
SIG_SUFFIX = ".sig"


class SignatureError(Exception):
    """Raised when an artifact's signature is missing or invalid."""


def generate_keypair(force=False):
    """Create the ed25519 keypair if absent. Returns (SigningKey, VerifyKey)."""
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    if PRIVATE_KEY_PATH.exists() and not force:
        return load_signing_key(), load_verify_key()
    sk = _nacl_signing.SigningKey.generate()
    PRIVATE_KEY_PATH.write_bytes(bytes(sk))
    try:
        os.chmod(PRIVATE_KEY_PATH, 0o600)
    except OSError:
        pass
    PUBLIC_KEY_PATH.write_bytes(bytes(sk.verify_key))
    return sk, sk.verify_key


def load_signing_key():
    """Load the private signing key, generating a keypair on first use."""
    if not PRIVATE_KEY_PATH.exists():
        return generate_keypair()[0]
    return _nacl_signing.SigningKey(PRIVATE_KEY_PATH.read_bytes())


def load_verify_key():
    """Load the public verify key. Raises FileNotFoundError if key does not exist."""
    if not PUBLIC_KEY_PATH.exists():
        raise FileNotFoundError(
            f"Public key not found at {PUBLIC_KEY_PATH}. Generate keys first using generate_keypair()."
        )
    return _nacl_signing.VerifyKey(PUBLIC_KEY_PATH.read_bytes())


def sig_path_for(path):
    return Path(str(path) + SIG_SUFFIX)


def sign_file(path, signing_key=None):
    """Sign a file's raw bytes; write a detached <path>.sig. Returns sig hex."""
    path = Path(path)
    signing_key = signing_key or load_signing_key()
    signature = signing_key.sign(path.read_bytes()).signature
    sig_path_for(path).write_bytes(signature)
    return signature.hex()


def verify_file(path, signature=None, verify_key=None):
    """Verify a file against its detached signature WITHOUT deserializing it.

    Returns True on success; raises SignatureError on a missing or bad signature.
    """
    path = Path(path)
    verify_key = verify_key or load_verify_key()
    if signature is None:
        sp = sig_path_for(path)
        if not sp.exists():
            raise SignatureError(f"no signature found for {path} (expected {sp})")
        signature = sp.read_bytes()
    try:
        verify_key.verify(path.read_bytes(), signature)
    except BadSignatureError as exc:
        raise SignatureError(f"signature verification FAILED for {path}") from exc
    return True


def verified_torch_load(path, *, map_location=None, expect_signature=True):
    """The secure replacement for ``torch.load(path, weights_only=False)``.

    Verifies the ed25519 signature over the raw file bytes FIRST. Only if that
    passes do we deserialize. A tampered checkpoint never reaches the unpickler.

    ``expect_signature=False`` is an explicit, logged escape hatch for legacy
    unsigned checkpoints; it warns loudly and still refuses ``weights_only=False``
    in favour of the safe tensor-only loader.
    """
    import torch  # lazy: keep crypto importable without torch
    from io import BytesIO

    path = Path(path)
    if expect_signature:
        # Read file once, verify signature, then deserialize from same bytes
        file_bytes = path.read_bytes()
        verify_key = load_verify_key()
        sp = sig_path_for(path)
        if not sp.exists():
            raise SignatureError(f"no signature found for {path} (expected {sp})")
        signature = sp.read_bytes()
        try:
            verify_key.verify(file_bytes, signature)
        except BadSignatureError as exc:
            raise SignatureError(f"signature verification FAILED for {path}") from exc
        return torch.load(BytesIO(file_bytes), map_location=map_location, weights_only=False)

    import warnings

    warnings.warn(
        f"Loading {path} WITHOUT signature verification; refusing weights_only=False. "
        "Sign your checkpoints with signing.sign_file to enable safe full loads."
    )
    return torch.load(path, map_location=map_location, weights_only=True)


def signed_torch_save(obj, path, signing_key=None):
    """``torch.save`` then sign: produces a checkpoint plus a detached signature."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
    sign_file(path, signing_key=signing_key)
    return path


if __name__ == "__main__":
    # Tiny self-check: sign a file, tamper it, prove verification rejects it
    # BEFORE deserialization.
    import tempfile

    sk, vk = generate_keypair()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "artifact.bin"
        p.write_bytes(b"trusted-bytes" * 100)
        sign_file(p, sk)
        print("clean verify:", verify_file(p, verify_key=vk))
        p.write_bytes(b"tampered-bytes" * 100)  # attacker edits the file
        try:
            verify_file(p, verify_key=vk)
            print("ERROR: tamper not detected")
        except SignatureError as e:
            print("tamper rejected (pre-deserialization):", e)
