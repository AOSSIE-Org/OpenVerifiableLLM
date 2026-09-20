"""Actual hash gates on synthetic local authorization; no paid resources."""
import hashlib
from pathlib import Path
import pytest
import rental_quote as quote
from test_rental_controller import intent
from ovl_pipeline.canonical import EvidenceError


def test_fixture_authorization_is_hashed_and_private():
    r=intent()
    assert quote.AUTHORIZATION_SHA256==hashlib.sha256(quote.AUTHORIZATION.read_bytes()).hexdigest()
    assert not quote.AUTHORIZATION.is_relative_to(Path(__file__).resolve().parents[1])
    assert quote.validate_quote(r['quote'],r['payload'],r['watchdog_intent']['plan'])==r['watchdog_intent']['plan']['input']['hourly_upper_usd']


@pytest.mark.parametrize('damage',['missing','altered','symlink','wrong-plan-root'])
def test_private_authorization_failures_remain_strict(tmp_path,monkeypatch,damage):
    r=intent();path=tmp_path/'authorization.json';path.write_bytes(quote.AUTHORIZATION.read_bytes());monkeypatch.setattr(quote,'AUTHORIZATION',path)
    if damage=='missing':path.unlink()
    elif damage=='altered':path.write_bytes(b'changed')
    elif damage=='symlink':
        copy=tmp_path/'copy.json';path.rename(copy);path.symlink_to(copy)
    else:r['watchdog_intent']['plan']['input']['authorization_sha256']='0'*64
    with pytest.raises(EvidenceError):quote.validate_quote(r['quote'],r['payload'],r['watchdog_intent']['plan'])
