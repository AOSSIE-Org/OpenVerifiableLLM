"""Synthetic rental authorization for tests; no real account or owner record."""
import hashlib
from pathlib import Path
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def pytest_configure(config):
    import rental_quote
    directory=tempfile.TemporaryDirectory(prefix='ovllm-synthetic-test-policy-')
    config._ovllm_authorization_fixture=directory
    path=Path(directory.name)/'authorization.json'
    data=b'{"schema":"ovl.synthetic-test-authorization.v1","purpose":"offline provider doubles only; never authorizes paid work"}\n'
    path.write_bytes(data)
    rental_quote.AUTHORIZATION=path
    rental_quote.AUTHORIZATION_SHA256=hashlib.sha256(data).hexdigest()


def pytest_unconfigure(config):
    directory=getattr(config,'_ovllm_authorization_fixture',None)
    if directory is not None:directory.cleanup()
