"""Check the historical preparation contract without substituting training code.

Only reviewed, fixed source snapshots may execute. The original source signature
must already have been verified by the caller; this check supplies no signature,
reconstruction or training credit. The retained revision is a byte supplier, not
a replacement for the original signing revision. Python and the hash-installed
dependency/site environment remain trusted; isolated mode is not a sandbox or
an attestation of installed package bytes.
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .canonical import EvidenceError, canonical, digest, inventory, parse_json, sha256

# Reviewed public continuity after the history cutover. Neither packet contents
# nor a caller-provided policy may select another executable source snapshot.
PREPARATION_SNAPSHOTS = {
    'ec263b5c3914fd1a8bd25e97fa377b4f8416c38a9ef216d27c203252948b9129':
        '3f882ad9d57a0eeacb7c5db288af4242f8de5779',
}
PREPARATION_PATHS = tuple(sorted([
    'src/ovl_pipeline/' + name for name in (
        '__init__.py', 'acquisition.py', 'anchoring.py', 'canonical.py',
        'conversations.py', 'data.py', 'extraction_workers.py', 'preparation.py',
        'preparation_stages.py', 'schema.py', 'source_commitment.py')
] + ['requirements/preparation.in', 'requirements/preparation.lock',
     '.github/workflows/anchor-pipeline.yml']))

_CHECK = r'''
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
statement = root / 'statement.json'
raw = statement.read_bytes()
assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
sys.path.insert(0, str(root / 'src'))
from ovl_pipeline.preparation import validate_contract
validate_contract(json.loads(raw))
for name, module in tuple(sys.modules.items()):
    if name == 'ovl_pipeline' or name.startswith('ovl_pipeline.'):
        origin = pathlib.Path(module.__file__).resolve()
        assert origin.is_relative_to(root / 'src/ovl_pipeline'), name
assert statement.read_bytes() == raw
print(json.dumps({'result': 'PASS', 'statement_sha256': sys.argv[2]}, sort_keys=True, separators=(',', ':')))
'''


def _materialize(root, revision, entries, target):
    """Copy a closed regular-blob inventory; authenticate all bytes before import."""
    if type(entries) is not list or [e.get('path') for e in entries] != list(PREPARATION_PATHS):
        raise EvidenceError('historical preparation requires exact closed source inventory')
    listing = subprocess.check_output([
        'git', '-C', str(root), 'ls-tree', '-r', '-z', revision, '--', *PREPARATION_PATHS])
    objects = {}
    for row in filter(None, listing.decode().split('\0')):
        meta, name = row.split('\t', 1)
        mode, kind, oid = meta.split()
        if name in objects or name not in PREPARATION_PATHS or mode not in ('100644', '100755') or kind != 'blob':
            raise EvidenceError('historical preparation contains nonregular or unexpected source')
        objects[name] = oid
    if set(objects) != set(PREPARATION_PATHS):
        raise EvidenceError('historical preparation source is missing')
    for entry in entries:
        if set(entry) != {'path', 'bytes', 'sha256'}:
            raise EvidenceError('invalid historical preparation inventory entry')
        raw = subprocess.check_output(['git', '-C', str(root), 'cat-file', 'blob', objects[entry['path']]])
        if len(raw) != entry['bytes'] or sha256(raw) != entry['sha256']:
            raise EvidenceError('historical preparation source bytes differ')
        destination = target / entry['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)


def verify_preparation(root, statement):
    """Execute the unchanged strict validator under its authenticated source bytes."""
    raw = canonical(statement)
    statement_hash = sha256(raw)
    revision = PREPARATION_SNAPSHOTS.get(statement_hash)
    if revision is None:
        raise EvidenceError('preparation statement has no reviewed executable snapshot')
    with tempfile.TemporaryDirectory(prefix='ovl-preparation-contract-') as directory:
        target = Path(directory)
        _materialize(root, revision, statement['code'], target)
        (target / 'statement.json').write_bytes(raw)
        # Isolated Python excludes PYTHONPATH, user site and the working-directory
        # import path. Trusted dependency/site startup remains enabled.
        # Do not pass signing credentials or normalize a mismatching tokenizer flag.
        environment = {'LC_ALL': 'C.UTF-8'}
        if 'TOKENIZERS_PARALLELISM' in os.environ:
            environment['TOKENIZERS_PARALLELISM'] = os.environ['TOKENIZERS_PARALLELISM']
        try:
            result = subprocess.run(
                [sys.executable, '-I', '-B', '-c', _CHECK, str(target), statement_hash],
                cwd=target, env=environment, capture_output=True, timeout=120, check=False)
        except subprocess.TimeoutExpired as error:
            raise EvidenceError('historical preparation validation timed out') from error
        if result.returncode:
            raise EvidenceError('historical preparation contract/source/environment validation failed')
        if parse_json(result.stdout.strip(), canonical_required=True) != {
                'result': 'PASS', 'statement_sha256': statement_hash}:
            raise EvidenceError('historical preparation execution result differs')
        if (target / 'statement.json').read_bytes() != raw or inventory(target, PREPARATION_PATHS) != statement['code']:
            raise EvidenceError('historical preparation inputs changed during validation')
    return {'result': 'PASS', 'scope': 'historical-contract-source-and-committed-environment-profile-only',
            'source_statement_sha256': statement_hash, 'original_source_revision': statement['source_revision'],
            'retained_source_revision': revision, 'source_inventory_sha256': digest(statement['code']),
            'environment_sha256': digest(statement['environment']), 'raw_reconstruction': 'NOT_RUN'}
