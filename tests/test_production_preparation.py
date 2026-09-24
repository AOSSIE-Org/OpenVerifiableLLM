"""Synthetic source snapshots; no signatures, reconstruction or GPU credit."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ovl_pipeline import production_preparation as m
from ovl_pipeline.canonical import EvidenceError, canonical, digest, inventory
from test_production_commitment import repository


@pytest.fixture
def historical(tmp_path, monkeypatch):
    root = tmp_path / 'history'
    root.mkdir()
    git, commit = repository(root)
    for name in m.PREPARATION_PATHS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source fixture\n')
    (root / 'src/ovl_pipeline/preparation.py').write_text(
        'import os\n'
        'def validate_contract(value):\n'
        '    assert value["synthetic"] is True\n'
        '    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"\n'
        '    assert "SYNTHETIC_PRIVATE_SIGNING_ENV" not in os.environ\n')
    revision = commit()['GITHUB_SHA']
    statement = {'synthetic': True, 'source_revision': '1' * 40,
                 'environment': {'synthetic': True},
                 'code': inventory(root, m.PREPARATION_PATHS)}
    # Explicit synthetic execution allowlist. Production has no caller override.
    monkeypatch.setattr(m, 'PREPARATION_SNAPSHOTS', {digest(statement): revision})
    monkeypatch.setenv('TOKENIZERS_PARALLELISM', 'false')
    return root, git, commit, statement


def test_historical_bytes_are_separate_and_imports_isolated(historical, tmp_path, monkeypatch):
    root, git, commit, statement = historical
    # Dirty current source and a hostile PYTHONPATH must never be imported.
    (root / 'src/ovl_pipeline/preparation.py').write_text('raise RuntimeError("wrong tree")\n')
    monkeypatch.setenv('PYTHONPATH', str(root / 'src'))
    monkeypatch.setenv('SYNTHETIC_PRIVATE_SIGNING_ENV', 'synthetic-only')
    result = m.verify_preparation(root, statement)
    assert result['result'] == 'PASS'
    assert result['original_source_revision'] != result['retained_source_revision']
    assert result['raw_reconstruction'] == 'NOT_RUN'


def test_unreviewed_statement_never_materializes(historical, monkeypatch):
    root, _, _, statement = historical
    statement['source_revision'] = '2' * 40
    monkeypatch.setattr(m, '_materialize', lambda *a: pytest.fail('executed unreviewed input'))
    with pytest.raises(EvidenceError, match='no reviewed executable snapshot'):
        m.verify_preparation(root, statement)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'extra', 'traversal', 'bytes', 'length'])
def test_closed_inventory_and_hashes_before_execution(historical, monkeypatch, mutation):
    root, git, _, statement = historical
    statement = deepcopy(statement)
    if mutation == 'missing': statement['code'].pop()
    elif mutation == 'duplicate': statement['code'].append(statement['code'][0])
    elif mutation == 'extra': statement['code'].append({'path': 'extra.py', 'bytes': 1, 'sha256': '0' * 64})
    elif mutation == 'traversal': statement['code'][0]['path'] = '../outside.py'
    elif mutation == 'bytes': statement['code'][0]['sha256'] = '0' * 64
    else: statement['code'][0]['bytes'] += 1
    monkeypatch.setattr(m, 'PREPARATION_SNAPSHOTS', {digest(statement): git('rev-parse', 'HEAD')})
    with pytest.raises(EvidenceError, match='inventory|bytes differ'):
        m.verify_preparation(root, statement)


@pytest.mark.parametrize('mutation', ['symlink', 'missing'])
def test_nonregular_or_absent_git_source_rejected(historical, monkeypatch, mutation):
    root, git, _, statement = historical
    path = root / 'src/ovl_pipeline/data.py'
    path.unlink()
    if mutation == 'symlink': path.symlink_to('preparation.py')
    git('add', '.')
    # Existing invalid history in a remote-free adversarial fixture. No push and
    # no publication-hook bypass; ordinary commits still use installed hooks.
    assert not git('remote')
    bad = git('commit-tree', git('write-tree'), '-p', git('rev-parse', 'HEAD'), '-m', 'synthetic invalid tree')
    monkeypatch.setattr(m, 'PREPARATION_SNAPSHOTS', {digest(statement): bad})
    with pytest.raises(EvidenceError, match='nonregular|missing'):
        m.verify_preparation(root, statement)


@pytest.mark.parametrize('flag', [None, 'true'])
def test_environment_mismatch_is_not_normalized(historical, monkeypatch, flag):
    root, _, _, statement = historical
    if flag is None: monkeypatch.delenv('TOKENIZERS_PARALLELISM')
    else: monkeypatch.setenv('TOKENIZERS_PARALLELISM', flag)
    with pytest.raises(EvidenceError, match='environment validation failed'):
        m.verify_preparation(root, statement)


@pytest.mark.parametrize('case', ['nonzero', 'stale-output', 'timeout', 'changed-input', 'changed-statement', 'extra-output', 'malformed'])
def test_child_execution_failure_and_changed_bytes(historical, monkeypatch, case):
    root, _, _, statement = historical
    original = subprocess.run
    def run(command, **kwargs):
        if command[0] != sys.executable: return original(command, **kwargs)
        if case == 'timeout': raise subprocess.TimeoutExpired(command, 120)
        if case in ('changed-input', 'changed-statement'):
            result = original(command, **kwargs)
            name = 'src/ovl_pipeline/data.py' if case == 'changed-input' else 'statement.json'
            (Path(command[-2]) / name).write_text('# changed after execution')
            return result
        output = {'result': 'PASS', 'statement_sha256': digest(statement) if case == 'nonzero' else '0' * 64}
        if case in ('extra-output', 'malformed'):
            return subprocess.CompletedProcess(command, 0, canonical(output) + (b'\nextra' if case == 'extra-output' else b'}'), b'')
        return subprocess.CompletedProcess(command, 1 if case == 'nonzero' else 0, canonical(output), b'')
    monkeypatch.setattr(m.subprocess, 'run', run)
    with pytest.raises(EvidenceError): m.verify_preparation(root, statement)


def test_repeated_check_launches_two_fresh_children(historical, monkeypatch):
    root, _, _, statement = historical
    original = subprocess.run
    directories = []
    def run(command, **kwargs):
        if command[0] == sys.executable: directories.append(command[-2])
        return original(command, **kwargs)
    monkeypatch.setattr(m.subprocess, 'run', run)
    assert m.verify_preparation(root, statement)['result'] == 'PASS'
    assert m.verify_preparation(root, statement)['result'] == 'PASS'
    assert len(directories) == len(set(directories)) == 2
    assert all(not Path(path).exists() for path in directories)


@pytest.mark.parametrize('failure', [None, 'signature', 'parent', 'environment', 'training-code'])
def test_complete_generate_orchestration(historical, tmp_path, monkeypatch, failure):
    """Real Git request/download/parents/child; explicit signature and code doubles.

    Cryptographic verification and actual production source binding are tested
    separately. Synthetic assertions supply no production acceptance credit.
    """
    from dataclasses import asdict, replace
    from test_production_parents import parents, rebind
    from test_production_identity import policy
    from ovl_pipeline import production_commitment as signing, production_anchoring as anchors
    from ovl_pipeline.anchoring import WORKFLOW, PublisherPolicy
    from ovl_pipeline.canonical import write_json, read_json
    root, git, commit, historical_statement = historical
    registration, objects = parents()
    objects['source'].update(historical_statement)
    source = objects['source']
    source_policy = PublisherPolicy(**asdict(replace(policy(), workflow=WORKFLOW,
                                                    statement_sha256=digest(source))))
    objects['source_policy'] = asdict(source_policy)
    objects['prepared'].update(code=source['code'], environment=source['environment'],
                               source_commitment_sha256=digest(source))
    registration['source_policy_sha256'] = digest(objects['source_policy'])
    rebind(registration, objects)
    bundle = {'test': 'unsigned synthetic signature'}
    registration['source_bundle_sha256'] = digest(bundle)
    monkeypatch.setattr(m, 'PREPARATION_SNAPSHOTS', {digest(source): git('rev-parse', 'HEAD')})
    data = {name: canonical(objects[key]) for key, name in anchors.PARENT_FILES.items()}
    for phase in ('wikipedia', 'conversation'):
        for plural in ('records', 'replays'):
            data[f'{phase}-pilot-{plural[:-1]}.json'] = canonical(objects['pilot_' + plural][phase])
    data['registration.json'] = canonical(registration)
    data['source-statement.sigstore.json'] = canonical(bundle)
    if failure == 'parent': data['initial-record.json'] = canonical({'invalid': True})
    request = {'schema': 'ovl.production-signing-request.v1', 'registration_sha256': digest(registration),
               'source_policy': asdict(source_policy), 'packet': {
                   'repo': 'AOSSIE/openverifiable-synthetic-test-evidence', 'revision': '1' * 40,
                   'prefix': 'production-registration/' + digest(registration),
                   'inventory': [{'path': n, 'bytes': len(b), 'sha256': m.sha256(b)} for n, b in sorted(data.items())]}}
    name = registration['run_id'] + '-' + registration['attempt_id'] + '.json'
    write_json(root / signing.REQUEST_DIRECTORY / name, request)
    environ = commit()
    prefix = request['packet']['prefix']
    def fetch(url):
        if '/api/datasets/' in url:
            return canonical([{'type': 'file', 'path': prefix + '/' + n, 'size': len(b)} for n, b in data.items()])
        return data[url.rsplit('/', 1)[-1]]
    download = signing.download_packet
    monkeypatch.setattr(signing, 'download_packet', lambda request, output: download(request, output, fetch=fetch))
    calls = []
    def signature(*args, **kwargs):
        calls.append('signature')
        if failure == 'signature': raise EvidenceError('synthetic invalid signature')
        return {'result': 'PASS', 'test_double': True}
    monkeypatch.setattr(anchors, 'verify_anchor', signature)
    historical_check = m.verify_preparation
    def preparation(*args):
        assert calls == ['signature']
        calls.append('preparation')
        return historical_check(*args)
    monkeypatch.setattr(m, 'verify_preparation', preparation)
    def code(*args):
        assert calls == ['signature', 'preparation']
        calls.append('code')
        if failure == 'training-code': raise EvidenceError('synthetic training code mismatch')
        return {'result': 'PASS', 'test_double': True}
    monkeypatch.setattr(signing, 'verify_code', code)
    if failure == 'environment': monkeypatch.setenv('TOKENIZERS_PARALLELISM', 'true')
    output = tmp_path / 'endorsement-output'
    if failure:
        with pytest.raises(EvidenceError): signing.generate(root, environ, output)
        assert not (output / 'ci-self-check-policy.json').exists()
        if failure in ('signature', 'parent'): assert calls == ['signature']
    else:
        signing.generate(root, environ, output)
        checks = read_json(output / 'ci-parent-checks.json')
        assert calls == ['signature', 'preparation', 'code']
        assert checks['parents']['result'] == checks['preparation_contract']['result'] == 'PASS'
        assert checks['preparation_contract']['raw_reconstruction'] == 'NOT_RUN'
        assert read_json(output / 'request.json') == request
