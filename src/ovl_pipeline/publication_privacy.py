"""Review-bound publication inventory. No computation or trust-policy credit."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
import tarfile

from .canonical import EvidenceError, digest, file_hash, read_json

_PRIVATE_PATH = re.compile(r'(^|/)(?:AGENTS\.md|CLAUDE\.md|PROJECT_GOAL\.md|AUTONOMOUS_START\.md|goal_state\.json|RESUME\.md|COORDINATED_CLEANUP\.md)$|(^|/)(?:[^/]*advisory[^/]*|prompts?|private|\.git|\.ovllm-cache)(/|$)|(?:advisory-spec|astra-disposition|cli-process|usage-monitor)', re.I)
_PRIVATE_TEXT = re.compile(rb'/home/[^/\s]+/|/tmp/pytest-of-[^/\s]+/|"(?:account_balance_usd|baseline_balance_usd|balance_usd|account_balance|clientBalance|currentSpend|credit_balance|operator_conversation|owner_instruction|native_advisory)"\s*:|Astra/high|claude-opus-5')
_CHECKPOINT_FIELDS = frozenset('schema scope result observed_epoch pod_id bytes files stages cycle_verification_sha256 qualification_checkpoint_sha256 sustained_verification_sha256 record_checkpoint_sha256 failure_checkpoint_sha256 workload_plan_sha256 retained_export_archive_sha256 retained_export_inventory_sha256 privacy_reissued scope_notice'.split())


def path_allowed(name):
    p = PurePosixPath(name)
    if (not name or p.is_absolute() or '..' in p.parts or str(p) != name or '\\' in name or _PRIVATE_PATH.search(name)):
        raise EvidenceError('private or unsafe public export path')


def check_text(data, public_examples=()):
    # Normalize ASCII JSON escapes so escaping does not hide a private field.
    normalized = re.sub(rb'\\u00([0-9a-fA-F]{2})', lambda m: bytes([int(m[1], 16)]), data).replace(b'\\/', b'/')
    for match in _PRIVATE_TEXT.finditer(normalized):
        if match[0] not in public_examples:
            raise EvidenceError('private coordination or account content in public export')


def review_export(plan, staging, review_path):
    """Require a local reviewed exact allowlist; review itself is never uploaded.

    This controls operational exports only. Public conversation training inputs
    keep their own source/selection policy and are not operator conversations.
    A scanner is an additional guard, not evidence that human context is public.
    """
    if plan['kind'] != 'operational-evidence':
        return
    review = read_json(review_path)
    if (set(review) - {'public_dependency_examples'} != {'schema', 'plan_sha256', 'files', 'archive_members', 'checkpoint_fields'}
            or review['schema'] != 'ovl.operational-publication-review.v1'
            or review['plan_sha256'] != digest(plan) or review['files'] != plan['files']):
        raise EvidenceError('missing or mismatched reviewed publication allowlist')
    checkpoint = read_json(staging / 'checkpoint.json')
    if (set(checkpoint) - _CHECKPOINT_FIELDS or sorted(checkpoint) != review['checkpoint_fields']):
        raise EvidenceError('unreviewed operational checkpoint fields')
    inventory = read_json(staging / 'retained-export-inventory.json')
    if inventory != review['archive_members']:
        raise EvidenceError('archive members differ from reviewed allowlist')
    for entry in plan['files']:
        path_allowed(entry['path'])
        f = staging / entry['path']
        if f.is_symlink() or not f.is_file() or f.stat().st_size != entry['bytes'] or file_hash(f) != entry['sha256']:
            raise EvidenceError('reviewed public file bytes changed')
        if not entry['path'].endswith('.tar.gz'):
            check_text(f.read_bytes())
    expected = {}
    for entry in inventory:
        if set(entry) != {'path', 'bytes', 'sha256'} or entry['path'] in expected:
            raise EvidenceError('invalid reviewed archive inventory')
        path_allowed(entry['path']); expected[entry['path']] = entry
    # Exceptions are local review decisions for exact upstream documentation
    # examples in an exact member, never permission to waive operator fields.
    examples = {}
    for item in review.get('public_dependency_examples', []):
        if (set(item) != {'path', 'sha256', 'markers', 'source_url', 'source_sha256', 'reason'}
                or item['path'] not in expected or item['path'] in examples
                or item['sha256'] != expected[item['path']]['sha256']
                or not re.fullmatch(r'https://raw\.githubusercontent\.com/[^/]+/[^/]+/[0-9a-f]{40}/.+', item['source_url'])
                or not re.fullmatch(r'[0-9a-f]{64}', item['source_sha256'])
                or not isinstance(item['reason'], str) or not item['reason'].strip()
                or not isinstance(item['markers'], list) or not item['markers']
                or any(not isinstance(m, str) or not re.fullmatch(r'/home/[A-Za-z0-9_-]+/', m) for m in item['markers'])):
            raise EvidenceError('invalid exact-member public dependency review')
        examples[item['path']] = tuple(m.encode('ascii') for m in item['markers'])
    seen = set()
    with tarfile.open(staging / 'retained-exports.tar.gz', 'r|gz') as archive:
        for member in archive:
            path_allowed(member.name)
            if not member.isfile() or member.name in seen or member.name not in expected:
                raise EvidenceError('unreviewed or nonregular archive member')
            seen.add(member.name); entry = expected[member.name]
            if member.size != entry['bytes']:
                raise EvidenceError('reviewed archive member size changed')
            stream = archive.extractfile(member); hashed = hashlib.sha256(); tail = b''
            # Nested archives and Git object stores need their own explicit
            # transformation/review rather than hiding inside this export.
            if member.name.endswith(('.tar', '.gz', '.zip', '.pack', '.bz2', '.xz')):
                raise EvidenceError('nested archive requires separate reviewed publication')
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hashed.update(block)
                check_text(tail + block, examples.get(member.name, ())); tail = block[-256:]
            if hashed.hexdigest() != entry['sha256']:
                raise EvidenceError('reviewed archive member bytes changed')
    if seen != set(expected):
        raise EvidenceError('missing reviewed archive member')
