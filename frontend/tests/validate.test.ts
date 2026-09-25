import { describe, expect, it } from 'vitest';
import { loadSnapshot } from '../src/data/adapters/snapshotAdapter';
import { SnapshotError } from '../src/data/contracts';
import { releaseIsDownloadable, validateSnapshot } from '../src/data/validate';
import emptyChecks from '../src/data/fixtures/empty-checks.json';
import publicSnapshot from '../src/data/fixtures/public-snapshot.json';

describe('validateSnapshot', () => {
  it('accepts the public unreleased snapshot', () => {
    const snap = validateSnapshot(publicSnapshot);
    expect(snap.mode).toBe('public-snapshot');
    expect(snap.releases.every((item) => item.availability === 'not-released')).toBe(
      true,
    );
    expect(snap.checks.some((item) => item.result === 'PASS')).toBe(true);
    expect(snap.checks.find((item) => item.id === 'chk-g01')?.locallyRecomputed).toBe(
      false,
    );
  });

  it('refuses an empty check list', () => {
    expect(() => validateSnapshot(emptyChecks)).toThrow(SnapshotError);
    expect(() => validateSnapshot(emptyChecks)).toThrow(/no checks/);
  });

  it('does not treat ok as PASS', () => {
    const raw = structuredClone(publicSnapshot) as Record<string, unknown>;
    const checks = raw.checks as Record<string, unknown>[];
    checks[0] = { ...checks[0], result: 'ok' };
    expect(() => validateSnapshot(raw)).toThrow(/not a known check result/);
  });

  it('rejects PASS without evidence references', () => {
    const raw = structuredClone(publicSnapshot) as Record<string, unknown>;
    const checks = raw.checks as Record<string, unknown>[];
    checks[0] = { ...checks[0], evidence: [] };
    expect(() => validateSnapshot(raw)).toThrow(/missing evidence/);
  });

  it('rejects a javascript: locator', () => {
    const raw = structuredClone(publicSnapshot) as Record<string, unknown>;
    raw.source = { ...(raw.source as object), url: 'javascript:alert(1)' };
    expect(() => validateSnapshot(raw)).toThrow(/disallowed scheme/);
  });
});

describe('loadSnapshot', () => {
  it('loads missing-parent without pretending the parent exists', () => {
    const snap = loadSnapshot('missing-parent');
    const child = snap.evidence[0];
    const ids = new Set(snap.evidence.map((item) => item.id));
    expect(child.parentIds.some((id) => !ids.has(id))).toBe(true);
  });

  it('keeps superseded FAIL rows', () => {
    const snap = loadSnapshot('superseded');
    const failed = snap.evidence.find((item) => item.result === 'FAIL');
    const passed = snap.evidence.find((item) => item.result === 'PASS');
    expect(failed?.supersededBy).toBe(passed?.id);
    expect(passed?.supersedes).toBe(failed?.id);
  });
});

describe('release gating', () => {
  it('does not treat unreleased cards as downloadable', () => {
    const snap = loadSnapshot('public');
    expect(snap.releases.every((item) => !releaseIsDownloadable(item))).toBe(true);
  });
});
