import { describe, expect, it, vi } from 'vitest';
import fixture from '../public/data/snapshot.json';
import { immutableFileUrl, missingCheckReferences, validateSnapshot } from '../src/data/contracts';
import { scenario } from '../src/data/scenarios';
import { isStale, loadSnapshot, snapshotUrl, MAX_SNAPSHOT_BYTES } from '../src/data/load';

describe('presentation contract fails closed', () => {
  it('preserves a scoped pilot pass, pending workflow and NOT_RUN production separately', () => {
    const s = validateSnapshot(fixture);
    expect(s.workflow.state).toBe('pending');
    expect(s.checks[0]).toMatchObject({ scope: 'pilot', result: 'PASS', locallyRecomputed: false });
    expect(s.checks[1]).toMatchObject({ scope: 'production', result: 'NOT_RUN' });
    expect(s.releases.every(r => r.availability === 'not-released')).toBe(true);
  });
  it.each(['checks', 'evidence', 'releases', 'commands', 'sources'])('rejects absent required %s array', key => {
    const data = structuredClone(fixture) as Record<string, unknown>;
    delete data[key]; expect(() => validateSnapshot(data)).toThrow();
  });
  it.each(['malformed', 'unsupported-schema', 'public-unapproved'])('rejects %s instead of returning a success fixture', name => expect(() => validateSnapshot(scenario(name))).toThrow());
  it('accepts empty checks without inserting a success', () => expect(validateSnapshot(scenario('empty')).checks).toEqual([]));
  it('rejects mode mismatch', () => expect(() => validateSnapshot(fixture, 'public-snapshot')).toThrow('mode'));
  it('requires fixture identity designation', () => {
    const s = structuredClone(fixture); s.evidence[0].synthetic = false;
    expect(() => validateSnapshot(s)).toThrow();
  });
  it('rejects synthetic evidence even when public approval fields are present', () => {
    const revision = 'a'.repeat(40);
    const data = {
      ...structuredClone(fixture), mode: 'public-snapshot',
      approval: { approvedBy: 'Schema test only', approvedAt: fixture.generatedAt, revision },
      sources: [{ url: 'https://example.org/snapshot.json', revision, path: null, sha256: null }],
    };
    expect(() => validateSnapshot(data, 'public-snapshot')).toThrow('cannot contain synthetic evidence');
  });
  it('rejects unknown result and workflow values', () => {
    expect(() => validateSnapshot({ ...fixture, checks: [{ ...fixture.checks[0], result: 'complete' }] })).toThrow();
    expect(() => validateSnapshot({ ...fixture, workflow: { ...fixture.workflow, state: 'PASS' } })).toThrow();
  });
  it('rejects invalid dates, digests, duplicate IDs and unsafe source URLs', () => {
    for (const data of [
      { ...fixture, generatedAt: 'yesterday' },
      { ...fixture, evidence: [...fixture.evidence, fixture.evidence[0]] },
      { ...fixture, sources: [{ ...fixture.sources[0], url: 'javascript:alert(1)' }] },
      { ...fixture, evidence: [{ ...fixture.evidence[0], source: { ...fixture.evidence[0].source, sha256: 'bad' } }] },
    ]) expect(() => validateSnapshot(data)).toThrow();
  });
  it('does not infer independent attribution without supporting evidence', () => {
    const s = structuredClone(fixture); s.checks[0].attribution = 'independent-third-party';
    expect(() => validateSnapshot(s)).toThrow('Independent');
  });
  it('requires evidence references for a reported pass', () => {
    const s = structuredClone(fixture); s.checks[0].evidenceIds = [];
    expect(() => validateSnapshot(s)).toThrow();
  });
  it('keeps missing references explicit rather than resolving them to other records', () => {
    const s = validateSnapshot(scenario('missing-evidence'));
    expect(missingCheckReferences(s, s.checks[0])).toEqual(['synthetic-pilot-replay']);
  });
  it('preserves superseded original failures and replacement passes separately', () => {
    const s = validateSnapshot(fixture);
    expect(s.evidence.find(e => e.id === 'synthetic-replay-old')?.supersededBy).toBe('synthetic-pilot-replay');
    expect(s.checks.find(c => c.evidenceIds.includes('synthetic-replay-old'))?.result).toBe('FAIL');
    expect(s.checks[0].result).toBe('PASS');
  });
  it('does not produce download URLs for unavailable or uninventoried files', () => {
    const s = validateSnapshot(fixture); expect(immutableFileUrl(s.releases[0], 'model.safetensors')).toBeNull();
    const future = validateSnapshot(scenario('future-release')).releases[0];
    expect(immutableFileUrl(future, 'model.safetensors')).toContain('/resolve/' + 'a'.repeat(40) + '/model.safetensors');
    expect(immutableFileUrl(future, '../../private')).toBeNull();
    expect(() => validateSnapshot({ ...fixture, releases: [{ ...future, identity: { ...future.identity, revision: 'main' } }, fixture.releases[1]] })).toThrow();
  });
  it('requires an available chat release parent to match the base release ID', () => {
    const s = structuredClone(scenario('future-release')) as any;
    s.releases[1].parentReleaseId = 'unrelated-id';
    expect(() => validateSnapshot(s)).toThrow('Chat release parent must reference the base release');
  });
  it('rejects untested command syntax and commands with missing validation evidence', () => {
    expect(() => validateSnapshot({ ...fixture, commands: [{ id: 'command', approved: false }] })).toThrow();
  });
});
describe('small snapshot loader', () => {
  it('confines metadata paths to the deployed directory', () => {
    expect(snapshotUrl('/project/', 'data/snapshot.json', 'https://site.test/project/#/evidence').href).toBe('https://site.test/project/data/snapshot.json');
    expect(() => snapshotUrl('/project/', '../private.json', 'https://site.test/project/')).toThrow();
    expect(() => snapshotUrl('/project', '../project-private/x.json', 'https://site.test/project/')).toThrow();
    expect(() => snapshotUrl('./', 'https://other.test/payload', 'https://site.test/')).toThrow();
  });
  it('reports stale data independently of verification', () => {
    const s = validateSnapshot(scenario('stale')); expect(isStale(s)).toBe(true); expect(s.checks[0].result).toBe('PASS');
  });
  it('loads and validates without setting locallyRecomputed', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(fixture))));
    const s = await loadSnapshot(new URL('https://site.test/data.json'), 'fixture', new AbortController().signal);
    expect(s.checks.every(c => !c.locallyRecomputed)).toBe(true);
    vi.unstubAllGlobals();
  });
  it.each([new Response('not JSON'), new Response('{}'), new Response('', { status: 404 }), new Response('x'.repeat(MAX_SNAPSHOT_BYTES + 1))])('rejects invalid, missing or excessive response data', async response => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response));
    await expect(loadSnapshot(new URL('https://site.test/data.json'), 'fixture', new AbortController().signal)).rejects.toThrow();
    vi.unstubAllGlobals();
  });
});
