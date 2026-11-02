import { describe, expect, it } from 'vitest';
import { DataError, detect, minChecks, validateSite, validateStrict } from '../../src/data/validate';
import type { SiteData } from '../../src/data/types';
import { fresh, sha } from './helpers';

const broken = (f: (d: SiteData) => void) => {
  const d = fresh();
  f(d);
  return d;
};

describe('validateSite', () => {
  it('accepts the committed data with chain hashes recomputed', () => {
    const { errors } = validateSite(fresh(), sha);
    expect(errors).toEqual({});
    expect(() => validateStrict(fresh(), sha)).not.toThrow();
  });

  it.each([
    ['unknown schema', (d: SiteData) => ((d as unknown as Record<string, unknown>).schema = 'v0')],
    ['unpinned revision', (d: SiteData) => (d.revision = 'main')],
    ['javascript: URL', (d: SiteData) => (d.anchor.src.url = 'javascript:alert(1)')],
  ])('throws on fatal problem: %s', (_n, f) => {
    expect(() => validateSite(broken(f))).toThrow(DataError);
  });

  const cases: [string, keyof ReturnType<typeof validateSite>['errors'], (d: SiteData) => void, RegExp][] = [
    ['unknown result', 'verifications', (d) => ((d.verifications.rows[0] as { result: string }).result = 'ok'), /unknown result/],
    ['row count says 37 instead of 46', 'verifications', (d) => (d.verifications.rowCount = 37), /row count/],
    ['missing tile', 'verifications', (d) => d.verifications.tiles.pop(), /tile count/],
    ['tile result drift', 'verifications', (d) => (d.verifications.tiles[4]!.result = 'MISMATCH'), /result drift/],
    ['independent count drift', 'verifications', (d) => (d.verifications.independentChecks = 1), /independent count/],
    ['fresh machine claimed independent', 'verifications', (d) => (d.verifications.rows[45]!.checkedBy = 'independent'), /checkedBy/],
    ['group drift', 'verifications', (d) => (d.verifications.groups[1]!.rows = 9), /row count drift/],
    ['segment 8 added to fresh machine', 'verifications', (d) => d.verifications.freshSegments.unshift(8), /fresh-machine/],
    ['broken chain link', 'chain', (d) => (d.chain.boundaries[5]!.previous = '0'.repeat(64)), /chain link broken/],
    ['odds M not minimal', 'odds', (d) => (d.odds.m95 = 111), /minimal/],
    ['odds probability typed', 'odds', (d) => (d.odds.p99 = 0.99), /probability/],
    ['event not marked on tile', 'events', (d) => (d.verifications.tiles[10]!.events = []), /not marked/],
    ['code change moved', 'events', (d) => { d.events[1]!.segment = 18; d.verifications.tiles[18]!.events = ['code-change']; }, /code-change/],
    ['hardware result', 'hardware', (d) => ((d.hardware[0] as { result: string }).result = 'ok'), /hardware\[0\]/],
    ['bad sha256', 'downloads', (d) => (d.downloads[0]!.files[0]!.sha256 = 'xyz'), /bad sha256/],
    ['chain head mismatch', 'anchor', (d) => (d.anchor.chainHead = 'a'.repeat(64)), /chain head/],
    ['loss delta typed', 'exclusion', (d) => (d.exclusion.removedLoss.delta = 0.03), /delta/],
    ['commands from the v1.0.0 tag', 'check', (d) => { d.check.commands = d.check.commands.replace('v1.0.1', 'v1.0.0'); }, /v1\.0\.1/],
    ['cost without citation', 'check', (d) => (d.check.costSrc.line = 86), /simple_run/],
  ];
  it.each(cases)('flags %s in its own section only', (_n, section, f, msg) => {
    const { errors } = validateSite(broken(f), sha);
    expect(errors[section]).toMatch(msg);
    expect(validateSite(fresh(), sha).errors[section]).toBeUndefined();
    expect(() => validateStrict(broken(f), sha)).toThrow(DataError);
  });

  it('keeps other sections usable when one is malformed', () => {
    const { errors } = validateSite(broken((d) => ((d as unknown as Record<string, unknown>).hardware = 'nonsense')), sha);
    expect(Object.keys(errors)).toEqual(['hardware']);
  });
});

describe('detection odds', () => {
  it('matches 1-(36/37)^M at the documented points', () => {
    expect(detect(37, 110)).toBeCloseTo(0.9509, 4);
    expect(detect(37, 169)).toBeCloseTo(0.99025, 5);
    expect(minChecks(37, 0.95)).toBe(110);
    expect(minChecks(37, 0.99)).toBe(169);
    expect(detect(37, 0)).toBe(0);
  });
});
