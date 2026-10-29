import { describe, expect, it } from 'vitest';
import { bannedIn, buildClaim, claim, claims, copy, ledger, numbersIn, ui, whoChecked } from '../../src/content/ledger';
import { auditClaims, KNOWN_ISSUES, unsupportedNumbers } from '../../src/content/audit';
import { fresh } from './helpers';

describe('claim ledger', () => {
  it('renders only ALLOWED claims (71 of 74)', () => {
    expect(ledger.claims).toHaveLength(74);
    expect(claims.size).toBe(71);
    for (const id of ['VE-05', 'UN-01', 'UN-02']) expect(() => claim(id)).toThrow(/not an ALLOWED/);
  });

  it('uses the Open Source Ops headline for every allowed claim', () => {
    for (const c of claims.values()) expect(c.plain).toBe(copy[c.id]?.plain);
  });

  it('falls back to the ledger line when no copy exists (copy can be swapped)', () => {
    const raw = ledger.claims.find((c) => c.id === 'OV-01')!;
    expect(buildClaim(raw, {}).plain).toBe(raw.plain);
    expect(buildClaim(raw, { 'OV-01': { plain: 'Swapped.' } }).plain).toBe('Swapped.');
  });

  it('never marks a check as independent (there are none in the pinned data)', () => {
    for (const c of ledger.claims) expect(whoChecked(c)).not.toBe('independent');
    expect(claim('VE-04').who).toBe('project');
    expect(claim('OV-05').who).toBe('project');
  });

  it('passes the full audit: sources, banned phrases, numbers, verbatim commands', () => {
    expect(auditClaims(fresh())).toEqual([]);
  });

  it('keeps the known-issue list exact, so a new unsupported number fails', () => {
    const found = unsupportedNumbers(fresh()).map((u) => `${u.id}:${u.number}`);
    expect(found).toEqual(KNOWN_ISSUES.map((u) => `${u.id}:${u.number}`));
  });

  it('flags a number that is not in the sources or the data', () => {
    const d = fresh();
    d.claimSources['OV-01'] = d.claimSources['OV-01']!.map((s) => ({ ...s, text: '', quote: '' }));
    d.model.millionParameters = 24;
    expect(auditClaims(d).some((e) => e.startsWith('OV-01'))).toBe(true);
  });

  it('flags a claim with no source', () => {
    const d = fresh();
    delete d.claimSources['VE-03'];
    expect(auditClaims(d)).toContain('VE-03: no source');
  });
});

describe('banned phrases', () => {
  it.each(ledger.forbidden_phrases.map((b) => [b.phrase]))('catches "%s"', (phrase) => {
    expect(bannedIn(`Some text with ${phrase} in it`).length).toBeGreaterThan(0);
  });
  it('applies the ledger exceptions', () => {
    expect(bannedIn('release/unlearning-3i-atlas/release-hashes.txt')).toEqual([]);
    expect(bannedIn('about US$1–1.50 rented')).toEqual([]);
    expect(bannedIn('Signed results are on the roadmap.')).toEqual([]);
  });
  it('enforces the brief wording rules', () => {
    expect(bannedIn('We checked it')).toContain('bare "we"');
    expect(bannedIn('after 100 training steps')).toContain('say "segments", not "steps"');
    expect(bannedIn('segment 8 matched on a fresh machine')).toContain('segment 8 is not in the pinned data');
    expect(bannedIn('an independent fresh machine')).toContain('the fresh machine is not independent');
  });
  it('finds none in UI labels', () => {
    for (const v of Object.values(ui)) expect(bannedIn(v)).toEqual([]);
  });
});

describe('numbersIn', () => {
  it('keeps thousands separators and splits lists', () => {
    expect(numbersIn('7,118,121,139 targets; segments (30,31) and 3.3860')).toEqual(['7118121139', '30', '31', '3.3860']);
  });
});
