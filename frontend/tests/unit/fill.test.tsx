import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render } from '@testing-library/react';
import { fillCounts } from '../../src/content/fill';
import { copy } from '../../src/content/ledger';
import { ClaimBlock } from '../../src/components/ClaimBlock';
import { DataContext } from '../../src/data/context';
import { fresh } from './helpers';

afterEach(cleanup);

describe('counts come from the data, never from copy', () => {
  it('fills {matched} of {total} from site-data.json', () => {
    const d = fresh();
    expect(fillCounts('OV-04', copy['OV-04']!.plain, d)).toContain(`${d.verifications.segmentsMatched} of ${d.chain.segments} segments`);
  });
  it('no copy line hardcodes "N of N" or "N/N" segments', () => {
    for (const [id, v] of Object.entries(copy)) for (const t of Object.values(v)) expect(t, id).not.toMatch(/\b\d+ of \d+ segments|\b\d+\/\d+ segments/);
  });
  it('withholds "{matched} of {total}" when a segment is not MATCH', () => {
    const d = fresh();
    d.verifications.tiles[5]!.result = 'MISMATCH';
    d.verifications.segmentsMatched -= 1;
    d.verifications.allMatch = false;
    expect(fillCounts('OV-04', copy['OV-04']!.plain, d)).toBeNull();
    expect(fillCounts('VE-01', copy['VE-01']!.plain, d)).toBeNull();
  });
  it('withholds any "all match" line when allMatch is false, and EX-* reads the exclusion data', () => {
    const d = fresh();
    d.verifications.allMatch = false;
    expect(fillCounts('VE-03', 'There are 46 check results, all matches', d)).toBeNull();
    expect(fillCounts('EX-05', 'all MATCH', d)).toBe('all MATCH');
    d.exclusion.allMatch = false;
    expect(fillCounts('EX-05', 'all MATCH', d)).toBeNull();
  });
  it('an unknown placeholder is withheld, not printed', () => {
    expect(fillCounts('X', 'got {nope}', fresh())).toBeNull();
  });
  it('ClaimBlock shows the scoped error, not the stale count, when data disagrees', () => {
    const d = fresh();
    d.verifications.segmentsMatched = 36;
    d.verifications.allMatch = false;
    const { container } = render(
      <DataContext.Provider value={{ data: d, errors: {} }}>
        <ClaimBlock id="OV-04" />
      </DataContext.Provider>,
    );
    expect(container.querySelector('[data-claim-part="plain"]')).toBeNull();
    expect(container.querySelector('[data-claim-part="withheld"]')).not.toBeNull();
    expect(container.querySelector('[data-claim-part="developer"]')).toBeNull();
  });
});

describe('shipped copy.json, filled from the shipped data, agrees with the ledger', () => {
  const nums = (t: string) => new Set((t.match(/\d[\d,]*/g) ?? []).map((x) => x.replace(/,/g, '')));
  it('every placeholder value appears as a literal number in that claim\'s ledger text', async () => {
    const { ledger } = await import('../../src/content/ledger');
    const d = fresh();
    for (const [id, v] of Object.entries(copy)) {
      const lc = ledger.claims.find((c) => c.id === id);
      for (const t of Object.values(v)) {
        const keys = [...t.matchAll(/\{(\w+)\}/g)].map((m) => m[1]!);
        if (!keys.length) continue;
        expect(lc, id).toBeTruthy();
        const filled = fillCounts(id, t, d);
        expect(filled, `${id}: ${t}`).not.toBeNull();
        const lit = nums(`${lc!.plain} ${lc!.developer}`);
        const tpl = t.split(/\{\w+\}/);
        // value each placeholder took, recovered from the filled text
        let rest = filled!;
        for (let i = 0; i < keys.length; i++) {
          rest = rest.slice(tpl[i]!.length);
          const val = /^\d+/.exec(rest)![0];
          expect(lit.has(val), `${id} {${keys[i]}}=${val} not in ledger text`).toBe(true);
          rest = rest.slice(val.length);
        }
      }
    }
  });
  it('templated claim ids are exactly OV-04, VE-01, VE-03, VE-07, VE-08', () => {
    expect(Object.entries(copy).filter(([, v]) => Object.values(v).some((t) => /\{\w+\}/.test(t))).map(([id]) => id)).toEqual(['OV-04', 'VE-01', 'VE-03', 'VE-07', 'VE-08']);
  });
  it('fills detection-odds copy from site-data.json, not literals', () => {
    const d = fresh();
    expect(fillCounts('VE-07', copy['VE-07']!.plain, d)).toContain(`${d.odds.m95} independent random checks of ${d.chain.segments} segments`);
    expect(fillCounts('VE-08', copy['VE-08']!.plain, d)).toContain(`${d.odds.m99} independent random checks`);
  });
});
