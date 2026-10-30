import type { SiteData } from '../data/types';

/** Fill {matched} {total} {independent} {rows} {second} {training} {fresh} {m95} {m99} {N} in a copy line from site-data.json. Returns null (the caller
 *  withholds the line) when it would be false for this data: "{matched} of|/ {total}" while they differ,
 *  any "all match…" while a row is not MATCH (EX-* reads the exclusion data), or an unknown placeholder. */
export function fillCounts(id: string, text: string, d: SiteData): string | null {
  const v = d.verifications;
  const o = d.odds;
  const rowsOf = (c: string) => v.groups.find((g) => g.checker === c)?.rows ?? NaN;
  const n: Record<string, number> = {
    matched: v.segmentsMatched, total: d.chain.segments, independent: v.independentChecks, rows: v.rowCount,
    second: rowsOf(v.secondMachine), fresh: rowsOf(v.freshMachine), training: rowsOf('project training pod'),
    m95: o.m95, m99: o.m99, N: o.N,
  };
  if (/\{matched\}( of |\/)\{total\}/.test(text) && n.matched !== n.total) return null;
  if (/\ball match/i.test(text) && !(id.startsWith('EX-') ? d.exclusion.allMatch : v.allMatch)) return null;
  const out = text.replace(/\{(\w+)\}/g, (_, k: string) => (Number.isFinite(n[k]) ? String(n[k]) : '\0'));
  return out.includes('\0') ? null : out;
}
