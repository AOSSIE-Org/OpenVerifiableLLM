import type { SiteData } from '../data/types';

/** Count placeholders a copy line may use. Each is read from site-data.json, never typed in copy. */
export const COUNTS = {
  matched: (d: SiteData) => d.verifications.segmentsMatched,
  total: (d: SiteData) => d.chain.segments,
  independent: (d: SiteData) => d.verifications.independentChecks,
} as const;
export type CountKey = keyof typeof COUNTS;

/** Fill {matched} {total} {independent} from the data. Returns null when the line would be false for this
 *  data, so the caller withholds it instead of showing a stale or overstated count:
 *  - "{matched} of {total}" / "{matched}/{total}" while matched != total;
 *  - any "all match(ed|es)" / "all MATCH" while the data has a non-MATCH row (exclusion data for EX-*);
 *  - an unknown placeholder. */
export function fillCounts(id: string, text: string, d: SiteData): string | null {
  if (/\{matched\}( of |\/)\{total\}/.test(text) && d.verifications.segmentsMatched !== d.chain.segments) return null;
  const allMatch = id.startsWith('EX-') ? d.exclusion.allMatch : d.verifications.allMatch;
  if (/\ball match/i.test(text) && !allMatch) return null;
  let bad = false;
  const out = text.replace(/\{(\w+)\}/g, (_, k: string) => {
    if (!(k in COUNTS)) {
      bad = true;
      return '';
    }
    return String(COUNTS[k as CountKey](d));
  });
  return bad ? null : out;
}
