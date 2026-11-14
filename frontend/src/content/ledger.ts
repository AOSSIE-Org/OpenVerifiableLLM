// The site's words come from three files, none written by the site code:
//   claims.json  the claim ledger (Research Lab): ids, sources, limits, banned phrases
//   copy.json    plain-language headlines per claim id (Open Source Ops, from COPY.md)
//   ui.json      navigation, headings and table labels (no facts, no numbers)
import ledgerJson from './claims.json' with { type: 'json' };
import copyJson from './copy.json' with { type: 'json' };
import { ui } from './ui';

export interface LedgerSource { rev: string; file: string; lines: string | number; quote?: string | null }
export interface LedgerClaim {
  id: string;
  page: string;
  status: 'ALLOWED' | 'CONFLICT' | 'UNSOURCED' | 'FORBIDDEN';
  plain: string;
  developer: string;
  value: unknown;
  sources: LedgerSource[];
  rule: string | null;
  limit: string | null;
  note: string | null;
  verbatim: string | null;
}
export interface Banned { phrase: string; why: string; exception?: string }
export interface Ledger { schema: 'ovllm.claims.v1'; pins: Record<string, string>; claims: LedgerClaim[]; forbidden_phrases: Banned[] }
export type Who = 'project' | 'independent' | null;
export interface Claim { id: string; page: string; plain: string; developer: string; limit: string | null; verbatim: string | null; who: Who; sources: LedgerSource[] }

export const ledger = ledgerJson as unknown as Ledger;
export const copy = copyJson as Record<string, { plain: string; developer?: string }>;
export { ui };

const CHECK_FILES = /verifications\.jsonl|hardware-map\.jsonl|loo-results\.json/;

/** Who ran the check behind a claim. null = the claim states a fact or a rule, not a check result. */
export function whoChecked(c: LedgerClaim): Who {
  const isCheck = c.sources.some((s) => CHECK_FILES.test(s.file)) || /checked by the project/i.test(c.rule ?? '');
  if (!isCheck) return null;
  return 'project';
}

const dash = (s: string | null) => (s === null || s.trim() === '' || s.trim() === '—' ? null : s);

export function buildClaim(c: LedgerClaim, cp = copy): Claim {
  const over = cp[c.id];
  return {
    id: c.id,
    page: c.page,
    plain: over?.plain ?? c.plain,
    developer: over?.developer ?? c.developer,
    limit: dash(c.limit),
    verbatim: c.verbatim,
    who: whoChecked(c),
    sources: c.sources,
  };
}

/** Only ALLOWED claims are ever rendered. */
export const claims: ReadonlyMap<string, Claim> = new Map(
  ledger.claims.filter((c) => c.status === 'ALLOWED').map((c) => [c.id, buildClaim(c)]),
);

export function claim(id: string): Claim {
  const c = claims.get(id);
  if (!c) throw new Error(`claim ${id} is not an ALLOWED ledger claim`);
  return c;
}
export { label, hasLabel } from './ui';
export const claimsOn = (page: string, prefix?: string) =>
  [...claims.values()].filter((c) => c.page === page && (!prefix || c.id.startsWith(prefix)));

/** Banned phrases from the ledger, with the ledger's own exceptions, plus the brief's wording rules. */
/** styleRules: the brief's wording rules for the site's own copy; off when scanning verbatim source quotes. */
export function bannedIn(text: string, banned: Banned[] = ledger.forbidden_phrases, styleRules = true): string[] {
  const hits: string[] = [];
  const low = text.toLowerCase();
  for (const b of banned) {
    const p = b.phrase.toLowerCase();
    let t = low;
    if (p.includes('unlearn')) t = t.replaceAll('unlearning-3i-atlas', '');
    if (p === 'us$1.50') t = t.replaceAll('us$1–1.50', '');
    if (p === 'signed results' || p === 'signed check results') t = t.replace(/[^.]*roadmap[^.]*/g, '');
    if (t.includes(p)) hits.push(b.phrase);
  }
  const rules: [RegExp, string][] = [
    [/\bwe\b/, 'bare "we"'],
    [/\btraining steps?\b/, 'say "segments", not "steps"'],
    [/\bone file\b/, 'never "one file"'],
    [/\bsegment 8\b[^.\n]*\b(?:fresh|match)/, 'segment 8 is not in the pinned data'],
    [/\bindependent(?:ly)? (?:fresh|computer|machine)\b/, 'the fresh machine is not independent'],
  ];
  if (styleRules) for (const [re, why] of rules) if (re.test(low)) hits.push(why);
  return hits;
}

const NUM = /\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?/g;
/** Hash ids and the word SHA-256 are identifiers, not quantities. */
const HASH_ID = /[0-9a-f]{4,}(?:…|\.\.\.)[0-9a-f]*|sha-?256|\b(?=[0-9a-f]*[a-f])[0-9a-f]{8,}\b/gi;
export const numbersIn = (s: string): string[] =>
  [...s.replace(HASH_ID, ' ').matchAll(NUM)].map((m) => m[0].replace(/,/g, '').replace(/\.$/, ''));
