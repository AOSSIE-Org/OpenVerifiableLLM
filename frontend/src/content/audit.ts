import type { SiteData } from '../data/types';
import { formatValue } from '../data/format';
import issues from './ledger-issues.json' with { type: 'json' };
import { bannedIn, claims, ledger, numbersIn, ui } from './ledger';

/** Every number a reader sees in a claim must be found in the claim's cited source lines or in site-data.json. */
export function numberUniverse(data: SiteData): Set<string> {
  const out = new Set<string>();
  const add = (v: unknown): void => {
    if (typeof v === 'number') {
      out.add(String(v));
      out.add(String(Math.trunc(v)));
      out.add(formatValue(v, 'pct2').replace('%', ''));
      for (let d = 1; d <= 5; d++) out.add(v.toFixed(d));
    } else if (typeof v === 'string') numbersIn(v).forEach((n) => out.add(n));
    else if (Array.isArray(v)) v.forEach(add);
    else if (v && typeof v === 'object') Object.values(v).forEach(add);
  };
  add({ ...data, claimSources: undefined });
  return out;
}

const same = (a: string, set: Set<string>) => set.has(a) || [...set].some((b) => Number(b) === Number(a) && !Number.isNaN(Number(a)));

export interface LedgerIssue { id: string; part: string; number: string; reason: string }
export const KNOWN_ISSUES = issues as LedgerIssue[];
const known = (id: string, part: string, n: string) => KNOWN_ISSUES.some((i) => i.id === id && i.part === part && i.number === n);

/** Numbers the ledger states that the pinned files do not support. Reported, and pinned in ledger-issues.json
 *  so the build still fails on any new one. */
export function unsupportedNumbers(data: SiteData): LedgerIssue[] {
  const out: LedgerIssue[] = [];
  const universe = numberUniverse(data);
  for (const c of claims.values()) {
    const local = new Set<string>();
    for (const s of data.claimSources[c.id] ?? []) for (const n of numbersIn(`${s.text ?? ''} ${s.quote ?? ''} ${s.file} ${s.lines}`)) local.add(n);
    for (const [part, text] of [['plain', c.plain], ['developer', c.developer], ['limit', c.limit ?? '']] as const)
      for (const n of numbersIn(text)) if (!same(n, local) && !same(n, universe)) out.push({ id: c.id, part, number: n, reason: '' });
  }
  return out;
}

export function auditClaims(data: SiteData): string[] {
  const errs: string[] = [];
  const universe = numberUniverse(data);
  for (const [k, v] of Object.entries(ui)) {
    if (/\d/.test(v.replace('SHA-256', ''))) errs.push(`ui.${k}: UI labels must not carry numbers`);
    for (const b of bannedIn(v)) errs.push(`ui.${k}: banned "${b}"`);
  }
  for (const c of claims.values()) {
    const srcs = data.claimSources[c.id];
    if (!srcs || srcs.length === 0) {
      errs.push(`${c.id}: no source`);
      continue;
    }
    const local = new Set<string>();
    for (const s of srcs) for (const n of numbersIn(`${s.text ?? ''} ${s.quote ?? ''} ${s.file} ${s.lines}`)) local.add(n);
    for (const [part, text] of [['plain', c.plain], ['developer', c.developer], ['limit', c.limit ?? ''], ['verbatim', c.verbatim ?? '']] as const) {
      for (const b of bannedIn(text)) errs.push(`${c.id}.${part}: banned "${b}"`);
      if (part === 'verbatim') continue;
      for (const n of numbersIn(text)) if (!same(n, local) && !same(n, universe) && !known(c.id, part, n)) errs.push(`${c.id}.${part}: number ${n} not in its sources or site-data.json`);
    }
  }
  const ck02 = claims.get('CK-02');
  if (ck02 && ck02.verbatim !== data.check.commands) errs.push('CK-02: commands differ from README main@1049d2a7 lines');
  for (const i of KNOWN_ISSUES) if (!unsupportedNumbers(data).some((u) => u.id === i.id && u.part === i.part && u.number === i.number)) errs.push(`ledger-issues.json: ${i.id} ${i.number} is resolved; remove it`);
  const allowed = ledger.claims.filter((c) => c.status === 'ALLOWED').length;
  if (claims.size !== allowed) errs.push('allowed claim count drift');
  return errs;
}
