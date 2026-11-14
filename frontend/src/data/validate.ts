import type { CheckRow, SiteData, Src, Tile } from './types';

export class DataError extends Error {
  override name = 'DataError';
}
export const SECTIONS = ['chain', 'verifications', 'odds', 'events', 'hardware', 'downloads', 'anchor', 'exclusion', 'preparation', 'check', 'model'] as const;
export type SectionId = (typeof SECTIONS)[number];
export type SectionErrors = Partial<Record<SectionId, string>>;
export interface Validated { data: SiteData; errors: SectionErrors }
export type Sha256 = (text: string) => string;

export const PINNED_REVISION = /^96dc282a[0-9a-f]{32}$/;
export const UPSTREAM_COMMIT = '1049d2a782b8f63103abfa0cca8deab0d2037290';
const HEX64 = /^[0-9a-f]{64}$/;
const RESULTS = new Set(['MATCH', 'MISMATCH']);

function must(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new DataError(msg);
}
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const close = (a: number, b: number) => Math.abs(a - b) < 1e-9;

function checkSrc(s: unknown, where: string, needRaw = false): void {
  must(isObj(s), `${where}: missing source`);
  must(typeof s.file === 'string' && s.file.length > 0, `${where}: source without file`);
  must(typeof s.url === 'string' && s.url.startsWith('https://'), `${where}: source URL must be https`);
  must(s.rev === '96dc282a' || s.rev === UPSTREAM_COMMIT.slice(0, 8), `${where}: source not at a pinned revision`);
  if (needRaw) must(typeof s.raw === 'string' && s.raw.length > 0 && typeof s.line === 'number', `${where}: row without source line`);
}

const whoOf = (checker: string) => (checker.startsWith('project') ? 'project' : 'independent');

/** Smallest M with 1-((N-1)/N)^M >= p. Recomputed here so a typed constant can't drift. */
export function minChecks(N: number, p: number): number {
  let m = 0;
  while (1 - ((N - 1) / N) ** m < p) m += 1;
  return m;
}
export const detect = (N: number, m: number) => 1 - ((N - 1) / N) ** m;

function checkRows(rows: unknown, where: string, N: number): asserts rows is CheckRow[] {
  must(Array.isArray(rows) && rows.length > 0, `${where}: no rows`);
  rows.forEach((r: unknown, i) => {
    must(isObj(r), `${where}[${i}] not an object`);
    must(RESULTS.has(r.result as string), `${where}[${i}]: unknown result ${String(r.result)}`);
    must(typeof r.checker === 'string', `${where}[${i}]: no checker`);
    must(r.checkedBy === whoOf(r.checker), `${where}[${i}]: checkedBy does not follow the checker`);
    must(Number.isInteger(r.segment) && (r.segment as number) >= 0 && (r.segment as number) <= N, `${where}[${i}]: segment out of range`);
    checkSrc(r.src, `${where}[${i}]`, true);
  });
}

const sections: Record<SectionId, (d: SiteData, sha?: Sha256) => void> = {
  model(d) {
    must(Number.isInteger(d.model?.millionParameters) && d.model.millionParameters > 0, 'model size');
    checkSrc(d.model.src, 'model', true);
  },
  chain(d, sha) {
    const c = d.chain, B = c?.boundaries;
    must(Array.isArray(B) && B.length === c.entries && c.segments === B.length - 1, 'segment count != entries-1');
    B.forEach((b, i) => {
      must(b.index === i, `chain index gap at ${i}`);
      checkSrc(b.src, `chain[${i}]`, true);
      if (sha) must(i === 0 ? b.previous === null : b.previous === sha(B[i - 1]!.src.raw ?? ''), `chain link broken at ${i}`);
    });
    must(Array.isArray(c.phases) && c.phases.length > 0, 'chain phases');
    const covered = c.phases.reduce((n, p) => n + p.last - p.first + 1, 0);
    must(covered === c.segments, 'phases do not cover every segment');
  },
  verifications(d) {
    const v = d.verifications, N = d.chain.segments;
    checkRows(v?.rows, 'verifications', N - 1);
    must(v.rowCount === v.rows.length, `row count ${v.rowCount} != ${v.rows.length} rows`);
    const checkers = [...new Set(v.rows.map((r) => r.checker))];
    must(v.machines === checkers.length && v.groups.length === checkers.length, 'machine count drift');
    for (const g of v.groups) {
      const mine = v.rows.filter((r) => r.checker === g.checker);
      must(g.rows === mine.length, `group ${g.checker} row count drift`);
      must(g.checkedBy === whoOf(g.checker), `group ${g.checker} checkedBy drift`);
      must(JSON.stringify([...g.segments].sort((a, b) => a - b)) === JSON.stringify([...new Set(mine.map((r) => r.segment))].sort((a, b) => a - b)), `group ${g.checker} segments drift`);
    }
    must(v.groups.reduce((n, g) => n + g.rows, 0) === v.rowCount, 'groups do not add up to all rows');
    must(Array.isArray(v.tiles) && v.tiles.length === N, `tile count ${v.tiles?.length} != ${N} segments`);
    v.tiles.forEach((t: Tile, k) => {
      must(t.segment === k, `tile ${k} out of order`);
      const idx = v.rows.flatMap((r, i) => (r.segment === k ? [i] : []));
      must(JSON.stringify(idx) === JSON.stringify(t.checks), `tile ${k} checks drift`);
      const res = idx.map((i) => v.rows[i]!.result);
      const want = res.length === 0 ? 'NOT_RUN' : res.every((x) => x === 'MATCH') ? 'MATCH' : 'MISMATCH';
      must(t.result === want, `tile ${k} result drift`);
    });
    must(v.segmentsMatched === v.tiles.filter((t) => t.result === 'MATCH').length, 'segmentsMatched drift');
    must(v.secondMachineMatched === new Set(v.rows.filter((r) => r.checker === v.secondMachine && r.result === 'MATCH').map((r) => r.segment)).size, 'secondMachineMatched drift');
    must(v.independentChecks === v.rows.filter((r) => r.checkedBy === 'independent').length, 'independent count drift');
    must(v.projectChecks === v.rows.filter((r) => r.checkedBy === 'project').length, 'project count drift');
    must(v.allMatch === v.rows.every((r) => r.result === 'MATCH'), 'allMatch drift');
    must(JSON.stringify(v.freshSegments) === JSON.stringify(v.rows.filter((r) => r.checker === v.freshMachine).map((r) => r.segment)), 'fresh-machine segments drift');
  },
  odds(d) {
    const o = d.odds, N = d.chain.segments;
    must(o.N === N && o.Nm1 === N - 1, 'odds N drift');
    must(o.formula === `1 − (${N - 1}/${N})^M`, 'odds formula drift');
    must(o.m95 === minChecks(N, 0.95) && o.m99 === minChecks(N, 0.99), 'odds M is not the minimal value');
    must(close(o.p95, detect(N, o.m95)) && close(o.p99, detect(N, o.m99)) && close(o.pAtN, detect(N, N)), 'odds probability drift');
    must(o.currentM === d.verifications.independentChecks && close(o.currentP, detect(N, o.currentM)), 'odds M != independent check count');
    checkSrc(o.src, 'odds', true);
  },
  events(d) {
    must(Array.isArray(d.events) && d.events.length > 0, 'no events');
    for (const e of d.events) {
      checkSrc(e.src, `event ${e.id}`, true);
      const tile = d.verifications.tiles[e.segment];
      must(tile && tile.events.includes(e.id), `event ${e.id} not marked on tile ${e.segment}`);
    }
    for (const t of d.verifications.tiles) for (const id of t.events) must(d.events.some((e) => e.id === id && e.segment === t.segment), `tile ${t.segment} marks unknown event ${id}`);
    const cc = d.events.find((e) => e.id === 'code-change');
    if (cc) {
      const second = d.verifications.rows.filter((r) => r.checker === d.verifications.secondMachine).sort((a, b) => a.segment - b.segment);
      const at = second.findIndex((r, i) => i > 0 && r.code !== second[i - 1]!.code);
      must(at > 0 && second[at]!.segment === cc.segment, 'code-change segment does not follow the data');
    }
  },
  hardware(d) {
    must(Array.isArray(d.hardware) && d.hardware.length > 0, 'no hardware rows');
    d.hardware.forEach((h, i) => {
      must(RESULTS.has(h.result), `hardware[${i}] result`);
      must(typeof h.hardware === 'string' && h.hardware.length > 0, `hardware[${i}] name`);
      checkSrc(h.src, `hardware[${i}]`, true);
    });
  },
  downloads(d) {
    must(Array.isArray(d.downloads) && d.downloads.length > 0, 'no downloads');
    for (const r of d.downloads) {
      must(Array.isArray(r.files) && r.files.length > 0, `${r.release}: no files`);
      for (const f of r.files) {
        must(HEX64.test(f.sha256), `bad sha256 for ${f.file}`);
        must(f.download.startsWith('https://huggingface.co/') && f.download.includes(d.revision), `${f.file}: download not pinned`);
        checkSrc(f.src, f.file, true);
      }
    }
  },
  anchor(d, sha) {
    const a = d.anchor;
    must(HEX64.test(a.sha256) && HEX64.test(a.chainHead), 'anchor hashes');
    must(a.entries === d.chain.entries, 'anchor entry count mismatch');
    if (sha) must(sha(d.chain.boundaries.at(-1)?.src.raw ?? '') === a.chainHead, 'chain head != anchor.json');
    must(a.tsa.length > 0 && a.tsa.every((t) => !Number.isNaN(Date.parse(t.genTime))), 'timestamp tokens');
  },
  exclusion(d) {
    const x = d.exclusion;
    checkRows(x.rows, 'exclusion', x.entries);
    must(x.rowCount === x.rows.length, 'exclusion row count drift');
    must(x.allMatch === x.rows.every((r) => r.result === 'MATCH'), 'exclusion allMatch drift');
    must(x.groups.reduce((n, g) => n + g.rows, 0) === x.rowCount, 'exclusion groups');
    must(x.sharedLast === x.sharedFirst + x.sharedPrefix - 1, 'shared prefix drift');
    for (const l of [x.removedLoss, x.controlLoss]) must(close(l.delta, Math.round((l.after - l.before) * 1e4) / 1e4), 'loss delta drift');
    checkSrc(x.titleSrc, 'exclusion title', true);
  },
  preparation(d) {
    const p = d.preparation;
    must(HEX64.test(p.sha256), 'preparation sha');
    must(p.src.file.includes(p.sha256), 'preparation.json sha256 != its folder id');
    for (const k of ['articles', 'wikipediaTargets', 'conversations'] as const) must(Number.isInteger(p[k]) && p[k] > 0, `preparation.${k}`);
  },
  check(d) {
    const c = d.check;
    must(typeof c.commands === 'string' && c.commands.includes('--branch v1.0.1'), 'commands must clone --branch v1.0.1 (README main@1049d2a7)');
    for (const s of [c.introSrc, c.commandsSrc, c.costSrc, c.runpodSrc] as Src[]) {
      checkSrc(s, 'check', true);
      must(s.url.includes(UPSTREAM_COMMIT), 'check sources must be at 1049d2a7');
    }
    must(c.commandsSrc.raw === c.commands, 'commands are not verbatim');
    must(c.costSrc.file === 'docs/design/simple_run.md' && c.costSrc.line === 85 && (c.costSrc.raw ?? '').includes(c.cost), 'cost is not cited from simple_run.md:85');
  },
};

function walkUrls(o: unknown): void {
  if (Array.isArray(o)) return o.forEach(walkUrls);
  if (!isObj(o)) return;
  for (const [k, v] of Object.entries(o)) {
    if ((k === 'url' || k === 'download' || k === 'runpod' || k === 'hfTree') && typeof v === 'string') must(v.startsWith('https://'), `disallowed URL scheme ${v}`);
    walkUrls(v);
  }
}

/** Fatal problems throw; a broken section is reported so only that section shows the fallback. */
export function validateSite(input: unknown, sha?: Sha256): Validated {
  must(isObj(input), 'data is not an object');
  must(input.schema === 'ovl-site-data.v2', 'unknown schema');
  must(typeof input.revision === 'string' && PINNED_REVISION.test(input.revision), 'revision is not the pinned 96dc282a');
  walkUrls(input);
  const data = input as unknown as SiteData;
  const errors: SectionErrors = {};
  for (const id of SECTIONS) {
    try {
      sections[id](data, sha);
    } catch (e) {
      errors[id] = e instanceof Error ? e.message : String(e);
    }
  }
  return { data, errors };
}

/** Build-time: any problem at all fails. */
export function validateStrict(input: unknown, sha?: Sha256): SiteData {
  const { data, errors } = validateSite(input, sha);
  const first = Object.entries(errors)[0];
  if (first) throw new DataError(`${first[0]}: ${first[1]}`);
  return data;
}
