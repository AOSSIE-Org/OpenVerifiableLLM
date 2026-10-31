import { describe, expect, it } from 'vitest';
import { execFileSync, spawnSync } from 'node:child_process';
import { cpSync, mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fresh } from './helpers';

const root = path.resolve(__dirname, '../..');

function python(): string {
  for (const bin of [process.env.PYTHON, 'python3', 'python']) {
    if (!bin) continue;
    const r = spawnSync(bin, ['--version'], { encoding: 'utf8' });
    if (r.status === 0 && /Python 3/.test(`${r.stdout}${r.stderr}`)) return bin;
  }
  throw new Error('Python 3 is required (python3 on Linux CI; python on Windows)');
}

describe('scripts/build-data.py', () => {
  it('reproduces the committed site-data.json byte for byte from the pinned files (--check)', () => {
    const out = execFileSync(python(), ['scripts/build-data.py', '--check'], { cwd: root, encoding: 'utf8' });
    expect(out).toContain('matches the pinned files');
  });

  it('fails when an upstream GitHub copy no longer matches its manifest sha256', () => {
    const tmp = mkdtempSync(path.join(tmpdir(), 'ovl-'));
    for (const d of ['scripts', 'data', 'public', 'src/content']) cpSync(path.join(root, d), path.join(tmp, d), { recursive: true });
    const f = path.join(tmp, 'data/upstream/1049d2a7/README.md');
    writeFileSync(f, readFileSync(f, 'utf8') + '\n');
    const r = spawnSync(python(), ['scripts/build-data.py'], { cwd: tmp, encoding: 'utf8' });
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/upstream copy README\.md changed/);
  });

  it('fails when a pinned raw file no longer matches its manifest sha256', () => {
    const tmp = mkdtempSync(path.join(tmpdir(), 'ovl-'));
    for (const d of ['scripts', 'data', 'public', 'src/content']) cpSync(path.join(root, d), path.join(tmp, d), { recursive: true });
    const f = path.join(tmp, 'data/raw/release/simple-run/verifications.jsonl');
    writeFileSync(f, readFileSync(f, 'utf8').replace('"MATCH", "segment": 3}', '"MISMATCH", "segment": 3}'));
    const r = spawnSync(python(), ['scripts/build-data.py'], { cwd: tmp, encoding: 'utf8' });
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/does not match manifest sha256/);
  });

  it('fails when a ledger quote is not on its cited lines', () => {
    const tmp = mkdtempSync(path.join(tmpdir(), 'ovl-'));
    for (const d of ['scripts', 'data', 'public', 'src/content']) cpSync(path.join(root, d), path.join(tmp, d), { recursive: true });
    const p = path.join(tmp, 'src/content/claims.json');
    const l = JSON.parse(readFileSync(p, 'utf8'));
    l.claims.find((c: { id: string }) => c.id === 'OV-01').sources[0].quote = 'A 99-million-parameter GPT';
    writeFileSync(p, JSON.stringify(l));
    const r = spawnSync(python(), ['scripts/build-data.py'], { cwd: tmp, encoding: 'utf8' });
    expect(r.status).not.toBe(0);
    expect(r.stderr).toMatch(/quote not found/);
  });

  it('withholds (links but does not quote) cited lines that use a banned phrase', () => {
    const d = fresh();
    const withheld = Object.values(d.claimSources).flat().filter((s) => 'withheld' in s);
    expect(withheld.length).toBeGreaterThan(0);
    for (const s of withheld) expect(s.text).toBeNull();
    expect(d.events[0]!.medianGapSeconds).toBe(1968);
    expect(d.events[0]!.gapSeconds).toBe(3297);
  });

  it('derives the audited counts from the data', () => {
    const d = fresh();
    expect(d.revision).toBe('96dc282a9803616c6b30bb31bfc3a0b91a7191e6');
    expect(d.chain.entries).toBe(38);
    expect(d.chain.segments).toBe(37);
    expect(d.verifications.rowCount).toBe(46);
    expect(d.verifications.groups.map((g) => [g.checker, g.rows])).toEqual([
      ['project second pod', 37],
      ['project training pod', 8],
      ['project fresh machine public files only', 1],
    ]);
    expect(d.verifications.machines).toBe(3);
    expect(d.verifications.secondMachineMatched).toBe(37);
    expect(d.verifications.independentChecks).toBe(0);
    expect(d.verifications.freshSegments).toEqual([21]);
    expect([d.odds.m95, d.odds.m99]).toEqual([110, 169]);
    expect(d.events.map((e) => [e.id, e.segment])).toEqual([['restart', 10], ['code-change', 19]]);
    expect(d.exclusion.rowCount).toBe(10);
    expect(d.exclusion.sharedPrefix).toBe(31);
    expect(d.preparation.articles).toBe(7232582);
    expect(d.anchor.tsa.map((t) => t.genTime)).toEqual(['2026-10-20T14:38:43Z', '2026-10-20T14:38:43Z']);
  });

  it('marks tiles 10 and 19 from the events, and no others', () => {
    const marked = fresh().verifications.tiles.filter((t) => t.events.length).map((t) => [t.segment, t.events]);
    expect(marked).toEqual([[10, ['restart']], [19, ['code-change']]]);
  });

  it('copies the Getting started commands verbatim from README main@1049d2a7 and cites the cost at simple_run.md:85', () => {
    const d = fresh();
    const readme = readFileSync(path.join(root, 'data/upstream/1049d2a7/README.md'), 'utf8').split('\n');
    expect(d.check.commands).toBe(readme.slice(155, 167).join('\n'));
    expect(d.check.commands).toContain('--branch v1.0.1');
    expect(d.check.commands).not.toContain('v1.0.0');
    expect(d.check.cost).toBe('about US$1–1.50');
    expect(d.check.costSrc.line).toBe(85);
  });

  it('has the exact cited lines for every allowed ledger claim', () => {
    const d = fresh();
    expect(Object.keys(d.claimSources)).toHaveLength(71);
    for (const srcs of Object.values(d.claimSources)) for (const s of srcs) if ((s.origin === 'hf' || s.origin === 'github') && !('withheld' in s)) expect(s.text).toBeTruthy();
  });
});
