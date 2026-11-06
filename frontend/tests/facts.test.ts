import { describe, it, expect } from 'vitest';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { claimsFor, timelineClaims, TRAINING_EVENTS } from '../src/content/claims';
import type { RawVerificationCheck } from '../src/data/contracts';
import { sha256Hex } from '../src/data/sha256';
import { checksForOdds, combine, datasetUrl, deriveCoverage, detectionOdds, summarize, verdict } from '../src/data/facts';
import { loadEvidence, withMismatch, withoutChecks } from './evidence-cases';

const check = (segment: number, checker: string, result: RawVerificationCheck['result']): RawVerificationCheck => ({
  segment,
  checker,
  result,
  code: 'c',
  driver: 'd',
  gpu: 'g',
});

describe('sha256Hex', () => {
  it('matches the published test vectors', () => {
    expect(sha256Hex('')).toBe('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855');
    expect(sha256Hex('abc')).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
    expect(sha256Hex('abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq')).toBe(
      '248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1',
    );
  });

  it('agrees with node:crypto across block boundaries and non-ASCII text', () => {
    for (let length = 0; length < 200; length++) {
      const text = 'é€'.repeat(length).slice(0, length);
      expect(sha256Hex(text)).toBe(createHash('sha256').update(text).digest('hex'));
    }
  });

  it('reproduces every published entry hash', () => {
    for (const entry of loadEvidence().chain) {
      expect(sha256Hex(entry.record)).toBe(entry.entryHash);
    }
  });
});

describe('verdict and coverage', () => {
  it('needs every check to match, and treats no checks as no result', () => {
    expect(verdict([{ result: 'MATCH' }, { result: 'MATCH' }])).toBe('MATCH');
    expect(verdict([{ result: 'MATCH' }, { result: 'MISMATCH' }])).toBe('MISMATCH');
    expect(verdict([])).toBeNull();
  });

  it('keeps project and independent results apart', () => {
    const categories = { us: 'project', them: 'independent' } as const;
    const coverage = deriveCoverage(
      [check(0, 'us', 'MATCH'), check(0, 'them', 'MISMATCH'), check(1, 'them', 'MATCH')],
      3,
      categories,
    );
    expect(coverage[0]).toMatchObject({ projectResult: 'MATCH', independentResult: 'MISMATCH' });
    expect(coverage[1]).toMatchObject({ projectResult: null, independentResult: 'MATCH' });
    expect(coverage[2]).toMatchObject({ projectResult: null, independentResult: null });
  });

  it('never lets a missing segment pass a combined result', () => {
    expect(combine(['MATCH', 'MATCH'])).toBe('MATCH');
    expect(combine(['MATCH', null])).toBeNull();
    expect(combine(['MISMATCH', null])).toBe('MISMATCH');
    expect(combine([])).toBeNull();
  });
});

describe('detection odds', () => {
  it('reproduces the published formula', () => {
    expect(checksForOdds(0.95, 37)).toBe(110);
    expect(checksForOdds(0.99, 37)).toBe(169);
    expect(detectionOdds(0, 37)).toBe(0);
    expect(detectionOdds(110, 37)).toBeGreaterThanOrEqual(0.95);
    expect(detectionOdds(109, 37)).toBeLessThan(0.95);
  });

  it('follows the segment count', () => {
    expect(checksForOdds(0.95, 10)).not.toBe(checksForOdds(0.95, 37));
  });
});

describe('summary of the real snapshot', () => {
  const facts = summarize(loadEvidence());

  it('counts from the records', () => {
    expect(facts).toMatchObject({
      segments: 37,
      boundaries: 38,
      projectMatched: 37,
      projectMismatched: 0,
      projectUnchecked: 0,
      independentChecked: 0,
      totalChecks: 46,
      projectChecks: 46,
      independentChecks: 0,
      forkFromEntry: 30,
      forkSegments: 8,
      forkSegmentsMatched: 8,
      modelMillions: 23,
      checkGpu: 'RTX 5090',
      stampDate: '20 October 2026',
      stampDateShort: '20 Oct 2026',
      authorities: ['DigiCert', 'FreeTSA'],
    });
  });

  it('links to the exact pinned revision', () => {
    expect(facts.evidenceUrl).toBe(datasetUrl(facts.dataset, '96dc282a'));
    expect(facts.evidenceUrl).toMatch(/\/tree\/96dc282a$/);
  });

  it('pins the download command to the same revision', () => {
    const command = claimsFor(facts).ladder.map(s => s.command ?? '').join('\n');
    expect(command).toContain(`--revision ${facts.revisionFull}`);
    expect(facts.revisionFull.startsWith(facts.revision)).toBe(true);
    expect(command).toContain(`--branch ${facts.releaseTag}`);
  });

  it('agrees with the static page description', () => {
    const html = readFileSync(resolve(process.cwd(), 'index.html'), 'utf8');
    expect(html).toContain(`each of its ${facts.segments} training segments`);
    expect(html).toContain(`A ${facts.modelMillions}M-parameter model`);
    expect(facts.independentChecks === 0).toBe(html.includes('no independent checks yet'));
  });
});

describe('facts that come from the snapshot', () => {
  it('moves the fork claim with the fork record', () => {
    const evidence = loadEvidence();
    expect(claimsFor(summarize(evidence)).forkWhy[0]).toBe('Entries 0 to 30 of its record are identical to the original.');
    evidence.exclusion.fork = { ...evidence.exclusion.fork, fromEntry: 12, entries: 20 };
    const facts = summarize(evidence);
    expect(claimsFor(facts).forkWhy[0]).toBe('Entries 0 to 12 of its record are identical to the original.');
    expect(facts.forkSegments).toBe(7);
  });

  it('counts a fork segment with no check as not matched', () => {
    const evidence = loadEvidence();
    evidence.exclusion.forkChecks = evidence.exclusion.forkChecks.filter(c => c.segment !== 33);
    const facts = summarize(evidence);
    expect(facts).toMatchObject({ forkSegments: 8, forkSegmentsMatched: 7 });
    expect(claimsFor(facts).forkWhy[2]).toContain('7 of the 8');
  });

  it('moves the model size and GPU with the evidence', () => {
    const evidence = loadEvidence();
    evidence.preparation.modelParametersMillions = 116;
    evidence.verifications.rawChecks.forEach(c => { c.gpu = 'NVIDIA GeForce RTX 4090'; });
    const claims = claimsFor(summarize(evidence));
    expect(claims.lead).toContain('116 million parameters');
    expect(claims.results.subtitle).toContain('second RTX 4090');
    expect(claims.ladder[2].meta).toContain('RTX 4090');
  });

  it('keeps the written training events in line with the evidence', () => {
    const evidence = loadEvidence();
    const second = evidence.verifications.rawChecks.filter(c => c.checker === 'project second pod');
    const firstNewCode = second.find(c => c.code !== second[0].code);
    expect(firstNewCode?.segment).toBe(TRAINING_EVENTS.codeChangeBoundary);
    const allSegments = evidence.hardwareMap.find(p => typeof p.segment === 'string');
    expect(allSegments?.note).toContain(`segments 0-${TRAINING_EVENTS.hostFailureSegment - 1}`);
  });
});

describe('claims never go beyond the data', () => {
  it('drops the all-matched claims when a segment mismatches', () => {
    const facts = summarize(withMismatch(5));
    expect(facts).toMatchObject({ projectMatched: 36, projectMismatched: 1, projectUnchecked: 0 });
    const c = claimsFor(facts);
    expect(c.results.projectSummary).toBe('Project: 36 / 37 matched');
    expect(c.results.projectExtra).toContain('1 mismatched');
    expect(c.results.notice).not.toContain('all matching');
    expect(c.results.noticeClose).not.toContain('Every segment matched');
    expect(c.results.shown[0]).not.toContain('came from the recorded run');
    expect(c.hero.checks).not.toContain('all matching');
    expect(c.results.subtitle).toContain('36 of 37 segments');
  });

  it('reports a missing check as not checked, not as a pass or a failure', () => {
    const facts = summarize(withoutChecks(7));
    expect(facts).toMatchObject({ projectMatched: 36, projectMismatched: 0, projectUnchecked: 1 });
    const c = claimsFor(facts);
    expect(c.results.projectExtra).toContain('1 not checked');
    expect(c.results.projectExtra).not.toContain('mismatched');
    expect(c.results.mapLabel).toContain('36 matched, 0 mismatched, 1 not checked');
  });

  it('only says the events still matched when their segments did', () => {
    const facts = summarize(loadEvidence());
    expect(timelineClaims(facts, { failure: 'MATCH', codeChange: 'MATCH' }).intro).toContain('still matched');
    expect(timelineClaims(facts, { failure: 'MISMATCH', codeChange: 'MATCH' }).intro).not.toContain('still matched');
    expect(timelineClaims(facts, { failure: 'MATCH', codeChange: null }).intro).not.toContain('still matched');
  });
});
