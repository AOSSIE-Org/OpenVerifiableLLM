import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { validateEvidence, SnapshotError } from '../src/data/validate';

describe('Evidence Validator', () => {
  const getFixturePath = (filename: string) =>
    resolve(process.cwd(), 'tests/fixtures', filename);

  const validData = JSON.parse(
    readFileSync(resolve(process.cwd(), 'public/data/evidence.json'), 'utf8')
  );

  it('validates official pinned evidence.json successfully', () => {
    const validated = validateEvidence(validData);
    expect(validated.revision).toBe('96dc282a');
    expect(validated.chain.length).toBe(38);
    expect(validated.verifications.uniqueSegments).toBe(37);
    expect(validated.verifications.totalChecks).toBe(46);
    expect(validated.preparation.articles).toBeGreaterThan(0);
    expect(validated.preparation.wikipediaTargets).toBeGreaterThan(0);
    expect(validated.preparation.conversations).toBeGreaterThan(0);
    expect(validated.exclusion.removedArticle).toBe('3I/ATLAS');
    expect(validated.exclusion.forkRemovedLoss).toBeGreaterThan(validated.exclusion.originalRemovedLoss);
  });

  it('fails if chain length is less than 2 boundaries', () => {
    const bad = JSON.parse(JSON.stringify(validData));
    bad.chain = [bad.chain[0]];
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Chain must have at least 2 entries/);
  });

  it('fails if unique segment count does not match derived count from chain.jsonl', () => {
    const bad = JSON.parse(
      readFileSync(getFixturePath('invalid-segment-count.json'), 'utf8')
    );
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Segment count mismatch/);
  });

  it('fails if previous link in chain is broken', () => {
    const bad = JSON.parse(
      readFileSync(getFixturePath('broken-link.json'), 'utf8')
    );
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Broken previous link at boundary 10/);
  });

  it('fails if any row in verifications points at an invalid segment', () => {
    const bad = JSON.parse(JSON.stringify(validData));
    bad.verifications.rawChecks[0].segment = 999;
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/points at invalid segment 999/);
  });

  it('fails if checker string is unknown / unauthorized', () => {
    const bad = JSON.parse(
      readFileSync(getFixturePath('invalid-checker.json'), 'utf8')
    );
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Unknown or unauthorized checker string/);
  });

  it('fails if a source SHA-256 hash is corrupted or invalid', () => {
    const bad = JSON.parse(
      readFileSync(getFixturePath('hash-mismatch.json'), 'utf8')
    );
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Invalid or corrupted SHA-256 hash/);
  });

  it('fails if verifiable exclusion loss delta is mathematically inconsistent', () => {
    const bad = JSON.parse(JSON.stringify(validData));
    bad.exclusion.removedLossDelta = 9.9999;
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Exclusion removed loss delta inconsistent/);
  });

  it('fails on null or non-object payloads', () => {
    expect(() => validateEvidence(null)).toThrow(SnapshotError);
    expect(() => validateEvidence('not-an-object')).toThrow(SnapshotError);
  });
});
