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
    expect(validated.preparation.articles).toBe(7232582);
    expect(validated.preparation.wikipediaTargets).toBe(7118121139);
    expect(validated.preparation.conversations).toBe(3481);
    expect(validated.exclusion.removedArticle).toBe('3I/ATLAS');
    expect(validated.exclusion.originalRemovedLoss).toBe(3.3604);
    expect(validated.exclusion.forkRemovedLoss).toBe(3.386);
  });

  it('fails if chain length is not exactly 38 entries', () => {
    const bad = JSON.parse(
      readFileSync(getFixturePath('invalid-chain-len.json'), 'utf8')
    );
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Chain must have exactly 38 entries/);
  });

  it('fails if previous link in chain is broken', () => {
    const bad = JSON.parse(
      readFileSync(getFixturePath('broken-link.json'), 'utf8')
    );
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Broken previous link at boundary 10/);
  });

  it('fails if unique segment count is not 37', () => {
    const bad = JSON.parse(
      readFileSync(getFixturePath('invalid-segment-count.json'), 'utf8')
    );
    expect(() => validateEvidence(bad)).toThrow(SnapshotError);
    expect(() => validateEvidence(bad)).toThrow(/Segment count must be exactly 37/);
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

  it('fails on null or non-object payloads', () => {
    expect(() => validateEvidence(null)).toThrow(SnapshotError);
    expect(() => validateEvidence('not-an-object')).toThrow(SnapshotError);
  });
});
