// Strict validator for OpenVerifiableLLM evidence data
import {
  type EvidenceData,
  KNOWN_CHECKER_MAP,
} from './contracts';

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    throw new SnapshotError(message);
  }
}

export function validateEvidence(data: unknown): EvidenceData {
  assert(typeof data === 'object' && data !== null, 'Evidence data must be a non-null object');

  const root = data as Record<string, unknown>;
  assert(root.schemaVersion === 1, `Unsupported schemaVersion: ${root.schemaVersion}`);
  assert(root.revision === '96dc282a', `Evidence revision must be 96dc282a, got ${root.revision}`);

  // 1. Preparation counts
  assert(typeof root.preparation === 'object' && root.preparation !== null, 'Missing preparation section');
  const prep = root.preparation as Record<string, unknown>;
  assert(prep.articles === 7232582, `Expected 7,232,582 articles, got ${prep.articles}`);
  assert(prep.wikipediaTargets === 7118121139, `Expected 7,118,121,139 targets, got ${prep.wikipediaTargets}`);
  assert(prep.conversations === 3481, `Expected 3,481 conversations, got ${prep.conversations}`);

  // 2. Chain boundaries
  assert(Array.isArray(root.chain), 'Chain must be an array');
  assert(root.chain.length === 38, `Chain must have exactly 38 entries, got ${root.chain.length}`);

  const chain = root.chain as Array<Record<string, unknown>>;
  for (let i = 0; i < chain.length; i++) {
    const entry = chain[i];
    assert(entry.index === i, `Chain entry index mismatch: expected ${i}, got ${entry.index}`);
    assert(typeof entry.state === 'string' && entry.state.length === 64, `Invalid state hash at boundary ${i}`);
    assert(typeof entry.model === 'string' && entry.model.length === 64, `Invalid model hash at boundary ${i}`);
    assert(typeof entry.entryHash === 'string' && entry.entryHash.length === 64, `Invalid entryHash at boundary ${i}`);

    if (i === 0) {
      assert(entry.previous === null, 'Boundary 0 must have null previous pointer');
    } else {
      const prevEntry = chain[i - 1];
      assert(
        entry.previous === prevEntry.entryHash,
        `Broken previous link at boundary ${i}: ${entry.previous} !== ${prevEntry.entryHash}`
      );
    }
  }

  // 3. Verifications and segment coverage
  assert(typeof root.verifications === 'object' && root.verifications !== null, 'Missing verifications section');
  const ver = root.verifications as Record<string, unknown>;
  assert(ver.uniqueSegments === 37, `Segment count must be exactly 37, got ${ver.uniqueSegments}`);
  assert(Array.isArray(ver.coverage), 'Verifications coverage must be an array');
  assert(ver.coverage.length === 37, `Coverage array must have exactly 37 segments, got ${ver.coverage.length}`);

  const coverage = ver.coverage as Array<Record<string, unknown>>;
  for (let s = 0; s < 37; s++) {
    const seg = coverage[s];
    assert(seg.segment === s, `Coverage segment order mismatch at index ${s}`);
    assert(seg.projectResult === 'MATCH', `Project segment ${s} must have MATCH result, got ${seg.projectResult}`);
  }

  // 4. Strict Checker validation
  assert(Array.isArray(ver.rawChecks), 'verifications.rawChecks must be an array');
  const rawChecks = ver.rawChecks as Array<Record<string, unknown>>;
  for (const check of rawChecks) {
    assert(typeof check.checker === 'string', 'Checker name must be a string');
    assert(
      Object.prototype.hasOwnProperty.call(KNOWN_CHECKER_MAP, check.checker),
      `Unknown or unauthorized checker string: "${check.checker}"`
    );
    const category = KNOWN_CHECKER_MAP[check.checker];
    assert(
      category === 'project' || category === 'independent',
      `Checker category outside {project, independent}: ${category}`
    );
    assert(
      check.result === 'MATCH' || check.result === 'MISMATCH',
      `Invalid verification result: ${check.result}`
    );
  }

  // 5. Source Hashes
  assert(typeof root.sourceHashes === 'object' && root.sourceHashes !== null, 'Missing sourceHashes');
  const hashes = root.sourceHashes as Record<string, unknown>;
  const hex64Regex = /^[0-9a-f]{64}$/i;
  for (const [file, hash] of Object.entries(hashes)) {
    assert(
      typeof hash === 'string' && hex64Regex.test(hash),
      `Invalid or corrupted SHA-256 hash for ${file}: ${hash}`
    );
  }

  // 6. Exclusion validation
  assert(typeof root.exclusion === 'object' && root.exclusion !== null, 'Missing exclusion section');
  const excl = root.exclusion as Record<string, unknown>;
  assert(excl.removedArticle === '3I/ATLAS', `Exclusion removedArticle mismatch: ${excl.removedArticle}`);
  assert(excl.originalRemovedLoss === 3.3604, `originalRemovedLoss mismatch: ${excl.originalRemovedLoss}`);
  assert(excl.forkRemovedLoss === 3.386, `forkRemovedLoss mismatch: ${excl.forkRemovedLoss}`);
  assert(excl.controlArticlesCount === 50, `controlArticlesCount mismatch: ${excl.controlArticlesCount}`);

  return data as EvidenceData;
}
