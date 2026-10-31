// Structural and internal consistency validator for OpenVerifiableLLM evidence data
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

const HEX_64_REGEX = /^[0-9a-f]{64}$/i;

export function validateEvidence(data: unknown): EvidenceData {
  assert(typeof data === 'object' && data !== null, 'Evidence data must be a non-null object');

  const root = data as Record<string, unknown>;
  assert(
    typeof root.schemaVersion === 'number' && Number.isInteger(root.schemaVersion) && root.schemaVersion >= 1,
    `Invalid schemaVersion: ${root.schemaVersion}`
  );
  assert(
    typeof root.revision === 'string' && /^[0-9a-f]{7,40}$/i.test(root.revision),
    `Invalid evidence revision format: ${root.revision}`
  );

  // 1. Preparation counts & types (internal consistency, no hand-typed numbers)
  assert(typeof root.preparation === 'object' && root.preparation !== null, 'Missing preparation section');
  const prep = root.preparation as Record<string, unknown>;
  assert(
    typeof prep.articles === 'number' && Number.isInteger(prep.articles) && prep.articles > 0,
    `articles count must be a positive integer, got ${prep.articles}`
  );
  assert(
    typeof prep.wikipediaTargets === 'number' && Number.isInteger(prep.wikipediaTargets) && prep.wikipediaTargets > 0,
    `wikipediaTargets count must be a positive integer, got ${prep.wikipediaTargets}`
  );
  assert(
    typeof prep.conversations === 'number' && Number.isInteger(prep.conversations) && prep.conversations > 0,
    `conversations count must be a positive integer, got ${prep.conversations}`
  );
  assert(
    typeof prep.preparedDataId === 'string' && HEX_64_REGEX.test(prep.preparedDataId),
    `Invalid preparedDataId SHA-256 hash: ${prep.preparedDataId}`
  );

  // 2. Chain boundaries: cryptographic continuity and segment count derivation
  assert(Array.isArray(root.chain), 'Chain must be an array');
  assert(
    root.chain.length >= 2,
    `Chain must have at least 2 entries to define training segments, got ${root.chain.length}`
  );

  const chain = root.chain as Array<Record<string, unknown>>;
  for (let i = 0; i < chain.length; i++) {
    const entry = chain[i];
    assert(entry.index === i, `Chain entry index mismatch: expected ${i}, got ${entry.index}`);
    assert(typeof entry.state === 'string' && HEX_64_REGEX.test(entry.state), `Invalid state hash at boundary ${i}`);
    assert(typeof entry.model === 'string' && HEX_64_REGEX.test(entry.model), `Invalid model hash at boundary ${i}`);
    assert(typeof entry.entryHash === 'string' && HEX_64_REGEX.test(entry.entryHash), `Invalid entryHash at boundary ${i}`);

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

  // Number of segments is derived directly from chain boundaries (N boundaries => N - 1 segments)
  const segmentCount = chain.length - 1;

  // 3. Verifications and segment coverage
  assert(typeof root.verifications === 'object' && root.verifications !== null, 'Missing verifications section');
  const ver = root.verifications as Record<string, unknown>;
  assert(
    ver.uniqueSegments === segmentCount,
    `Segment count mismatch: chain has ${chain.length} boundaries (${segmentCount} segments), but verifications reported ${ver.uniqueSegments}`
  );
  assert(Array.isArray(ver.coverage), 'Verifications coverage must be an array');
  assert(
    ver.coverage.length === segmentCount,
    `Coverage array length (${ver.coverage.length}) does not match derived segment count (${segmentCount})`
  );

  const coverage = ver.coverage as Array<Record<string, unknown>>;
  for (let s = 0; s < segmentCount; s++) {
    const seg = coverage[s];
    assert(seg.segment === s, `Coverage segment order mismatch at index ${s}`);
    assert(
      seg.projectResult === 'MATCH' || seg.projectResult === 'MISMATCH',
      `Invalid projectResult for segment ${s}: ${seg.projectResult}`
    );
  }

  // 4. Verification checks: every row in verifications.jsonl points at a real segment
  assert(Array.isArray(ver.rawChecks), 'verifications.rawChecks must be an array');
  const rawChecks = ver.rawChecks as Array<Record<string, unknown>>;
  assert(rawChecks.length >= segmentCount, `Expected at least ${segmentCount} checks, got ${rawChecks.length}`);

  for (let r = 0; r < rawChecks.length; r++) {
    const check = rawChecks[r];
    assert(
      typeof check.segment === 'number' && Number.isInteger(check.segment),
      `Check #${r} segment must be an integer, got ${check.segment}`
    );
    assert(
      check.segment >= 0 && check.segment < segmentCount,
      `Check #${r} points at invalid segment ${check.segment}; must be in [0, ${segmentCount - 1}]`
    );
    assert(typeof check.checker === 'string', `Check #${r} checker name must be a string`);
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

  // 5. Source Hashes & Release Hashes validation
  assert(typeof root.sourceHashes === 'object' && root.sourceHashes !== null, 'Missing sourceHashes');
  const sourceHashes = root.sourceHashes as Record<string, unknown>;
  for (const [file, hash] of Object.entries(sourceHashes)) {
    assert(
      typeof hash === 'string' && HEX_64_REGEX.test(hash),
      `Invalid or corrupted SHA-256 hash for ${file}: ${hash}`
    );
  }

  assert(typeof root.releaseHashes === 'object' && root.releaseHashes !== null, 'Missing releaseHashes');
  const releaseHashes = root.releaseHashes as Record<string, unknown>;
  assert(Object.keys(releaseHashes).length > 0, 'releaseHashes must not be empty');
  for (const [file, hash] of Object.entries(releaseHashes)) {
    assert(
      typeof hash === 'string' && HEX_64_REGEX.test(hash),
      `Invalid SHA-256 hash in releaseHashes for ${file}: ${hash}`
    );
  }

  // 6. Exclusion validation (internal loss delta consistency)
  assert(typeof root.exclusion === 'object' && root.exclusion !== null, 'Missing exclusion section');
  const excl = root.exclusion as Record<string, unknown>;
  assert(typeof excl.removedArticle === 'string' && excl.removedArticle.length > 0, 'Missing removedArticle');
  assert(typeof excl.originalRemovedLoss === 'number' && excl.originalRemovedLoss > 0, 'Invalid originalRemovedLoss');
  assert(typeof excl.forkRemovedLoss === 'number' && excl.forkRemovedLoss > 0, 'Invalid forkRemovedLoss');
  assert(typeof excl.removedLossDelta === 'number', 'Invalid removedLossDelta');
  assert(
    Math.abs(excl.removedLossDelta - (excl.forkRemovedLoss - excl.originalRemovedLoss)) < 0.001,
    `Exclusion removed loss delta inconsistent: ${excl.removedLossDelta} !== ${excl.forkRemovedLoss} - ${excl.originalRemovedLoss}`
  );

  assert(
    typeof excl.controlArticlesCount === 'number' && Number.isInteger(excl.controlArticlesCount) && excl.controlArticlesCount > 0,
    `controlArticlesCount must be a positive integer, got ${excl.controlArticlesCount}`
  );
  assert(typeof excl.originalControlLossMean === 'number' && excl.originalControlLossMean > 0, 'Invalid originalControlLossMean');
  assert(typeof excl.forkControlLossMean === 'number' && excl.forkControlLossMean > 0, 'Invalid forkControlLossMean');
  assert(typeof excl.controlLossDelta === 'number', 'Invalid controlLossDelta');
  assert(
    Math.abs(excl.controlLossDelta - (excl.forkControlLossMean - excl.originalControlLossMean)) < 0.001,
    `Exclusion control loss delta inconsistent: ${excl.controlLossDelta} !== ${excl.forkControlLossMean} - ${excl.originalControlLossMean}`
  );
  assert(excl.evidencePath === 'release/unlearning-3i-atlas/', `Invalid exclusion evidencePath: ${excl.evidencePath}`);

  return data as EvidenceData;
}
