// Checks the evidence snapshot against itself, so it fails closed. Nothing here is
// compared with a number typed by hand: the chain gives the segment count, the raw
// check rows give the results and counts, and the anchor ties the files together.
//
// This is a consistency check, not verification. It recomputes the chain's hashes, so
// editing one field breaks them. It can't tell who wrote the snapshot: someone rewriting
// every hash, the anchor included, would pass. Only the timestamp signatures can catch
// that, and `ovl.py verify` checks those, not this file.
import { type EvidenceData, KNOWN_CHECKER_MAP, type RawVerificationCheck } from './contracts';
import { deriveCoverage } from './facts';
import { sha256Hex } from './sha256';

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

type Fields = Record<string, unknown>;

const HEX_64_REGEX = /^[0-9a-f]{64}$/i;
const CHAIN_KINDS = new Set(['initial', 'progress', 'base', 'transition', 'epoch', 'final']);
const RUN_CHAIN = 'release/simple-run/chain.jsonl';
const RUN_HASHES = 'release/simple-run/release-hashes.txt';
// These go into download links and into commands readers copy and run, so only the
// exact expected shape gets through: no spaces, newlines or shell characters.
const DATASET_ID_REGEX = /^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?\/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$/;
const RELEASE_TAG_REGEX = /^v\d+\.\d+\.\d+$/;
const DOWNLOADED_FILES = ['ovl-23m-chat.gguf', 'Modelfile', 'conversation-model.pt'];

const isFields = (value: unknown): value is Fields =>
  typeof value === 'object' && value !== null && !Array.isArray(value);
const isHash = (value: unknown): value is string => typeof value === 'string' && HEX_64_REGEX.test(value);
const isText = (value: unknown): value is string => typeof value === 'string' && value.length > 0;
const isCount = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0;
const isPositive = (value: unknown): value is number => isCount(value) && value > 0;
const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const isDate = (value: unknown): value is string => isText(value) && !Number.isNaN(Date.parse(value));

function fields(value: unknown, name: string): Fields {
  assert(isFields(value), `Missing ${name} section`);
  return value;
}

// Same value, whatever order the keys are in
function sameJson(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((item, i) => sameJson(item, b[i]));
  }
  if (isFields(a) && isFields(b)) {
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every(key => Object.hasOwn(b, key) && sameJson(a[key], b[key]));
  }
  return a === b;
}

// The fields the page reads must be the ones in the record the hash covers.
const RECORD_FIELDS = ['index', 'kind', 'control', 'state', 'model', 'previous', 'time'];

function validateRecord(entry: Fields, i: number): void {
  assert(isText(entry.record), `Missing record at boundary ${i}`);
  assert(sha256Hex(entry.record) === entry.entryHash, `Entry hash at boundary ${i} does not match its record`);
  let parsed: unknown;
  try {
    parsed = JSON.parse(entry.record);
  } catch {
    throw new SnapshotError(`Record at boundary ${i} is not valid JSON`);
  }
  const record = fields(parsed, `parsed record at boundary ${i}`);
  for (const field of RECORD_FIELDS) {
    assert(sameJson(entry[field], record[field]), `${field} at boundary ${i} does not match its record`);
  }
}

function validateChain(root: Fields): Fields[] {
  assert(Array.isArray(root.chain), 'Chain must be an array');
  assert(
    root.chain.length >= 2,
    `Chain must have at least 2 entries to define training segments, got ${root.chain.length}`,
  );
  const chain: Fields[] = root.chain.map((entry: unknown, i: number) => fields(entry, `chain entry ${i}`));
  let lastTime = 0;
  let lastStep = 0;

  chain.forEach((entry, i) => {
    assert(entry.index === i, `Chain entry index mismatch: expected ${i}, got ${entry.index}`);
    assert(typeof entry.kind === 'string' && CHAIN_KINDS.has(entry.kind), `Invalid kind at boundary ${i}: ${entry.kind}`);
    assert(isHash(entry.state), `Invalid state hash at boundary ${i}`);
    assert(isHash(entry.model), `Invalid model hash at boundary ${i}`);
    assert(isHash(entry.entryHash), `Invalid entryHash at boundary ${i}`);
    assert(isFiniteNumber(entry.time) && entry.time > 0, `Invalid time at boundary ${i}: ${entry.time}`);

    const control = fields(entry.control, `control at boundary ${i}`);
    assert(isCount(control.global_step), `Invalid global_step at boundary ${i}: ${control.global_step}`);
    assert(isCount(control.phase_step), `Invalid phase_step at boundary ${i}: ${control.phase_step}`);
    assert(isCount(control.epoch), `Invalid epoch at boundary ${i}: ${control.epoch}`);
    assert(isCount(control.cursor), `Invalid cursor at boundary ${i}: ${control.cursor}`);
    assert(isText(control.phase), `Invalid phase at boundary ${i}: ${control.phase}`);

    if (i === 0) {
      assert(entry.previous === null, 'Boundary 0 must have null previous pointer');
    } else {
      assert(
        entry.previous === chain[i - 1].entryHash,
        `Broken previous link at boundary ${i}: ${entry.previous} !== ${chain[i - 1].entryHash}`,
      );
    }
    assert(entry.time >= lastTime, `Time goes backwards at boundary ${i}`);
    assert(control.global_step >= lastStep, `Training step goes backwards at boundary ${i}`);
    lastTime = entry.time;
    lastStep = control.global_step;
    validateRecord(entry, i);
  });
  return chain;
}

function validateCheckRow(check: unknown, label: string, segmentAllowed: (segment: number) => boolean, range: string): void {
  const row = fields(check, label);
  assert(isCount(row.segment), `${label} segment must be an integer, got ${row.segment}`);
  assert(segmentAllowed(row.segment), `${label} points at invalid segment ${row.segment}; must be in ${range}`);
  assert(typeof row.checker === 'string', `${label} checker name must be a string`);
  assert(Object.hasOwn(KNOWN_CHECKER_MAP, row.checker), `Unknown or unauthorized checker string: "${row.checker}"`);
  assert(row.result === 'MATCH' || row.result === 'MISMATCH', `Invalid verification result: ${row.result}`);
  assert(isText(row.gpu), `${label} is missing its GPU`);
  assert(isText(row.code), `${label} is missing its code version`);
  assert(isText(row.driver), `${label} is missing its driver`);
}

function validateVerifications(root: Fields, segmentCount: number, boundaries: number): void {
  const ver = fields(root.verifications, 'verifications');
  assert(
    ver.uniqueSegments === segmentCount,
    `Segment count mismatch: chain has ${boundaries} boundaries (${segmentCount} segments), but verifications reported ${ver.uniqueSegments}`,
  );
  assert(Array.isArray(ver.coverage), 'Verifications coverage must be an array');
  assert(
    ver.coverage.length === segmentCount,
    `Coverage array length (${ver.coverage.length}) does not match derived segment count (${segmentCount})`,
  );
  assert(Array.isArray(ver.rawChecks), 'verifications.rawChecks must be an array');

  // Every row points at a real segment and a classified checker
  ver.rawChecks.forEach((check: unknown, r: number) =>
    validateCheckRow(check, `Check #${r}`, s => s < segmentCount, `[0, ${segmentCount - 1}]`),
  );

  // The published summary must say exactly what the raw rows say
  const rows = ver.rawChecks as RawVerificationCheck[];
  const derived = deriveCoverage(rows, segmentCount);
  ver.coverage.forEach((entry: unknown, s: number) => {
    const seg = fields(entry, `coverage for segment ${s}`);
    const want = derived[s];
    assert(seg.segment === s, `Coverage segment order mismatch at index ${s}`);
    assert(
      seg.projectResult === want.projectResult,
      `Coverage disagrees with raw checks: segment ${s} says project ${seg.projectResult}, the checks say ${want.projectResult}`,
    );
    assert(
      seg.independentResult === want.independentResult,
      `Coverage disagrees with raw checks: segment ${s} says independent ${seg.independentResult}, the checks say ${want.independentResult}`,
    );
    assert(
      Array.isArray(seg.checks) && seg.checks.length === want.checks.length,
      `Coverage disagrees with raw checks: segment ${s} lists a different number of checks`,
    );
  });

  const independent = rows.filter(c => KNOWN_CHECKER_MAP[c.checker] === 'independent').length;
  assert(ver.totalChecks === rows.length, `totalChecks (${ver.totalChecks}) does not match ${rows.length} raw checks`);
  assert(
    ver.independentChecksCount === independent,
    `independentChecksCount (${ver.independentChecksCount}) does not match ${independent} raw checks`,
  );
  assert(
    ver.projectChecksCount === rows.length - independent,
    `projectChecksCount (${ver.projectChecksCount}) does not match ${rows.length - independent} raw checks`,
  );
}

function validateHashes(root: Fields, chainLength: number): Fields {
  assert(isFields(root.sourceHashes), 'Missing sourceHashes');
  for (const [file, hash] of Object.entries(root.sourceHashes)) {
    assert(isHash(hash), `Invalid or corrupted SHA-256 hash for ${file}: ${hash}`);
  }
  assert(isFields(root.releaseHashes), 'Missing releaseHashes');
  assert(Object.keys(root.releaseHashes).length > 0, 'releaseHashes must not be empty');
  for (const [file, hash] of Object.entries(root.releaseHashes)) {
    assert(isHash(hash), `Invalid SHA-256 hash in releaseHashes for ${file}: ${hash}`);
  }
  for (const file of DOWNLOADED_FILES) {
    assert(isHash(root.releaseHashes[file]), `releaseHashes has no hash for ${file}`);
  }
  // One checkpoint per boundary, and the fetched record is the one the release lists
  const checkpoints = Object.keys(root.releaseHashes).filter(file => /^ckpt-\d+\.pt$/.test(file)).length;
  assert(checkpoints === chainLength, `release-hashes.txt lists ${checkpoints} checkpoints for ${chainLength} boundaries`);
  assert(
    root.releaseHashes['chain.jsonl'] === root.sourceHashes[RUN_CHAIN],
    'chain.jsonl does not match its hash in release-hashes.txt',
  );
  return root.sourceHashes;
}

function validateAnchor(root: Fields, chain: Fields[], source: Fields): void {
  const anchor = fields(root.anchor, 'anchor');
  const head = chain[chain.length - 1];
  assert(anchor.chain_entries === chain.length, `Anchor records ${anchor.chain_entries} chain entries, the chain has ${chain.length}`);
  assert(anchor.chain_head_index === head.index, `Anchor head index ${anchor.chain_head_index} is not the last boundary ${head.index}`);
  assert(anchor.chain_head_sha256 === head.entryHash, 'Anchor chain head does not match the last chain entry');
  assert(anchor.final_state === head.state, 'Anchor final state does not match the last chain entry');
  assert(anchor.final_model === head.model, 'Anchor final model does not match the last chain entry');
  assert(anchor.chain_jsonl_sha256 === source[RUN_CHAIN], 'Anchor does not match the hash of chain.jsonl');
  assert(anchor.release_hashes_sha256 === source[RUN_HASHES], 'Anchor does not match the hash of release-hashes.txt');
  // The whole file, rebuilt from the records, must be the file the anchor names.
  const chainFile = `${chain.map(entry => entry.record).join('\n')}\n`;
  assert(sha256Hex(chainFile) === anchor.chain_jsonl_sha256, 'The chain records do not hash to the chain.jsonl the anchor names');
  assert(anchor.dataset === root.dataset, `Anchor names dataset ${anchor.dataset}, not ${root.dataset}`);

  assert(Array.isArray(anchor.timestamps) && anchor.timestamps.length > 0, 'Anchor must have at least one timestamp');
  anchor.timestamps.forEach((stamp: unknown, i: number) => {
    const row = fields(stamp, `anchor timestamp ${i}`);
    assert(isText(row.authority), `Anchor timestamp ${i} has no authority`);
    assert(isText(row.file), `Anchor timestamp ${i} has no file`);
    assert(isDate(row.time), `Anchor timestamp ${i} has an invalid time: ${row.time}`);
  });
}

function validateHardware(root: Fields, segmentCount: number): void {
  assert(Array.isArray(root.hardwareMap) && root.hardwareMap.length > 0, 'hardwareMap must be a non-empty array');
  root.hardwareMap.forEach((probe: unknown, i: number) => {
    const row = fields(probe, `hardware probe ${i}`);
    assert(isText(row.hardware), `Hardware probe ${i} has no hardware name`);
    assert(row.result === 'MATCH' || row.result === 'MISMATCH', `Invalid result for hardware probe ${i}: ${row.result}`);
    const inRange = isCount(row.segment) && row.segment < segmentCount;
    assert(inRange || row.segment === `all ${segmentCount}`, `Hardware probe ${i} points at invalid segment ${row.segment}`);
  });
}

function validatePreparation(root: Fields, source: Fields): void {
  const prep = fields(root.preparation, 'preparation');
  assert(isPositive(prep.modelParametersMillions), `modelParametersMillions must be a positive integer, got ${prep.modelParametersMillions}`);
  assert(isPositive(prep.articles), `articles count must be a positive integer, got ${prep.articles}`);
  assert(isPositive(prep.wikipediaTargets), `wikipediaTargets count must be a positive integer, got ${prep.wikipediaTargets}`);
  assert(isPositive(prep.conversations), `conversations count must be a positive integer, got ${prep.conversations}`);
  assert(isPositive(prep.trainingCodeLines), `trainingCodeLines must be a positive integer, got ${prep.trainingCodeLines}`);
  assert(isHash(prep.preparedDataId), `Invalid preparedDataId SHA-256 hash: ${prep.preparedDataId}`);
  assert(
    source[`prepared/${prep.preparedDataId}/preparation.json`] === prep.preparedDataId,
    'preparedDataId is not the hash of its preparation.json',
  );
}

function validateFork(excl: Fields, chain: Fields[]): { first: number; last: number } {
  const fork = fields(excl.fork, 'exclusion fork');
  const { fromEntry, entryIndex, entries, sharedEntries, previous, state } = fork;
  assert(isCount(fromEntry) && fromEntry < chain.length - 1, `Invalid fork point: from entry ${fromEntry}`);
  assert(entryIndex === fromEntry + 1, `Fork entry ${entryIndex} does not follow from entry ${fromEntry}`);
  assert(sharedEntries === fromEntry + 1, `Fork shares ${sharedEntries} entries but splits from entry ${fromEntry}`);
  assert(isCount(entries) && entries > entryIndex, `Fork has ${entries} entries, too few for fork entry ${entryIndex}`);
  // The split must point at the run's own entry and carry its state unchanged.
  const parent = chain[fromEntry];
  assert(previous === parent.entryHash, `Fork does not link to entry ${fromEntry} of the run`);
  assert(state === parent.state, `Fork entry changes the state of entry ${fromEntry}`);
  return { first: fromEntry, last: entries - 2 };
}

function validateExclusion(root: Fields, chain: Fields[]): void {
  const excl = fields(root.exclusion, 'exclusion');
  assert(isText(excl.removedArticle), 'Missing removedArticle');
  assert(isFiniteNumber(excl.originalRemovedLoss) && excl.originalRemovedLoss > 0, 'Invalid originalRemovedLoss');
  assert(isFiniteNumber(excl.forkRemovedLoss) && excl.forkRemovedLoss > 0, 'Invalid forkRemovedLoss');
  assert(isFiniteNumber(excl.removedLossDelta), 'Invalid removedLossDelta');
  assert(
    Math.abs(excl.removedLossDelta - (excl.forkRemovedLoss - excl.originalRemovedLoss)) < 0.001,
    `Exclusion removed loss delta inconsistent: ${excl.removedLossDelta} !== ${excl.forkRemovedLoss} - ${excl.originalRemovedLoss}`,
  );
  assert(isPositive(excl.controlArticlesCount), `controlArticlesCount must be a positive integer, got ${excl.controlArticlesCount}`);
  assert(isFiniteNumber(excl.originalControlLossMean) && excl.originalControlLossMean > 0, 'Invalid originalControlLossMean');
  assert(isFiniteNumber(excl.forkControlLossMean) && excl.forkControlLossMean > 0, 'Invalid forkControlLossMean');
  assert(isFiniteNumber(excl.controlLossDelta), 'Invalid controlLossDelta');
  assert(
    Math.abs(excl.controlLossDelta - (excl.forkControlLossMean - excl.originalControlLossMean)) < 0.001,
    `Exclusion control loss delta inconsistent: ${excl.controlLossDelta} !== ${excl.forkControlLossMean} - ${excl.originalControlLossMean}`,
  );
  assert(excl.evidencePath === 'release/unlearning-3i-atlas/', `Invalid exclusion evidencePath: ${excl.evidencePath}`);
  assert(Array.isArray(excl.forkChecks) && excl.forkChecks.length > 0, 'Exclusion has no fork checks');
  // Fork checks can only be on segments after the split, up to the fork's own last one.
  const { first, last } = validateFork(excl, chain);
  excl.forkChecks.forEach((check: unknown, i: number) =>
    validateCheckRow(check, `Fork check #${i}`, s => s >= first && s <= last, `[${first}, ${last}]`),
  );
}

export function validateEvidence(data: unknown): EvidenceData {
  assert(isFields(data), 'Evidence data must be a non-null object');
  const root = data;

  assert(isPositive(root.schemaVersion), `Invalid schemaVersion: ${root.schemaVersion}`);
  assert(
    typeof root.dataset === 'string' && DATASET_ID_REGEX.test(root.dataset) && !/\.\.|--/.test(root.dataset),
    `Invalid dataset identifier, expected owner/name: ${JSON.stringify(root.dataset)}`,
  );
  assert(
    typeof root.revision === 'string' && /^[0-9a-f]{7,40}$/i.test(root.revision),
    `Invalid evidence revision format: ${root.revision}`,
  );
  assert(
    typeof root.revisionFull === 'string' && /^[0-9a-f]{40}$/i.test(root.revisionFull) && root.revisionFull.startsWith(root.revision),
    `revisionFull ${root.revisionFull} is not the full form of revision ${root.revision}`,
  );
  const release = fields(root.release, 'release');
  assert(
    typeof release.tag === 'string' && RELEASE_TAG_REGEX.test(release.tag),
    `Invalid release tag, expected a version like v1.2.3: ${JSON.stringify(release.tag)}`,
  );
  assert(isDate(release.publishedAt), `Invalid release date: ${release.publishedAt}`);

  const chain = validateChain(root);
  const segmentCount = chain.length - 1;
  validateVerifications(root, segmentCount, chain.length);
  const source = validateHashes(root, chain.length);
  validateAnchor(root, chain, source);
  validateHardware(root, segmentCount);
  validatePreparation(root, source);
  validateExclusion(root, chain);

  return data as unknown as EvidenceData;
}
