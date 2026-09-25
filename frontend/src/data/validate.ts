import {
  AVAILABILITIES,
  CHECK_RESULTS,
  DATA_MODES,
  SCOPES,
  SNAPSHOT_SCHEMA,
  SnapshotError,
  type CheckResult,
  type CheckView,
  type CommandView,
  type DataMode,
  type EvidenceItem,
  type EvidenceScope,
  type PipelineStep,
  type ReleaseAvailability,
  type ReleaseRole,
  type ReleaseView,
  type Snapshot,
  type SnapshotSummary,
  type SourceReference,
} from './contracts';

const RESULT_ALIASES: Record<string, CheckResult> = {
  passed: 'PASS',
  pass: 'PASS',
  failed: 'FAIL',
  fail: 'FAIL',
  pending: 'NOT_RUN',
  not_run: 'NOT_RUN',
  unavailable: 'UNAVAILABLE',
  unsupported: 'UNSUPPORTED',
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asString(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new SnapshotError(`Missing or empty string: ${field}`);
  }
  return value;
}

function asNullableString(value: unknown, field: string): string | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== 'string') {
    throw new SnapshotError(`${field} must be a string or null`);
  }
  return value;
}

function asBoolean(value: unknown, field: string): boolean {
  if (typeof value !== 'boolean') {
    throw new SnapshotError(`${field} must be a boolean`);
  }
  return value;
}

function asNumberOrNull(value: unknown, field: string): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw new SnapshotError(`${field} must be a number or null`);
  }
  return value;
}

function asStringList(value: unknown, field: string): string[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new SnapshotError(`${field} must be a list of strings`);
  }
  return value;
}

function parseResult(value: unknown, field: string): CheckResult {
  if (typeof value !== 'string') {
    throw new SnapshotError(`${field} is not a result string`);
  }
  if ((CHECK_RESULTS as readonly string[]).includes(value)) {
    return value as CheckResult;
  }
  const alias = RESULT_ALIASES[value.toLowerCase()];
  if (alias) return alias;
  throw new SnapshotError(
    `${field} value ${JSON.stringify(value)} is not a known check result`,
  );
}

function parseScope(value: unknown, field: string): EvidenceScope {
  if (typeof value !== 'string' || !(SCOPES as readonly string[]).includes(value)) {
    throw new SnapshotError(`${field} must be fixture, pilot, or production`);
  }
  return value as EvidenceScope;
}

function parseMode(value: unknown): DataMode {
  if (typeof value !== 'string' || !(DATA_MODES as readonly string[]).includes(value)) {
    throw new SnapshotError('mode must be fixture or public-snapshot');
  }
  return value as DataMode;
}

function parseAvailability(value: unknown, field: string): ReleaseAvailability {
  if (
    typeof value !== 'string' ||
    !(AVAILABILITIES as readonly string[]).includes(value)
  ) {
    throw new SnapshotError(`${field} is not a known availability`);
  }
  return value as ReleaseAvailability;
}

function parseSource(value: unknown, field: string): SourceReference {
  if (!isRecord(value)) throw new SnapshotError(`${field} must be an object`);
  const url = asString(value.url, `${field}.url`);
  if (url.startsWith('javascript:') || url.startsWith('data:')) {
    throw new SnapshotError(`${field}.url uses a disallowed scheme`);
  }
  return {
    url,
    revision: asNullableString(value.revision, `${field}.revision`),
    path: asNullableString(value.path, `${field}.path`),
    sha256: asNullableString(value.sha256, `${field}.sha256`),
  };
}

function parseCheck(value: unknown, index: number): CheckView {
  if (!isRecord(value)) throw new SnapshotError(`checks[${index}] must be an object`);
  const result = parseResult(value.result, `checks[${index}].result`);
  const check: CheckView = {
    id: asString(value.id, `checks[${index}].id`),
    profile: asString(value.profile, `checks[${index}].profile`),
    phase: asString(value.phase, `checks[${index}].phase`),
    scope: parseScope(value.scope, `checks[${index}].scope`),
    result,
    explanation: asString(value.explanation, `checks[${index}].explanation`),
    performedBy: asNullableString(value.performedBy, `checks[${index}].performedBy`),
    attestedBy: asNullableString(value.attestedBy, `checks[${index}].attestedBy`),
    locallyRecomputed: asBoolean(
      value.locallyRecomputed,
      `checks[${index}].locallyRecomputed`,
    ),
    evidence: Array.isArray(value.evidence)
      ? value.evidence.map((ref, j) => parseSource(ref, `checks[${index}].evidence[${j}]`))
      : [],
  };
  if (result === 'PASS' && check.evidence.length === 0) {
    throw new SnapshotError(`checks[${index}] PASS is missing evidence references`);
  }
  return check;
}

function parseEvidence(value: unknown, index: number): EvidenceItem {
  if (!isRecord(value)) throw new SnapshotError(`evidence[${index}] must be an object`);
  return {
    id: asString(value.id, `evidence[${index}].id`),
    title: asString(value.title, `evidence[${index}].title`),
    kind: asString(value.kind, `evidence[${index}].kind`),
    phase: asString(value.phase, `evidence[${index}].phase`),
    scope: parseScope(value.scope, `evidence[${index}].scope`),
    result: parseResult(value.result, `evidence[${index}].result`),
    checkId: asNullableString(value.checkId, `evidence[${index}].checkId`),
    source: parseSource(value.source, `evidence[${index}].source`),
    parentIds: asStringList(value.parentIds ?? [], `evidence[${index}].parentIds`),
    childIds: asStringList(value.childIds ?? [], `evidence[${index}].childIds`),
    sizeBytes: asNumberOrNull(value.sizeBytes, `evidence[${index}].sizeBytes`),
    timestamp: asNullableString(value.timestamp, `evidence[${index}].timestamp`),
    supersededBy: asNullableString(value.supersededBy, `evidence[${index}].supersededBy`),
    supersedes: asNullableString(value.supersedes, `evidence[${index}].supersedes`),
  };
}

function parsePipeline(value: unknown, index: number): PipelineStep {
  if (!isRecord(value)) throw new SnapshotError(`pipeline[${index}] must be an object`);
  return {
    id: asString(value.id, `pipeline[${index}].id`),
    title: asString(value.title, `pipeline[${index}].title`),
    phase: asString(value.phase, `pipeline[${index}].phase`),
    result: parseResult(value.result, `pipeline[${index}].result`),
    scope: parseScope(value.scope, `pipeline[${index}].scope`),
    checkId: asNullableString(value.checkId, `pipeline[${index}].checkId`),
  };
}

function parseRelease(value: unknown, index: number): ReleaseView {
  if (!isRecord(value)) throw new SnapshotError(`releases[${index}] must be an object`);
  const role = asString(value.role, `releases[${index}].role`);
  if (role !== 'base' && role !== 'chat') {
    throw new SnapshotError(`releases[${index}].role must be base or chat`);
  }
  return {
    id: asString(value.id, `releases[${index}].id`),
    role: role as ReleaseRole,
    availability: parseAvailability(value.availability, `releases[${index}].availability`),
    title: asString(value.title, `releases[${index}].title`),
    parentReleaseId: asNullableString(
      value.parentReleaseId,
      `releases[${index}].parentReleaseId`,
    ),
    repository: asNullableString(value.repository, `releases[${index}].repository`),
    revision: asNullableString(value.revision, `releases[${index}].revision`),
    releaseRoot: asNullableString(value.releaseRoot, `releases[${index}].releaseRoot`),
    notes: asString(value.notes, `releases[${index}].notes`),
  };
}

function parseCommand(value: unknown, index: number): CommandView {
  if (!isRecord(value)) throw new SnapshotError(`commands[${index}] must be an object`);
  return {
    profile: asString(value.profile, `commands[${index}].profile`),
    text: asString(value.text, `commands[${index}].text`),
    prerequisites: asString(value.prerequisites, `commands[${index}].prerequisites`),
    sourceRevision: asNullableString(
      value.sourceRevision,
      `commands[${index}].sourceRevision`,
    ),
    expectedScope: asString(value.expectedScope, `commands[${index}].expectedScope`),
    resourceNote: asString(value.resourceNote, `commands[${index}].resourceNote`),
    validated: asBoolean(value.validated, `commands[${index}].validated`),
  };
}

function parseSummary(value: unknown): SnapshotSummary {
  if (!isRecord(value)) throw new SnapshotError('summary must be an object');
  return {
    headline: asString(value.headline, 'summary.headline'),
    mayClaim: asStringList(value.mayClaim, 'summary.mayClaim'),
    mayNotClaim: asStringList(value.mayNotClaim, 'summary.mayNotClaim'),
    workflow: asString(value.workflow, 'summary.workflow'),
  };
}

export function validateSnapshot(raw: unknown): Snapshot {
  if (!isRecord(raw)) throw new SnapshotError('Snapshot must be a JSON object');
  const schemaVersion = asString(raw.schemaVersion, 'schemaVersion');
  if (schemaVersion !== SNAPSHOT_SCHEMA) {
    throw new SnapshotError(
      `Unsupported schema ${schemaVersion}; expected ${SNAPSHOT_SCHEMA}`,
    );
  }

  const checks = Array.isArray(raw.checks)
    ? raw.checks.map((item, i) => parseCheck(item, i))
    : [];
  const evidence = Array.isArray(raw.evidence)
    ? raw.evidence.map((item, i) => parseEvidence(item, i))
    : [];
  const releases = Array.isArray(raw.releases)
    ? raw.releases.map((item, i) => parseRelease(item, i))
    : [];
  const pipeline = Array.isArray(raw.pipeline)
    ? raw.pipeline.map((item, i) => parsePipeline(item, i))
    : [];
  const commands = Array.isArray(raw.commands)
    ? raw.commands.map((item, i) => parseCommand(item, i))
    : [];

  if (checks.length === 0) {
    throw new SnapshotError(
      'Snapshot has no checks; an empty docket cannot be shown as success',
    );
  }

  return {
    schemaVersion,
    generatedAt: asString(raw.generatedAt, 'generatedAt'),
    mode: parseMode(raw.mode),
    source: parseSource(raw.source, 'source'),
    summary: parseSummary(raw.summary),
    pipeline,
    releases,
    checks,
    evidence,
    commands,
  };
}

export function releaseIsDownloadable(release: ReleaseView): boolean {
  return (
    release.availability === 'available' &&
    Boolean(release.repository) &&
    Boolean(release.revision)
  );
}
