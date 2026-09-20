export const SNAPSHOT_SCHEMA = 'ovllm.frontend.snapshot.v1';

export type CheckResult =
  | 'PASS'
  | 'FAIL'
  | 'NOT_RUN'
  | 'UNAVAILABLE'
  | 'UNSUPPORTED';

export type EvidenceScope = 'fixture' | 'pilot' | 'production';
export type DataMode = 'fixture' | 'public-snapshot';
export type ReleaseAvailability = 'not-released' | 'available' | 'withdrawn';
export type ReleaseRole = 'base' | 'chat';

export interface SourceReference {
  url: string;
  revision: string | null;
  path: string | null;
  sha256: string | null;
}

export interface CheckView {
  id: string;
  profile: string;
  phase: string;
  scope: EvidenceScope;
  result: CheckResult;
  explanation: string;
  performedBy: string | null;
  attestedBy: string | null;
  locallyRecomputed: boolean;
  evidence: SourceReference[];
}

export interface EvidenceItem {
  id: string;
  title: string;
  kind: string;
  phase: string;
  scope: EvidenceScope;
  result: CheckResult;
  checkId: string | null;
  source: SourceReference;
  parentIds: string[];
  childIds: string[];
  sizeBytes: number | null;
  timestamp: string | null;
  supersededBy: string | null;
  supersedes: string | null;
}

export interface PipelineStep {
  id: string;
  title: string;
  phase: string;
  result: CheckResult;
  scope: EvidenceScope;
  checkId: string | null;
}

export interface ReleaseView {
  id: string;
  role: ReleaseRole;
  availability: ReleaseAvailability;
  title: string;
  parentReleaseId: string | null;
  repository: string | null;
  revision: string | null;
  releaseRoot: string | null;
  notes: string;
}

export interface CommandView {
  profile: string;
  text: string;
  prerequisites: string;
  sourceRevision: string | null;
  expectedScope: string;
  resourceNote: string;
  validated: boolean;
}

export interface SnapshotSummary {
  headline: string;
  mayClaim: string[];
  mayNotClaim: string[];
  workflow: string;
}

export interface Snapshot {
  schemaVersion: string;
  generatedAt: string;
  mode: DataMode;
  source: SourceReference;
  summary: SnapshotSummary;
  pipeline: PipelineStep[];
  releases: ReleaseView[];
  checks: CheckView[];
  evidence: EvidenceItem[];
  commands: CommandView[];
}

export const CHECK_RESULTS: readonly CheckResult[] = [
  'PASS',
  'FAIL',
  'NOT_RUN',
  'UNAVAILABLE',
  'UNSUPPORTED',
];

export const SCOPES: readonly EvidenceScope[] = ['fixture', 'pilot', 'production'];
export const DATA_MODES: readonly DataMode[] = ['fixture', 'public-snapshot'];
export const AVAILABILITIES: readonly ReleaseAvailability[] = [
  'not-released',
  'available',
  'withdrawn',
];

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}
