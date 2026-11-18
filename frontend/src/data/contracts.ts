// Contracts and schemas for OpenVerifiableLLM evidence data
import checkers from './checkers.json' with { type: 'json' };

export type CanonicalCheckerCategory = 'project' | 'independent';
export type CheckResult = 'MATCH' | 'MISMATCH';
// null means nobody has checked it: never a pass, never a failure.
export type CheckOutcome = CheckResult | null;

// Every checker string in the evidence must be classified before it can be shown.
// The list is shared with scripts/fetch-evidence.mjs.
export const KNOWN_CHECKER_MAP = Object.freeze(checkers) as Readonly<Record<string, CanonicalCheckerCategory>>;

export interface ChainControl {
  batches?: string;
  cursor: number;
  epoch: number;
  global_step: number;
  phase: string;
  phase_step: number;
  pos: [number, number];
}

export interface ChainBoundary {
  // The published line of chain.jsonl. entryHash is the SHA-256 of these exact bytes.
  record: string;
  index: number;
  kind: 'initial' | 'progress' | 'base' | 'transition' | 'epoch' | 'final';
  control: ChainControl;
  state: string;
  model: string;
  previous: string | null;
  time: number;
  entryHash: string;
}

export interface RawVerificationCheck {
  checker: string;
  code: string;
  driver: string;
  gpu: string;
  result: CheckResult;
  segment: number;
}

export interface SegmentCoverage {
  segment: number;
  projectResult: CheckOutcome;
  independentResult: CheckOutcome;
  checks: RawVerificationCheck[];
}

export interface VerificationsData {
  totalChecks: number;
  uniqueSegments: number;
  projectChecksCount: number;
  independentChecksCount: number;
  coverage: SegmentCoverage[];
  rawChecks: RawVerificationCheck[];
}

export interface HardwareProbe {
  segment: number | string;
  result: CheckResult;
  hardware: string;
  driver: string;
  software: string;
  note: string;
}

export interface AnchorTimestamp {
  authority: string;
  file: string;
  time: string;
}

export interface ReleaseAnchor {
  chain_entries: number;
  chain_head_index: number;
  chain_head_sha256: string;
  chain_jsonl_sha256: string;
  commit: string;
  dataset: string;
  final_model: string;
  final_state: string;
  path: string;
  release_hashes_sha256: string;
  schema: string;
  timestamps: AnchorTimestamp[];
}

// The fork's split, as its own chain records it
export interface ForkPoint {
  fromEntry: number;
  entryIndex: number;
  entries: number;
  sharedEntries: number;
  previous: string;
  state: string;
  excluded: Record<string, [number, number][]>;
  anchored: boolean;
}

// Where training's two unplanned events happened, read from docs/how-it-works.md
export interface TrainingEvents {
  hostFailureSegment: number;
  codeChangeBoundary: number;
}

export interface ExclusionData {
  removedArticle: string;
  originalRemovedLoss: number;
  forkRemovedLoss: number;
  removedLossDelta: number;
  originalControlLossMean: number;
  forkControlLossMean: number;
  controlLossDelta: number;
  controlArticlesCount: number;
  evidencePath: string;
  fork: ForkPoint;
  forkChecks: RawVerificationCheck[];
}

export interface PreparationData {
  modelParametersMillions: number;
  articles: number;
  wikipediaTargets: number;
  conversations: number;
  trainingCodeLines: number;
  preparedDataId: string;
}

export interface ReleaseInfo {
  tag: string;
  publishedAt: string;
}

export interface EvidenceData {
  schemaVersion: number;
  dataset: string;
  revision: string;
  revisionFull: string;
  sourceHashes: Record<string, string>;
  release: ReleaseInfo;
  preparation: PreparationData;
  trainingEvents: TrainingEvents;
  chain: ChainBoundary[];
  verifications: VerificationsData;
  hardwareMap: HardwareProbe[];
  anchor: ReleaseAnchor;
  exclusion: ExclusionData;
  releaseHashes: Record<string, string>;
}
