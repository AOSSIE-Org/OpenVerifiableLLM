// Contracts and schemas for OpenVerifiableLLM evidence data

export type CanonicalCheckerCategory = 'project' | 'independent';

// Explicit whitelist mapping for recognized checker strings in evidence
export const KNOWN_CHECKER_MAP: Readonly<Record<string, CanonicalCheckerCategory>> = Object.freeze({
  'project second pod': 'project',
  'project training pod': 'project',
  'project fresh machine public files only': 'project',
});

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
  result: 'MATCH' | 'MISMATCH';
  segment: number;
}

export interface SegmentCoverage {
  segment: number;
  projectResult: 'MATCH' | 'MISMATCH';
  independentResult: 'MATCH' | 'MISMATCH' | null;
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
  result: 'MATCH' | 'MISMATCH';
  hardware: string;
  driver: string;
  software: string;
  note: string;
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
  timestamp: string;
  authorities: string[];
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
}

export interface PreparationData {
  articles: number;
  wikipediaTargets: number;
  conversations: number;
  trainingLoopLines: string;
  preparedDataId: string;
}

export interface EvidenceData {
  schemaVersion: number;
  dataset: string;
  revision: string;
  fetchedAt: string;
  sourceHashes: Record<string, string>;
  preparation: PreparationData;
  chain: ChainBoundary[];
  verifications: VerificationsData;
  hardwareMap: HardwareProbe[];
  anchor: ReleaseAnchor;
  exclusion: ExclusionData;
  releaseHashes: Record<string, string>;
}
