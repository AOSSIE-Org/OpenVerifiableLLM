export type Result = 'MATCH' | 'MISMATCH';
export type CheckedBy = 'project' | 'independent';

export interface Src {
  origin: 'hf' | 'github';
  file: string;
  rev: string;
  line: number | null;
  lineEnd: number | null;
  raw: string | null;
  checkedBy: CheckedBy | null;
  url: string;
}
export interface Boundary { index: number; kind: string; phase: string; time: number; utc: string; model: string; state: string; previous: string | null; src: Src }
export interface CheckRow { segment: number; result: Result; checker: string; machine: string; checkedBy: CheckedBy; code?: string | null; gpu?: string | null; driver?: string | null; src: Src }
export interface Group { checker: string; machine: string; checkedBy: CheckedBy; rows: number; match?: number; segments: number[] }
export interface Tile { segment: number; phase: string; result: Result | 'NOT_RUN'; checkedBy: CheckedBy | 'pending'; checks: number[]; events: string[] }
export interface EventItem { id: 'restart' | 'code-change'; segment: number; src: Src; gapSeconds?: number; medianGapSeconds?: number; codeBefore?: string; codeAfter?: string }
export interface HardwareRow { segment: string; result: Result; hardware: string; driver: string; software: string; note: string; src: Src }
export interface DownloadFile { file: string; sha256: string; download: string; src: Src }
export interface Release { release: string; files: DownloadFile[] }
export interface Loss { before: number; after: number; delta: number }

export interface ClaimSource { origin: 'hf' | 'github' | 'github-release' | 'brief'; file: string; rev: string; lines: string; text: string | null; url: string | null; quote: string | null }

export interface SiteData {
  schema: 'ovl-site-data.v2';
  dataset: string;
  revision: string;
  revShort: string;
  hfTree: string;
  upstream: { repo: string; commit: string; short: string };
  rawFiles: Record<string, { sha256: string; bytes: number }>;
  model: { millionParameters: number; src: Src; recipe: { layers: number; seed: number } };
  chain: { entries: number; segments: number; phases: { phase: string; first: number; last: number }[]; boundaries: Boundary[]; start: string; end: string; kinds: Record<string, number> };
  verifications: {
    rows: CheckRow[]; rowCount: number; groups: Group[]; tiles: Tile[]; machines: number; allMatch: boolean;
    segmentsMatched: number; secondMachineMatched: number; secondMachine: string; freshMachine: string; freshSegments: number[];
    projectChecks: number; independentChecks: number; src: Src;
  };
  odds: { N: number; Nm1: number; formula: string; m95: number; m99: number; p95: number; p99: number; pAtN: number; currentM: number; currentP: number; src: Src };
  events: EventItem[];
  hardware: HardwareRow[];
  downloads: Release[];
  anchor: { sha256: string; chainHead: string; entries: number; tsa: { authority: string; genTime: string; src: Src }[]; src: Src };
  exclusion: {
    title: string; page: string; pageRevision: string; titleSrc: Src; entries: number; sharedPrefix: number; sharedFirst: number; sharedLast: number;
    forkIndex: number; rows: CheckRow[]; rowCount: number; groups: Group[]; allMatch: boolean; controls: number;
    removedLoss: Loss; controlLoss: Loss; src: Src;
  };
  preparation: { sha256: string; articles: number; wikipediaTargets: number; conversations: number; conversationTargets: number; src: Src };
  claimSources: Record<string, ClaimSource[]>;
  claimExtraSources: Record<string, number>;
  check: { intro: string; introSrc: Src; commands: string; commandsSrc: Src; runpod: string; runpodSrc: Src; cost: string; costSrc: Src };
}
