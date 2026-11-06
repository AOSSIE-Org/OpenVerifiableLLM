// Everything the page says about the evidence is worked out here, from the records.
import {
  type CanonicalCheckerCategory,
  type CheckOutcome,
  type EvidenceData,
  KNOWN_CHECKER_MAP,
  type RawVerificationCheck,
  type SegmentCoverage,
} from './contracts';

// A group passes only if every check in it matched. No checks means no result.
export function verdict(checks: readonly Pick<RawVerificationCheck, 'result'>[]): CheckOutcome {
  if (checks.length === 0) return null;
  return checks.every(c => c.result === 'MATCH') ? 'MATCH' : 'MISMATCH';
}

export function deriveCoverage(
  rawChecks: readonly RawVerificationCheck[],
  segmentCount: number,
  categories: Readonly<Record<string, CanonicalCheckerCategory>> = KNOWN_CHECKER_MAP,
): SegmentCoverage[] {
  return Array.from({ length: segmentCount }, (_, segment) => {
    const checks = rawChecks.filter(c => c.segment === segment);
    return {
      segment,
      projectResult: verdict(checks.filter(c => categories[c.checker] === 'project')),
      independentResult: verdict(checks.filter(c => categories[c.checker] === 'independent')),
      checks,
    };
  });
}

// A set of segments passes only if each one passed. An unchecked segment means no result.
export function combine(outcomes: readonly CheckOutcome[]): CheckOutcome {
  if (outcomes.includes('MISMATCH')) return 'MISMATCH';
  if (outcomes.length === 0 || outcomes.includes(null)) return null;
  return 'MATCH';
}

// With N segments, the chance that M checkers who each pick one at random hit a single bad one.
export const detectionOdds = (checks: number, segments: number): number =>
  1 - Math.pow((segments - 1) / segments, checks);

export function checksForOdds(probability: number, segments: number): number {
  if (segments <= 1) return 1;
  return Math.ceil(Math.log(1 - probability) / Math.log((segments - 1) / segments));
}

const countOf = (
  coverage: readonly SegmentCoverage[],
  side: 'projectResult' | 'independentResult',
  outcome: CheckOutcome,
): number => coverage.filter(c => c[side] === outcome).length;

export const datasetUrl = (dataset: string, revision: string): string =>
  `https://huggingface.co/datasets/${dataset}/tree/${revision}`;

const dateFormat = (month: 'long' | 'short') =>
  new Intl.DateTimeFormat('en-GB', { day: 'numeric', month, year: 'numeric', timeZone: 'UTC' });

export const longDate = (iso: string): string => dateFormat('long').format(new Date(iso));
export const shortDate = (iso: string): string => dateFormat('short').format(new Date(iso));

export interface Facts {
  segments: number;
  boundaries: number;
  // Segments, by who checked and what they found
  projectMatched: number;
  projectMismatched: number;
  projectUnchecked: number;
  independentMatched: number;
  independentMismatched: number;
  independentChecked: number;
  allProjectMatched: boolean;
  // Individual check rows
  totalChecks: number;
  projectChecks: number;
  projectChecksMismatched: number;
  independentChecks: number;
  independentChecksMismatched: number;
  // Detection odds
  checksFor95: number;
  checksFor99: number;
  // Where it came from
  revision: string;
  revisionFull: string;
  evidenceUrl: string;
  dataset: string;
  releaseTag: string;
  releaseDate: string;
  stampDate: string;
  stampDateShort: string;
  authorities: string[];
  // The exclusion fork's own checks
  forkSegments: number;
  forkSegmentsMatched: number;
}

export function summarize(evidence: EvidenceData): Facts {
  const { verifications, chain, anchor, release, exclusion, dataset, revision, revisionFull } = evidence;
  const { coverage, rawChecks } = verifications;
  const segments = chain.length - 1;
  const isIndependent = (c: RawVerificationCheck) => KNOWN_CHECKER_MAP[c.checker] === 'independent';
  const independentChecks = rawChecks.filter(isIndependent);
  const projectChecks = rawChecks.filter(c => !isIndependent(c));
  const projectMatched = countOf(coverage, 'projectResult', 'MATCH');
  const independentMatched = countOf(coverage, 'independentResult', 'MATCH');
  const independentMismatched = countOf(coverage, 'independentResult', 'MISMATCH');
  const stampTime = anchor.timestamps.map(t => t.time).sort().at(-1) ?? '';
  const forkSegmentIds = [...new Set(exclusion.forkChecks.map(c => c.segment))];

  return {
    segments,
    boundaries: chain.length,
    projectMatched,
    projectMismatched: countOf(coverage, 'projectResult', 'MISMATCH'),
    projectUnchecked: countOf(coverage, 'projectResult', null),
    independentMatched,
    independentMismatched,
    independentChecked: independentMatched + independentMismatched,
    allProjectMatched: projectMatched === segments,
    totalChecks: rawChecks.length,
    projectChecks: projectChecks.length,
    projectChecksMismatched: projectChecks.filter(c => c.result === 'MISMATCH').length,
    independentChecks: independentChecks.length,
    independentChecksMismatched: independentChecks.filter(c => c.result === 'MISMATCH').length,
    checksFor95: checksForOdds(0.95, segments),
    checksFor99: checksForOdds(0.99, segments),
    revision,
    revisionFull,
    evidenceUrl: datasetUrl(dataset, revision),
    dataset,
    releaseTag: release.tag,
    releaseDate: longDate(release.publishedAt),
    stampDate: longDate(stampTime),
    stampDateShort: shortDate(stampTime),
    authorities: anchor.timestamps.map(t => t.authority),
    forkSegments: forkSegmentIds.length,
    forkSegmentsMatched: forkSegmentIds.filter(
      s => verdict(exclusion.forkChecks.filter(c => c.segment === s)) === 'MATCH',
    ).length,
  };
}
