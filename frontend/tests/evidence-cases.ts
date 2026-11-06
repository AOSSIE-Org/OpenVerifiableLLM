// Variations of the real snapshot: each stays internally consistent, so the validator
// accepts it, and the page has to show what it says.
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { EvidenceData } from '../src/data/contracts';
import { deriveCoverage } from '../src/data/facts';

export const loadEvidence = (): EvidenceData =>
  JSON.parse(readFileSync(resolve(process.cwd(), 'public/data/evidence.json'), 'utf8'));

function resummarize(evidence: EvidenceData): EvidenceData {
  const v = evidence.verifications;
  v.coverage = deriveCoverage(v.rawChecks, v.uniqueSegments);
  v.totalChecks = v.rawChecks.length;
  v.projectChecksCount = v.rawChecks.length;
  return evidence;
}

// One check on this segment came back different
export function withMismatch(segment: number, evidence = loadEvidence()): EvidenceData {
  const row = evidence.verifications.rawChecks.find(c => c.segment === segment);
  if (!row) throw new Error(`No check on segment ${segment}`);
  row.result = 'MISMATCH';
  return resummarize(evidence);
}

// Nobody has checked this segment
export function withoutChecks(segment: number, evidence = loadEvidence()): EvidenceData {
  evidence.verifications.rawChecks = evidence.verifications.rawChecks.filter(c => c.segment !== segment);
  return resummarize(evidence);
}
