import React from 'react';

// Citations, numbered like Wikipedia footnotes. Each one points at a file in
// the evidence dataset; the References section lists them with their SHA-256.
export const REFS = [
  { key: 'chain', match: (p: string) => p.endsWith('/chain.jsonl') },
  { key: 'verifications', match: (p: string) => p.endsWith('/verifications.jsonl') },
  { key: 'hardware', match: (p: string) => p.endsWith('/hardware-map.jsonl') },
  { key: 'hashes', match: (p: string) => p.endsWith('/release-hashes.txt') },
  { key: 'anchor', match: (p: string) => p.endsWith('/anchor.json') },
  { key: 'readme', match: (p: string) => p.endsWith('/README.md') },
  { key: 'loo', match: (p: string) => p.endsWith('/loo-results.json') },
  { key: 'preparation', match: (p: string) => p.endsWith('/preparation.json') },
] as const;

export type RefKey = (typeof REFS)[number]['key'];

export const refNumber = (key: RefKey): number => REFS.findIndex(r => r.key === key) + 1;

export const refKeyForPath = (path: string): RefKey | undefined => REFS.find(r => r.match(path))?.key;

export const Ref: React.FC<{ k: RefKey }> = ({ k }) => (
  <sup className="ref">
    <a href={`#ref-${k}`} aria-label={`Reference ${refNumber(k)}`}>
      [{refNumber(k)}]
    </a>
  </sup>
);
