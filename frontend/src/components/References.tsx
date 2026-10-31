import React from 'react';
import { copy, links } from '../content/copy';
import type { EvidenceData } from '../data/contracts';
import { SectionHeading } from './Contents';
import { refKeyForPath, refNumber } from './Ref';

export const Terms: React.FC = () => (
  <section id="terms" aria-labelledby="terms-h">
    <SectionHeading id="terms" />
    <dl className="terms">
      {copy.terms.map(t => (
        <React.Fragment key={t.term}>
          <dt>{t.term}</dt>
          <dd>{t.def}</dd>
        </React.Fragment>
      ))}
    </dl>
  </section>
);

export const References: React.FC<{ evidence: EvidenceData }> = ({ evidence }) => {
  const entries = Object.entries(evidence.sourceHashes)
    .map(([path, sha]) => ({ path, sha, key: refKeyForPath(path) }))
    .filter((e): e is { path: string; sha: string; key: NonNullable<typeof e.key> } => e.key !== undefined)
    .sort((a, b) => refNumber(a.key) - refNumber(b.key));

  return (
    <section id="references" aria-labelledby="references-h">
      <SectionHeading id="references" />
      <p>
        {copy.references.intro} Dataset <code>{evidence.dataset}</code>, revision <code>{evidence.revision}</code>.
      </p>
      <ol className="references">
        {entries.map(e => (
          <li key={e.path} id={`ref-${e.key}`} value={refNumber(e.key)}>
            {copy.references.labels[e.path] ?? copy.references.preparationLabel}.{' '}
            <a href={links.evidenceFile(evidence.revision, e.path)}>
              <code>{e.path}</code>
            </a>
            . SHA-256 <code className="hash full">{e.sha}</code>.
          </li>
        ))}
      </ol>
    </section>
  );
};
