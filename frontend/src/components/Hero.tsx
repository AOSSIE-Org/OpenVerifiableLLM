import React from 'react';
import { copy } from '../content/copy';
import type { EvidenceData } from '../data/contracts';

interface HeroProps {
  evidence: EvidenceData;
}

// What the project is, the two actions, and the numbers that matter most.
export const Hero: React.FC<HeroProps> = ({ evidence }) => {
  const a = copy.article;
  const { verifications, revision } = evidence;
  const segments = verifications.coverage.length;
  const matched = verifications.coverage.filter(c => c.projectResult === 'MATCH').length;

  return (
    <div className="hero" id="top">
      <p className="byline">
        {a.from} · {a.kicker}
      </p>
      <h1>{a.title}</h1>
      <p className="intro">
        <b>OVL</b> {a.lead[0]}
      </p>
      <p className="support">{a.lead[1]}</p>
      <p className="actions">
        <a className="action primary" href="#overview">
          {a.primaryAction}
        </a>
        <a className="action" href={copy.header.links[0].href}>
          {a.secondaryAction}
        </a>
      </p>

      <aside className="facts" aria-labelledby="facts-title">
        <h2 id="facts-title" className="sr-only">
          {a.factsTitle}
        </h2>
        <dl className="figures">
          <div>
            <dd>
              {matched} / {segments}
            </dd>
            <dt>{a.figures.recomputed}</dt>
          </div>
          <div className="caveat">
            <dd>{verifications.independentChecksCount}</dd>
            <dt>{a.figures.independent}</dt>
          </div>
          <div>
            <dd>{verifications.projectChecksCount}</dd>
            <dt>{a.figures.checks}</dt>
          </div>
          <div>
            <dd>{a.figures.stamped}</dd>
            <dt>{a.figures.stampedLabel}</dt>
          </div>
        </dl>
        <p className="note">
          {a.evidenceLine}{' '}
          <a href={copy.header.links[1].href}>
            <code>{revision}</code>
          </a>
          .
        </p>
      </aside>
    </div>
  );
};
