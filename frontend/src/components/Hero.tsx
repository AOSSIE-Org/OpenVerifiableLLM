import React from 'react';
import { claimsFor } from '../content/claims';
import { copy } from '../content/copy';
import type { Facts } from '../data/facts';

// What the project is, the two actions, and the numbers that matter most.
export const Hero: React.FC<{ facts: Facts }> = ({ facts }) => {
  const a = copy.article;
  const c = claimsFor(facts).hero;

  return (
    <div className="hero" id="top">
      <p className="byline">
        {a.from} · {c.kicker}
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
        <a className="action" href={copy.header.source}>
          {a.secondaryAction}
        </a>
      </p>

      <aside className="facts" aria-labelledby="facts-title">
        <h2 id="facts-title" className="sr-only">
          {a.factsTitle}
        </h2>
        <dl className="figures">
          <div className={facts.allProjectMatched ? undefined : 'caveat'}>
            <dt>{c.recomputed}</dt>
            <dd>
              {facts.projectMatched} / {facts.segments}
            </dd>
          </div>
          <div className="caveat">
            <dt>{c.independent}</dt>
            <dd>{facts.independentChecks}</dd>
          </div>
          <div className={facts.projectChecksMismatched === 0 ? undefined : 'caveat'}>
            <dt>{c.checks}</dt>
            <dd>{facts.projectChecks}</dd>
          </div>
          <div>
            <dt>{c.stampedLabel}</dt>
            <dd>{facts.stampDateShort}</dd>
          </div>
        </dl>
        <p className="note">
          {a.evidenceLine}{' '}
          <a href={facts.evidenceUrl}>
            <code>{facts.revision}</code>
          </a>
          .
        </p>
      </aside>
    </div>
  );
};
