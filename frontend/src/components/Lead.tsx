import React from 'react';
import { copy } from '../content/copy';
import type { EvidenceData } from '../data/contracts';

// The article title, a one-line byline and the lead paragraphs.
export const Lead: React.FC = () => {
  const a = copy.article;
  return (
    <div className="lead" id="top">
      <h1>{a.title}</h1>
      <p className="byline">
        {a.from} · {a.kicker}
      </p>
      <p className="intro">
        <b>{a.title}</b> (OVL) {a.lead[0]}
      </p>
      {a.lead.slice(1).map(p => (
        <p key={p}>{p}</p>
      ))}
    </div>
  );
};

interface KeyFactsProps {
  evidence: EvidenceData;
  // the card is rendered twice (rail and inline), so ids need a suffix
  idSuffix: string;
}

// The key numbers, kept beside the article on wide screens and under the
// title on narrow ones.
export const KeyFacts: React.FC<KeyFactsProps> = ({ evidence, idSuffix }) => {
  const titleId = `facts-title-${idSuffix}`;
  const box = copy.article.infobox;
  const { verifications, revision } = evidence;
  const segments = verifications.coverage.length;
  const matched = verifications.coverage.filter(c => c.projectResult === 'MATCH').length;

  return (
    <aside className="facts-card" aria-labelledby={titleId}>
      <div className="fc-head">
        <img className="logo-l" src="brand/logo-light.svg" width="28" height="28" alt="" />
        <img className="logo-d" src="brand/logo-dark.svg" width="28" height="28" alt="" />
        <h2 id={titleId}>{copy.article.factsTitle}</h2>
      </div>
      <dl>
        <div className="fc-big">
          <dt>Recomputed</dt>
          <dd>
            {matched} / {segments} <span>{box.checked}</span>
          </dd>
        </div>
        <div className="fc-big">
          <dt>Independent checks</dt>
          <dd>
            {verifications.independentChecksCount} <span>{box.independent}</span>
          </dd>
        </div>
        <div>
          <dt>Release</dt>
          <dd>{box.release}</dd>
        </div>
        <div>
          <dt>Parameters</dt>
          <dd>{box.parameters}</dd>
        </div>
        <div>
          <dt>Training data</dt>
          <dd>{box.data}</dd>
        </div>
        <div>
          <dt>Record stamped</dt>
          <dd>{box.stamped}</dd>
        </div>
        <div>
          <dt>Licence</dt>
          <dd>{box.license}</dd>
        </div>
        <div>
          <dt>Evidence</dt>
          <dd>
            <a href={copy.header.links[1].href}>
              revision <code>{revision}</code>
            </a>
          </dd>
        </div>
      </dl>
      <a className="fc-cta" href="#check">
        {copy.article.factsCta}
      </a>
    </aside>
  );
};
