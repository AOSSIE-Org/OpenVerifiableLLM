import React from 'react';
import { copy } from '../content/copy';
import type { PreparationData, VerificationsData } from '../data/contracts';

interface OverviewProps {
  preparation: PreparationData;
  verifications: VerificationsData;
  authorities: string[];
}

// A small drawing above each headline fact, made from the evidence.
const FactArt: React.FC<{ index: number } & OverviewProps> = ({ index, preparation, verifications, authorities }) => {
  if (index === 0) {
    return (
      <span className="art-ticks">
        {verifications.coverage.map(c => (
          <i key={c.segment} className={c.projectResult === 'MATCH' ? 'ok' : ''} />
        ))}
      </span>
    );
  }
  if (index === 1) {
    const id = `${preparation.preparedDataId.slice(0, 8)}…`;
    return (
      <span className="art-same">
        <span>
          raw dump <i /> <code>{id}</code>
        </span>
        <span>
          rebuilt <i /> <code>{id}</code>
        </span>
      </span>
    );
  }
  return (
    <span className="art-stamp">
      <svg viewBox="0 0 120 120">
        <circle cx="60" cy="60" r="52" />
        <circle cx="60" cy="60" r="42" className="inner" />
      </svg>
      <b>STAMPED</b>
      <small>{authorities.length} TSAs</small>
    </span>
  );
};

export const Overview: React.FC<OverviewProps> = props => {
  const { preparation } = props;
  return (
    <section id="overview">
      <div className="wrap">
        <div className="sh">
          <div className="kicker">{copy.overview.kicker}</div>
          <h2>{copy.overview.h2}</h2>
          <p>{copy.overview.subtitle}</p>
        </div>

        <div className="facts">
          {copy.overview.facts.map((fact, i) => (
            <div key={i} className="panel fact">
              <div className="fact-art" aria-hidden="true">
                <FactArt index={i} {...props} />
              </div>
              <div className="n">
                {fact.value}
                {fact.total && <small>{fact.total}</small>}
              </div>
              <h3>{fact.title}</h3>
              <span className="chip h">{fact.chip}</span>
            </div>
          ))}
        </div>

        <div className="split">
          <div className="panel yes">
            <h3>
              <span className="chip m">✓</span> {copy.overview.shownTitle}
            </h3>
            <ul>
              {copy.overview.shownItems.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="panel no">
            <h3>
              <span className="chip h">○</span> {copy.overview.notShownTitle}
            </h3>
            <ul>
              {copy.overview.notShownItems.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="counts">
          <div>
            <b>{preparation.articles.toLocaleString()}</b>
            <span>Articles</span>
          </div>
          <div>
            <b>{preparation.wikipediaTargets.toLocaleString()}</b>
            <span>Wikipedia targets</span>
          </div>
          <div>
            <b>{preparation.conversations.toLocaleString()}</b>
            <span>Conversations</span>
          </div>
          <div>
            <b>{preparation.trainingLoopLines}</b>
            <span>Training loop</span>
          </div>
        </div>
      </div>
    </section>
  );
};
