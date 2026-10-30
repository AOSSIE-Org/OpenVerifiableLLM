import React from 'react';
import { copy } from '../content/copy';
import type { PreparationData } from '../data/contracts';

interface OverviewProps {
  preparation: PreparationData;
}

export const Overview: React.FC<OverviewProps> = ({ preparation }) => {
  return (
    <section id="overview">
      <div className="wrap">
        <div className="sh">
          <div className="kicker">{copy.overview.kicker}</div>
          <h2>{copy.overview.h2}</h2>
        </div>

        <div className="facts">
          {copy.overview.facts.map((fact, i) => (
            <div key={i} className="panel fact">
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

