import React from 'react';
import { copy } from '../content/copy';
import type { ExclusionData } from '../data/contracts';

interface VerifiableExclusionProps {
  exclusion: ExclusionData;
}

export const VerifiableExclusion: React.FC<VerifiableExclusionProps> = ({ exclusion }) => {
  return (
    <section id="exclusion">
      <div className="wrap">
        <div className="sh">
          <div className="kicker">{copy.exclusion.kicker}</div>
          <h2>{copy.exclusion.h2}</h2>
          <p>
            The fork removed <b>{exclusion.removedArticle}</b> before training reached it. Everything else is identical.
          </p>
        </div>

        <div className="loss">
          <div className="panel lossrow">
            <div className="lbl">{copy.exclusion.removedArticleLabel}</div>
            <div className="vals">
              <span className="a">{exclusion.originalRemovedLoss.toFixed(4)}</span>
              <span aria-hidden="true">→</span>
              <span className="b">{exclusion.forkRemovedLoss.toFixed(4)}</span>
              <span className="d">
                {exclusion.removedLossDelta > 0 ? `+${exclusion.removedLossDelta.toFixed(4)}` : exclusion.removedLossDelta.toFixed(4)}
              </span>
            </div>
            <div className="sub" style={{ margin: '10px 0 0' }}>
              {copy.exclusion.removedArticleNote}
            </div>
          </div>

          <div className="panel lossrow">
            <div className="lbl">{copy.exclusion.controlArticlesLabel}</div>
            <div className="vals">
              <span className="a">{exclusion.originalControlLossMean.toFixed(4)}</span>
              <span aria-hidden="true">→</span>
              <span className="b">{exclusion.forkControlLossMean.toFixed(4)}</span>
              <span className="d">
                {exclusion.controlLossDelta > 0 ? `+${exclusion.controlLossDelta.toFixed(4)}` : exclusion.controlLossDelta.toFixed(4)}
              </span>
            </div>
            <div className="sub" style={{ margin: '10px 0 0' }}>
              {copy.exclusion.controlArticlesNote}
            </div>
          </div>
        </div>

        <div className="split">
          <div className="panel yes">
            <h3>{copy.exclusion.whyHoldsTitle}</h3>
            <ul>
              {copy.exclusion.whyHoldsItems.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="panel no">
            <h3>{copy.exclusion.limitsTitle}</h3>
            <ul>
              {copy.exclusion.limitsItems.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        </div>

        <p className="sub" style={{ marginTop: '16px' }}>
          {copy.exclusion.evidenceLabel} <code>{exclusion.evidencePath}</code>
        </p>
      </div>
    </section>
  );
};

