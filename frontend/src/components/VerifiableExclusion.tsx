import React from 'react';
import { copy } from '../content/copy';
import { Explain } from './Explain';
import type { ExclusionData } from '../data/contracts';

interface VerifiableExclusionProps {
  exclusion: ExclusionData;
}

const signed = (n: number) => (n > 0 ? `+${n.toFixed(4)}` : n.toFixed(4));

export const VerifiableExclusion: React.FC<VerifiableExclusionProps> = ({ exclusion }) => {
  const c = copy.exclusion;
  return (
    <section id="exclusion">
      <h2>{c.kicker}</h2>
      <p className="sub">
        {c.h2} Removed article: <b>{exclusion.removedArticle}</b>.
      </p>
      <Explain label="Why this matters">{c.explain}</Explain>

      <table className="wikitable">
        <caption>Loss before and after the removal</caption>
        <thead>
          <tr>
            <th scope="col">Measured on</th>
            <th scope="col">Original model</th>
            <th scope="col">Fork</th>
            <th scope="col">Change</th>
            <th scope="col">Reading</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">{c.removedArticleLabel.replace('Loss on ', '')}</th>
            <td className="num">{exclusion.originalRemovedLoss.toFixed(4)}</td>
            <td className="num">{exclusion.forkRemovedLoss.toFixed(4)}</td>
            <td className="num">{signed(exclusion.removedLossDelta)}</td>
            <td>{c.removedArticleNote}</td>
          </tr>
          <tr>
            <th scope="row">{c.controlArticlesLabel.replace('Loss on ', '')}</th>
            <td className="num">{exclusion.originalControlLossMean.toFixed(4)}</td>
            <td className="num">{exclusion.forkControlLossMean.toFixed(4)}</td>
            <td className="num">{signed(exclusion.controlLossDelta)}</td>
            <td>{c.controlArticlesNote}</td>
          </tr>
        </tbody>
      </table>

      <div className="cols ruled">
        <div>
          <h3>{c.whyHoldsTitle}</h3>
          <ul className="ticks">
            {c.whyHoldsItems.map(item => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
        <div>
          <h3>{c.limitsTitle}</h3>
          <ul className="circles">
            {c.limitsItems.map(item => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </div>
      <p className="note">
        {c.evidenceLabel} <code>{exclusion.evidencePath}</code>
      </p>
    </section>
  );
};
