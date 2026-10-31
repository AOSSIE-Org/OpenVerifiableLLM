import React from 'react';
import { copy } from '../content/copy';
import type { ExclusionData } from '../data/contracts';
import { SectionHeading } from './Contents';
import { Ref } from './Ref';

const signed = (d: number) => (d > 0 ? `+${d.toFixed(4)}` : d.toFixed(4));

export const VerifiableExclusion: React.FC<{ exclusion: ExclusionData }> = ({ exclusion: e }) => (
  <section id="exclusion" aria-labelledby="exclusion-h">
    <SectionHeading id="exclusion" />
    <p>
      {copy.exclusion.intro} The removed article: <b>{e.removedArticle}</b>.
    </p>

    <table className="wikitable">
      <caption>
        Loss before and after the exclusion
        <Ref k="loo" />
      </caption>
      <thead>
        <tr>
          <th scope="col">Measured on</th>
          <th scope="col">Original model</th>
          <th scope="col">Fork without the article</th>
          <th scope="col">Change</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th scope="row">The removed article</th>
          <td className="num">{e.originalRemovedLoss.toFixed(4)}</td>
          <td className="num">{e.forkRemovedLoss.toFixed(4)}</td>
          <td className="num">{signed(e.removedLossDelta)}</td>
        </tr>
        <tr>
          <th scope="row">{e.controlArticlesCount} control articles (mean)</th>
          <td className="num">{e.originalControlLossMean.toFixed(4)}</td>
          <td className="num">{e.forkControlLossMean.toFixed(4)}</td>
          <td className="num">{signed(e.controlLossDelta)}</td>
        </tr>
      </tbody>
    </table>
    <p>
      Loss rose on the removed article, which the fork never saw, and is practically unchanged on the control articles.
    </p>

    <h3>{copy.exclusion.whyTitle}</h3>
    <ul>
      {copy.exclusion.why.map(w => (
        <li key={w}>{w}</li>
      ))}
    </ul>

    <h3>{copy.exclusion.limitsTitle}</h3>
    <ul>
      {copy.exclusion.limits.map(l => (
        <li key={l}>{l}</li>
      ))}
    </ul>

    <p className="note">
      Evidence folder: <code>{e.evidencePath}</code>. {copy.exclusion.folderNote}
    </p>
  </section>
);
