import React from 'react';
import { copy } from '../content/copy';
import type { EvidenceData } from '../data/contracts';
import { SectionHeading } from './Contents';
import { Ref } from './Ref';

const fmt = (n: number) => n.toLocaleString('en-US');

export const Overview: React.FC<{ evidence: EvidenceData }> = ({ evidence }) => {
  const p = evidence.preparation;
  const refs = [<Ref key="v" k="verifications" />, <Ref key="p" k="preparation" />, <Ref key="a" k="anchor" />];
  return (
    <section id="overview" aria-labelledby="overview-h">
      <SectionHeading id="overview" />
      <p>What has been shown so far:</p>
      <ul>
        {copy.overview.facts.map((f, i) => (
          <li key={f}>
            {f}
            {refs[i]}
          </li>
        ))}
      </ul>

      <h3>{copy.overview.limitsTitle}</h3>
      <ul>
        {copy.overview.limits.map(l => (
          <li key={l}>{l}</li>
        ))}
      </ul>

      <table className="wikitable">
        <caption>
          {copy.overview.countsCaption}
          <Ref k="preparation" />
        </caption>
        <tbody>
          <tr>
            <th scope="row">Wikipedia articles</th>
            <td className="num">{fmt(p.articles)}</td>
          </tr>
          <tr>
            <th scope="row">Wikipedia targets</th>
            <td className="num">{fmt(p.wikipediaTargets)}</td>
          </tr>
          <tr>
            <th scope="row">Conversations</th>
            <td className="num">{fmt(p.conversations)}</td>
          </tr>
          <tr>
            <th scope="row">Prepared data ID</th>
            <td>
              <code className="hash full">{p.preparedDataId}</code>
            </td>
          </tr>
        </tbody>
      </table>
      <p className="note">{copy.overview.loopNote}</p>
    </section>
  );
};
