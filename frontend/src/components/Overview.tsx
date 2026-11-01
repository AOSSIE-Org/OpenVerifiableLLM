import React from 'react';
import { copy } from '../content/copy';
import { InShort } from './InShort';
import type { PreparationData } from '../data/contracts';

interface OverviewProps {
  preparation: PreparationData;
}

export const Overview: React.FC<OverviewProps> = ({ preparation }) => {
  const o = copy.overview;
  return (
    <section id="overview">
      <h2>{o.kicker}</h2>
      <InShort>
        {o.h2} {o.subtitle}
      </InShort>

      <div className="split">
        <div>
          <table className="wikitable">
            <caption>Headline results</caption>
            <thead>
              <tr>
                <th scope="col">Claim</th>
                <th scope="col">Result</th>
                <th scope="col">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {o.facts.map(fact => (
                <tr key={fact.title}>
                  <th scope="row">{fact.title}</th>
                  <td className="num">
                    {fact.value}
                    {fact.total}
                  </td>
                  <td>{fact.chip}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <table className="wikitable">
            <caption>Training data</caption>
            <tbody>
              <tr>
                <th scope="row">Articles</th>
                <td className="num">{preparation.articles.toLocaleString('en-US')}</td>
              </tr>
              <tr>
                <th scope="row">Wikipedia targets</th>
                <td className="num">{preparation.wikipediaTargets.toLocaleString('en-US')}</td>
              </tr>
              <tr>
                <th scope="row">Conversations</th>
                <td className="num">{preparation.conversations.toLocaleString('en-US')}</td>
              </tr>
              <tr>
                <th scope="row">Training loop</th>
                <td className="num">{preparation.trainingLoopLines}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="cols stack">
          <div>
            <h3>{o.shownTitle}</h3>
            <ul className="ticks">
              {o.shownItems.map(item => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div>
            <h3>{o.notShownTitle}</h3>
            <ul className="circles">
              {o.notShownItems.map(item => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </section>
  );
};
