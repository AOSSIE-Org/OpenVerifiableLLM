import React from 'react';
import { copy } from '../content/copy';
import type { EvidenceData } from '../data/contracts';

interface LeadProps {
  evidence: EvidenceData;
}

// The article title, a one-line byline, an infobox and the lead paragraphs.
export const Lead: React.FC<LeadProps> = ({ evidence }) => {
  const a = copy.article;
  const box = a.infobox;
  const { verifications, revision } = evidence;
  const segments = verifications.coverage.length;
  const matched = verifications.coverage.filter(c => c.projectResult === 'MATCH').length;

  return (
    <div className="lead" id="top">
      <h1>{a.title}</h1>
      <p className="byline">
        {a.from} · {a.kicker}
      </p>

      <table className="infobox">
        <caption>{box.caption}</caption>
        <tbody>
          <tr>
            <td colSpan={2} className="ib-logo">
              <img className="logo-l" src="brand/logo-light.svg" width="96" height="96" alt="" />
              <img className="logo-d" src="brand/logo-dark.svg" width="96" height="96" alt="" />
            </td>
          </tr>
          <tr>
            <th scope="row">Developer</th>
            <td>{box.developer}</td>
          </tr>
          <tr>
            <th scope="row">Release</th>
            <td>{box.release}</td>
          </tr>
          <tr>
            <th scope="row">Parameters</th>
            <td>{box.parameters}</td>
          </tr>
          <tr>
            <th scope="row">Training data</th>
            <td>{box.data}</td>
          </tr>
          <tr>
            <th scope="row">Segments</th>
            <td>{box.segments}</td>
          </tr>
          <tr>
            <th scope="row">Recomputed</th>
            <td>
              {matched} of {segments}, {box.checked}
            </td>
          </tr>
          <tr>
            <th scope="row">Independent checks</th>
            <td>{box.independent}</td>
          </tr>
          <tr>
            <th scope="row">Record stamped</th>
            <td>{box.stamped}</td>
          </tr>
          <tr>
            <th scope="row">Licence</th>
            <td>{box.license}</td>
          </tr>
          <tr>
            <th scope="row">Evidence</th>
            <td>
              <a href={copy.header.links[1].href}>
                revision <code>{revision}</code>
              </a>
            </td>
          </tr>
          <tr>
            <th scope="row">Source</th>
            <td>
              <a href={copy.header.links[0].href}>{box.source}</a>
            </td>
          </tr>
        </tbody>
      </table>

      <p>
        <b>{a.title}</b> (OVL) {a.lead[0]}
      </p>
      {a.lead.slice(1).map(p => (
        <p key={p}>{p}</p>
      ))}
    </div>
  );
};
