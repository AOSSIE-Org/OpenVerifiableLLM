import React from 'react';
import type { EvidenceData } from '../data/contracts';
import { Ref } from './Ref';

// The summary box at the top right, like a Wikipedia infobox.
export const Infobox: React.FC<{ evidence: EvidenceData }> = ({ evidence }) => {
  const { verifications: v, anchor, preparation: p } = evidence;
  const matched = v.coverage.filter(c => c.projectResult === 'MATCH').length;
  return (
    <aside className="infobox" aria-label="Summary">
      <table>
        <caption>OpenVerifiableLLM</caption>
        <tbody>
          <tr className="ib-section">
            <th colSpan={2} scope="colgroup">Model</th>
          </tr>
          <tr>
            <th scope="row">Parameters</th>
            <td>23M</td>
          </tr>
          <tr>
            <th scope="row">Training data</th>
            <td>
              English Wikipedia ({p.articles.toLocaleString('en-US')} articles), then{' '}
              {p.conversations.toLocaleString('en-US')} conversations
              <Ref k="preparation" />
            </td>
          </tr>
          <tr>
            <th scope="row">Release</th>
            <td>v1.0.1</td>
          </tr>
          <tr className="ib-section">
            <th colSpan={2} scope="colgroup">Evidence</th>
          </tr>
          <tr>
            <th scope="row">Segments</th>
            <td>
              {v.uniqueSegments} ({anchor.chain_entries} boundaries)
              <Ref k="chain" />
            </td>
          </tr>
          <tr>
            <th scope="row">Recomputed and matched</th>
            <td>
              {matched} of {v.uniqueSegments}, by the project
              <Ref k="verifications" />
            </td>
          </tr>
          <tr>
            <th scope="row">Independent checks</th>
            <td>{v.independentChecksCount}</td>
          </tr>
          <tr>
            <th scope="row">Reproducing hardware</th>
            <td>
              {[...new Set(evidence.hardwareMap.filter(h => h.result === 'MATCH').map(h => h.hardware))].join(', ')}
              <Ref k="hardware" />
            </td>
          </tr>
          <tr>
            <th scope="row">Record timestamped</th>
            <td>
              {anchor.timestamp}
              <br />
              by {anchor.authorities.join(' and ')}
              <Ref k="anchor" />
            </td>
          </tr>
          <tr>
            <th scope="row">Chain head</th>
            <td>
              <code className="hash" title={anchor.chain_head_sha256}>
                {anchor.chain_head_sha256.slice(0, 16)}…
              </code>
            </td>
          </tr>
          <tr>
            <th scope="row">Evidence revision</th>
            <td>
              <code>{evidence.revision}</code>
            </td>
          </tr>
        </tbody>
      </table>
    </aside>
  );
};
