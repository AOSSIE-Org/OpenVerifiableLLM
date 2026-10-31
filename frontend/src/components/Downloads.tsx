import React from 'react';
import { copy, links } from '../content/copy';
import { SectionHeading } from './Contents';
import { CopyButton } from './CopyButton';
import { Ref } from './Ref';

interface DownloadsProps {
  releaseHashes: Record<string, string>;
  revision: string;
}

export const Downloads: React.FC<DownloadsProps> = ({ releaseHashes, revision }) => (
  <section id="downloads" aria-labelledby="downloads-h">
    <SectionHeading id="downloads" />
    <p>
      {copy.downloads.intro}
      <Ref k="hashes" />
    </p>
    <div className="table-scroll" tabIndex={0} role="region" aria-label="Downloads">
      <table className="wikitable downloads">
        <thead>
          <tr>
            <th scope="col">File</th>
            <th scope="col">Source</th>
            <th scope="col">SHA-256</th>
          </tr>
        </thead>
        <tbody>
          {copy.downloads.files.map(f => {
            const hash = releaseHashes[f.name];
            const href = 'url' in f && f.url ? f.url : links.evidenceFile(revision, f.hfPath ?? '');
            const source = 'hfPath' in f ? `${f.source}, revision ${revision}` : f.source;
            return (
              <tr key={f.name}>
                <td>
                  <a href={href}>{f.name}</a>
                </td>
                <td>{source}</td>
                <td>
                  {hash ? (
                    <>
                      <code className="hash full">{hash}</code> <CopyButton textToCopy={hash} label="Copy hash" />
                    </>
                  ) : (
                    '—'
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
    <ul>
      <li>
        <a href={links.ollama}>{copy.downloads.ollama}</a>
      </li>
      <li>
        <a href={links.release}>GitHub release v1.0.1</a>
      </li>
      <li>
        <a href={links.evidence}>All evidence on Hugging Face</a>
      </li>
    </ul>
  </section>
);
