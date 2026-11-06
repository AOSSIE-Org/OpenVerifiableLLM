import React from 'react';
import { claimsFor } from '../content/claims';
import { copy } from '../content/copy';
import type { Facts } from '../data/facts';
import { CopyButton } from './CopyButton';

interface DownloadsProps {
  facts: Facts;
  releaseHashes: Record<string, string>;
}

// The release files, each with its hash folded underneath.
export const Downloads: React.FC<DownloadsProps> = ({ facts, releaseHashes }) => (
  <section id="downloads" className="sec">
    <div className="sec-head">
      <h2>{copy.downloads.kicker}</h2>
      <p className="sub">{copy.downloads.intro}</p>
    </div>

    <ul className="files">
      {claimsFor(facts).downloads.map(file => {
        const hash = releaseHashes[file.name];
        return (
          <li key={file.name}>
            <a href={file.url}>
              <code>{file.name}</code>
            </a>
            <span className="source">{file.source}</span>
            {hash ? (
              <details className="more">
                <summary aria-label={`SHA-256 of ${file.name}`}>SHA-256</summary>
                <p className="hash">
                  <code>{hash}</code> <CopyButton textToCopy={hash} label="Copy hash" className="copy-btn inline" />
                </p>
              </details>
            ) : (
              file.note && <span className="source">{file.note}</span>
            )}
          </li>
        );
      })}
    </ul>
  </section>
);
