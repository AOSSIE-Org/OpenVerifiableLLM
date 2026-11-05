import React from 'react';
import { copy } from '../content/copy';
import { CopyButton } from './CopyButton';

interface DownloadsProps {
  releaseHashes: Record<string, string>;
}

// The release files, each with its hash folded underneath.
export const Downloads: React.FC<DownloadsProps> = ({ releaseHashes }) => (
  <section id="downloads" className="sec">
    <div className="sec-head">
      <h2>{copy.downloads.kicker}</h2>
      <p className="sub">{copy.downloads.intro}</p>
    </div>

    <ul className="files">
      {copy.downloads.files.map(file => {
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
              <span className="source">{'note' in file && file.note}</span>
            )}
          </li>
        );
      })}
    </ul>
  </section>
);
