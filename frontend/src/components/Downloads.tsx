import React from 'react';
import { copy } from '../content/copy';
import { CopyButton } from './CopyButton';

interface DownloadsProps {
  releaseHashes: Record<string, string>;
}

export const Downloads: React.FC<DownloadsProps> = ({ releaseHashes }) => {
  return (
    <section id="downloads">
      <div className="wrap">
        <div className="sh">
          <div className="kicker">{copy.downloads.kicker}</div>
          <h2>{copy.downloads.h2}</h2>
        </div>

        <div className="files">
          {copy.downloads.files.map((file, i) => {
            const hash = releaseHashes[file.name] || 'Hash recorded in release';
            return (
              <div key={i} className="file">
                <div>
                  <b>{file.name}</b>
                  <div className="src">{file.source}</div>
                </div>

                <div className="h">
                  <span title={hash}>{hash}</span>
                  <div style={{ marginTop: '4px' }}>
                    <CopyButton textToCopy={hash} label="SHA-256" />
                  </div>
                </div>

                <a
                  className="btn"
                  href={file.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  {file.actionLabel}
                </a>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
};

