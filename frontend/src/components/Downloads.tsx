import React from 'react';
import { copy } from '../content/copy';
import { CopyButton } from './CopyButton';

interface DownloadsProps {
  releaseHashes: Record<string, string>;
}

export const Downloads: React.FC<DownloadsProps> = ({ releaseHashes }) => (
  <section id="downloads">
    <h2>{copy.downloads.kicker}</h2>
    <p className="sub">{copy.downloads.intro}</p>

    <table className="wikitable files">
      <thead>
        <tr>
          <th scope="col">File</th>
          <th scope="col">Source</th>
          <th scope="col">SHA-256</th>
        </tr>
      </thead>
      <tbody>
        {copy.downloads.files.map(file => {
          const hash = releaseHashes[file.name];
          return (
            <tr key={file.name}>
              <th scope="row">
                <a href={file.url}>
                  <code>{file.name}</code>
                </a>
              </th>
              <td>{file.source}</td>
              <td className="hash">
                {hash ? (
                  <>
                    <code>{hash}</code> <CopyButton textToCopy={hash} label="Copy hash" />
                  </>
                ) : (
                  'note' in file && file.note
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  </section>
);
