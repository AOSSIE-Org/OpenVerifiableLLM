import { useData } from '../data/context';
import { Hash } from './Hash';
import { SourceChip } from './SourceChip';
import { L, V } from './text';

export function DownloadsList() {
  const { data } = useData();
  return (
    <>
      {data.downloads.map((rel, ri) => (
        <section key={rel.release} className="dl" aria-labelledby={`dl-${ri}`}>
          <h3 id={`dl-${ri}`}><L k={`release.${rel.release.split('/')[1]}`} /></h3>
          <ul className="files">
            {rel.files.map((f, fi) => (
              <li key={f.file} className="file">
                <span className="file-name"><V path={`downloads.${ri}.files.${fi}.file`} /></span>
                <Hash value={f.sha256} path={`downloads.${ri}.files.${fi}.sha256`} />
                <a className="btn btn-small" href={f.download} rel="noopener noreferrer" target="_blank">
                  <L k="download" />
                  <span className="sr">
                    {' '}
                    <L k="newtab" />
                  </span>
                </a>
                <SourceChip s={f.src} />
              </li>
            ))}
          </ul>
        </section>
      ))}
    </>
  );
}
