import { releaseIsDownloadable } from '../data/validate';
import { useSnapshot } from '../snapshot';

export function ReleasesPage() {
  const snapshot = useSnapshot();
  return (
    <article>
      <p className="kicker">Release</p>
      <h1>Model files</h1>
      <p>
        <strong>What.</strong> Two published models are planned: a base model trained on Wikipedia,
        then a chat model trained on top of it.
      </p>
      <p>
        <strong>Why.</strong> Without a public folder, a fixed version, and checksums, there are no
        weights for anyone else to download or check.
      </p>
      <p>
        <strong>Current status.</strong> Neither release is listed. Download and generate stay off.
        Listing files here later would still not mean complete replay had passed.
      </p>
      <div className="cards">
        {snapshot.releases.map((release) => {
          const ready = releaseIsDownloadable(release);
          const isChat = release.role === 'chat';
          return (
            <section className="card" key={release.id}>
              <p className="kicker">{isChat ? 'Chat model' : 'Base model'}</p>
              <h2>{release.title}</h2>
              <p>
                Status: <strong>{release.availability === 'not-released' ? 'not released' : release.availability}</strong>
              </p>
              {release.parentReleaseId ? (
                <p>Depends on the base Wikipedia model.</p>
              ) : (
                <p>Wikipedia-trained model, when published.</p>
              )}
              <p>{release.notes}</p>
              <details>
                <summary>Technical fields</summary>
                <p className="step-meta">
                  Repository: {release.repository ?? 'not set'}
                  <br />
                  Revision: {release.revision ?? 'not set'}
                  <br />
                  Release root: {release.releaseRoot ?? 'not set'}
                </p>
              </details>
              <p>
                <button type="button" disabled={!ready}>
                  Download
                </button>{' '}
                <button type="button" disabled>
                  Generate
                </button>
              </p>
            </section>
          );
        })}
      </div>
    </article>
  );
}
