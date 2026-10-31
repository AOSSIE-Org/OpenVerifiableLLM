import React, { useEffect, useState } from 'react';
import type { EvidenceData } from './data/contracts';
import { validateEvidence, SnapshotError } from './data/validate';
import { copy } from './content/copy';
import { Header } from './components/Header';
import { Contents } from './components/Contents';
import { Infobox } from './components/Infobox';
import { Overview } from './components/Overview';
import { CheckYourself } from './components/CheckYourself';
import { Verification } from './components/Verification';
import { TrainingTimeline } from './components/TrainingTimeline';
import { VerifiableExclusion } from './components/VerifiableExclusion';
import { Downloads } from './components/Downloads';
import { References, Terms } from './components/References';
import { Footer } from './components/Footer';

export const App: React.FC = () => {
  const [evidence, setEvidence] = useState<EvidenceData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const res = await fetch('data/evidence.json');
        if (!res.ok) throw new SnapshotError(`Failed to load evidence.json: HTTP ${res.status}`);
        const validated = validateEvidence(await res.json());
        if (active) setEvidence(validated);
      } catch (err) {
        if (active) setError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <Header />
      <div className="page" id="top">
        {evidence && <Contents />}
        <main id="main">
          <h1>{copy.site.title}</h1>
          {error && (
            <div className="error-box" role="alert">
              <h2>Snapshot unavailable or invalid</h2>
              <p>
                The page shows no numbers unless the evidence snapshot loads and passes validation. Reason: {error}
              </p>
              <button type="button" onClick={() => window.location.reload()}>
                Retry loading snapshot
              </button>
            </div>
          )}
          {!error && !evidence && <p className="note">Loading evidence snapshot…</p>}
          {evidence && (
            <>
              <p className="tagline">
                {copy.lead.tagline} <code>{evidence.revision}</code>
              </p>
              <Infobox evidence={evidence} />
              <p>
                <b>{copy.site.title}</b> {copy.lead.body}
              </p>
              <p>{copy.lead.status}</p>
              <p className="hatnote" role="note">
                {copy.lead.caution}
              </p>
              <Overview evidence={evidence} />
              <CheckYourself />
              <Verification verifications={evidence.verifications} hardwareMap={evidence.hardwareMap} />
              <TrainingTimeline chain={evidence.chain} />
              <VerifiableExclusion exclusion={evidence.exclusion} />
              <Downloads releaseHashes={evidence.releaseHashes} revision={evidence.revision} />
              <Terms />
              <References evidence={evidence} />
            </>
          )}
        </main>
      </div>
      {evidence && <Footer revision={evidence.revision} fetchedAt={evidence.fetchedAt} />}
    </>
  );
};
export default App;
