import React, { useEffect, useState } from 'react';
import type { EvidenceData } from './data/contracts';
import { validateEvidence, SnapshotError } from './data/validate';
import { Header } from './components/Header';
import { Contents } from './components/Contents';
import { Hero } from './components/Hero';
import { HowItWorks } from './components/HowItWorks';
import { Results } from './components/Results';
import { TrainingHistory } from './components/TrainingHistory';
import { CheckLadder } from './components/CheckLadder';
import { Downloads } from './components/Downloads';
import { Faq } from './components/Faq';
import { Footer } from './components/Footer';

export const App: React.FC = () => {
  const [evidence, setEvidence] = useState<EvidenceData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadEvidence() {
      try {
        const res = await fetch('data/evidence.json');
        if (!res.ok) {
          throw new SnapshotError(`Failed to load evidence.json: HTTP ${res.status}`);
        }
        const json = await res.json();
        const validated = validateEvidence(json);
        if (active) {
          setEvidence(validated);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    }

    loadEvidence();
    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return (
      <div className="page-msg" role="alert">
        <h2>Snapshot unavailable or invalid</h2>
        <p className="err">{error}</p>
        <button type="button" className="btn" onClick={() => window.location.reload()}>
          Retry loading snapshot
        </button>
      </div>
    );
  }

  if (!evidence) {
    return (
      <div className="page-msg">
        <p>Loading display snapshot…</p>
      </div>
    );
  }

  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>

      <Header />
      <Contents />

      <main id="main" className="page">
        <Hero evidence={evidence} />
        <HowItWorks />
        <Results
          verifications={evidence.verifications}
          hardwareMap={evidence.hardwareMap}
          preparation={evidence.preparation}
        />
        <TrainingHistory chain={evidence.chain} exclusion={evidence.exclusion} />
        <CheckLadder />
        <Downloads releaseHashes={evidence.releaseHashes} />
        <Faq />
      </main>

      <Footer revision={evidence.revision} />
    </>
  );
};
export default App;
