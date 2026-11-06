import React, { useEffect, useState } from 'react';
import type { EvidenceData } from './data/contracts';
import { type Facts, summarize } from './data/facts';
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
  const [loaded, setLoaded] = useState<{ evidence: EvidenceData; facts: Facts } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadEvidence() {
      try {
        const res = await fetch('data/evidence.json');
        if (!res.ok) {
          throw new SnapshotError(`Failed to load evidence.json: HTTP ${res.status}`);
        }
        const evidence = validateEvidence(await res.json());
        const facts = summarize(evidence);
        if (active) {
          setLoaded({ evidence, facts });
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
      <main className="page-msg" role="alert">
        <h1>Snapshot unavailable or invalid</h1>
        <p className="err">{error}</p>
        <button type="button" className="btn" onClick={() => window.location.reload()}>
          Retry loading snapshot
        </button>
      </main>
    );
  }

  if (!loaded) {
    return (
      <main className="page-msg" role="status">
        <p>Loading display snapshot…</p>
      </main>
    );
  }

  const { evidence, facts } = loaded;

  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>

      <Header evidenceHref={facts.evidenceUrl} />
      <Contents />

      <main id="main" className="page" tabIndex={-1}>
        <Hero facts={facts} />
        <HowItWorks facts={facts} />
        <Results
          facts={facts}
          coverage={evidence.verifications.coverage}
          hardwareMap={evidence.hardwareMap}
          preparation={evidence.preparation}
        />
        <TrainingHistory
          facts={facts}
          chain={evidence.chain}
          coverage={evidence.verifications.coverage}
          exclusion={evidence.exclusion}
        />
        <CheckLadder facts={facts} />
        <Downloads facts={facts} releaseHashes={evidence.releaseHashes} />
        <Faq facts={facts} />
      </main>

      <Footer facts={facts} />
    </>
  );
};
export default App;
