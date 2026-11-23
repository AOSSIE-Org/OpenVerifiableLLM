import React, { useEffect, useState } from 'react';
import type { EvidenceData } from './data/contracts';
import { type Facts, summarize } from './data/facts';
import { validateEvidence, SnapshotError } from './data/validate';
import { ErrorBoundary, SnapshotProblem } from './components/ErrorBoundary';
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

export interface AppProps {
  initialData?: { evidence: EvidenceData; facts: Facts };
}

export const App: React.FC<AppProps> = ({ initialData }) => {
  const [loaded, setLoaded] = useState<{ evidence: EvidenceData; facts: Facts } | null>(
    () => initialData ?? null,
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;

    async function loadEvidence() {
      let res: Response;
      try {
        res = await fetch('data/evidence.json');
        if (!res.ok) {
          throw new SnapshotError(`Failed to load evidence.json: HTTP ${res.status}`);
        }
      } catch (err) {
        // A network failure says nothing about the evidence. If the prerendered report is
        // already up, and its embedded snapshot passed validation, keep showing it.
        if (active && !initialData) {
          setError(err instanceof Error ? err.message : String(err));
        }
        return;
      }
      // Evidence that arrives but fails validation still replaces the page with the error.
      try {
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
  }, [initialData]);

  if (error) {
    return <SnapshotProblem message={error} />;
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
    <ErrorBoundary>
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
    </ErrorBoundary>
  );
};
export default App;
