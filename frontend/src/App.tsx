import React, { useEffect, useState } from 'react';
import type { EvidenceData } from './data/contracts';
import { validateEvidence, SnapshotError } from './data/validate';
import { Header } from './components/Header';
import { Hero } from './components/Hero';
import { Story } from './components/Story';
import { ChainQuote } from './components/ChainQuote';
import { Overview } from './components/Overview';
import { CheckLadder } from './components/CheckLadder';
import { Verification } from './components/Verification';
import { TrainingTimeline } from './components/TrainingTimeline';
import { VerifiableExclusion } from './components/VerifiableExclusion';
import { Downloads } from './components/Downloads';
import { Faq } from './components/Faq';
import { CtaCard } from './components/CtaCard';
import { Footer } from './components/Footer';
import { Band } from './components/Band';
import { MiniBar } from './components/MiniBar';

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
      <div className="wrap" style={{ padding: '64px 32px' }}>
        <div className="panel" role="alert">
          <h2>Snapshot unavailable or invalid</h2>
          <p style={{ marginTop: '12px', color: 'var(--miss)' }}>{error}</p>
          <button
            type="button"
            className="btn"
            style={{ marginTop: '24px' }}
            onClick={() => window.location.reload()}
          >
            Retry loading snapshot
          </button>
        </div>
      </div>
    );
  }

  if (!evidence) {
    return (
      <div className="wrap" style={{ padding: '64px 32px', textAlign: 'center' }}>
        <p style={{ color: 'var(--ink-3)' }}>Loading display snapshot…</p>
      </div>
    );
  }

  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>

      <Header />

      <main id="main">
        <Hero evidence={evidence} />
        <Story evidence={evidence} />
        <ChainQuote chain={evidence.chain} anchor={evidence.anchor} />
        <Overview
          preparation={evidence.preparation}
          verifications={evidence.verifications}
          authorities={evidence.anchor.authorities}
        />
        <CheckLadder />
        <Band>
          <Verification
            verifications={evidence.verifications}
            hardwareMap={evidence.hardwareMap}
          />
          <TrainingTimeline chain={evidence.chain} />
        </Band>
        <VerifiableExclusion exclusion={evidence.exclusion} />
        <Downloads releaseHashes={evidence.releaseHashes} />
        <Faq />
        <CtaCard />
      </main>

      <MiniBar />
      <Footer revision={evidence.revision} />
    </>
  );
};
export default App;
