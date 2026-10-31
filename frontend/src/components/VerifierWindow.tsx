import React, { useEffect, useState } from 'react';
import type { EvidenceData } from '../data/contracts';

interface VerifierWindowProps {
  evidence: EvidenceData;
  // when true the answer types itself out; until then only the question shows
  play: boolean;
  // skip the typing and show the whole answer
  instant: boolean;
}

const shortHash = (hash: string) => `${hash.slice(0, 10)}…`;

// A verifier "app window" filled with the real evidence. The same facts are on
// the page below, so the window is decorative for screen readers.
export const VerifierWindow: React.FC<VerifierWindowProps> = ({ evidence, play, instant }) => {
  const { verifications, chain, anchor } = evidence;
  const segments = verifications.coverage.length;
  const matched = verifications.coverage.filter(c => c.projectResult === 'MATCH').length;
  const lastSegment = verifications.coverage[segments - 1];
  const gpu = lastSegment?.checks[0]?.gpu ?? '';
  const finalBoundary = chain[chain.length - 1];
  const stampDate = anchor.timestamp.slice(0, 10);

  const answer =
    `${matched} of ${segments} segments were recomputed bit for bit and matched` +
    `${gpu ? ` on ${gpu}` : ''}. No one outside the project has checked yet.`;

  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!play || instant) return;
    const timer = window.setInterval(() => {
      setCount(n => {
        if (n >= answer.length) {
          window.clearInterval(timer);
          return n;
        }
        return n + 2;
      });
    }, 22);
    return () => window.clearInterval(timer);
  }, [play, instant, answer]);

  const typed = instant ? answer : answer.slice(0, count);

  return (
    <div className="appwin" aria-hidden="true">
      <div className="aw-side">
        <div className="aw-brand">
          <img src="brand/logo-dark.svg" width="18" height="18" alt="" />
          <b>OVL Verifier</b>
        </div>
        <div className="aw-search">
          <span>Ask about the run…</span>
          <kbd>⌘K</kbd>
        </div>
        <ul className="aw-nav">
          <li>Overview</li>
          <li>Coverage</li>
          <li>Timeline</li>
          <li>Exclusion</li>
          <li>Downloads</li>
          <li className="on">Assistant</li>
        </ul>
        <div className="aw-group">Checkers</div>
        <ul className="aw-checkers">
          <li>
            <i className="live" />
            Project <span>{matched} / {segments}</span>
          </li>
          <li>
            <i />
            Independent <span>{verifications.independentChecksCount}</span>
          </li>
        </ul>
      </div>

      <div className="aw-main">
        <div className="aw-title">Assistant</div>
        <div className="aw-chat">
          <div className="aw-q">Has every training segment been recomputed?</div>
          <div className="aw-a">
            <div className="aw-meta">
              ✓ Read {chain.length} chain entries and {verifications.totalChecks} checks
            </div>
            <p>
              {typed}
              <span className="aw-caret" />
            </p>
          </div>
        </div>
        <div className="aw-input">
          <span>Ask a follow-up</span>
          <i>→</i>
        </div>
      </div>

      <div className="aw-sources">
        <div className="aw-title">Sources</div>
        <div className="aw-src">
          <small>1 · verifications.jsonl</small>
          <b>Segment {lastSegment?.segment} · ✓ MATCH</b>
          <span>{lastSegment?.checks[0]?.checker}</span>
        </div>
        <div className="aw-src">
          <small>2 · chain.jsonl</small>
          <b>
            Boundary {finalBoundary.index} · {finalBoundary.kind}
          </b>
          <span>state {shortHash(finalBoundary.state)}</span>
        </div>
        <div className="aw-src">
          <small>3 · anchor.json</small>
          <b>Stamped {stampDate}</b>
          <span>{anchor.authorities.join(' · ')}</span>
        </div>
      </div>
    </div>
  );
};
