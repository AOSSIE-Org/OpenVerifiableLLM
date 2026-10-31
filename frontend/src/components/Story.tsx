import React, { useState } from 'react';
import { copy } from '../content/copy';
import type { EvidenceData } from '../data/contracts';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';
import { useNarrow, useScrollProgress } from '../hooks/useScrollProgress';

interface StoryProps {
  evidence: EvidenceData;
}

const steps = copy.story.steps;
const short = (hash: string, n = 12) => `${hash.slice(0, n)}…`;

// One preview per step, all drawn from the evidence file.
const Preview: React.FC<{ index: number; evidence: EvidenceData }> = ({ index, evidence }) => {
  const { verifications, chain, anchor, exclusion } = evidence;
  const last = chain[chain.length - 1];
  const gpu = verifications.coverage[verifications.coverage.length - 1]?.checks[0]?.gpu ?? '';

  switch (index) {
    case 0:
      return (
        <div className="pv-card">
          <div className="pv-head">
            Segment {verifications.coverage.length - 1} <span className="chip m">✓ MATCH</span>
          </div>
          <div className="pv-row">
            <small>Recorded state</small>
            <code>{short(last.state, 24)}</code>
          </div>
          <div className="pv-row">
            <small>Recomputed state</small>
            <code>{short(last.state, 24)}</code>
          </div>
          <p className="pv-note">A match means the two are the same bytes.</p>
        </div>
      );
    case 1:
      return (
        <div className="pv-card">
          <div className="pv-head">
            {verifications.coverage.filter(c => c.projectResult === 'MATCH').length} /{' '}
            {verifications.coverage.length} segments matched
          </div>
          <div className="pv-tiles">
            {verifications.coverage.map(c => (
              <i key={c.segment} className={c.projectResult === 'MATCH' ? 'ok' : 'bad'} />
            ))}
          </div>
          <p className="pv-note">{gpu}</p>
        </div>
      );
    case 2:
      return (
        <div className="pv-stack">
          {chain.slice(-3).map(b => (
            <div className="pv-card pv-link" key={b.index}>
              <div className="pv-head">
                Boundary {b.index} <small>{b.kind}</small>
              </div>
              <div className="pv-row">
                <small>previous</small>
                <code>{b.previous ? short(b.previous) : 'none'}</code>
              </div>
            </div>
          ))}
        </div>
      );
    case 3:
      return (
        <div className="pv-card">
          <div className="pv-head">anchor.json</div>
          <div className="pv-row">
            <small>Stamped</small>
            <span>{anchor.timestamp}</span>
          </div>
          <div className="pv-row">
            <small>Chain head</small>
            <code>{short(anchor.chain_head_sha256, 20)}</code>
          </div>
          <div className="pv-chips">
            {anchor.authorities.map(a => (
              <span className="chip h" key={a}>
                {a}
              </span>
            ))}
          </div>
        </div>
      );
    case 4: {
      const max = Math.max(exclusion.forkRemovedLoss, exclusion.originalControlLossMean) * 1.05;
      const bar = (v: number) => `${(v / max) * 100}%`;
      return (
        <div className="pv-card">
          <div className="pv-head">{exclusion.removedArticle}</div>
          <div className="pv-bars">
            <small>Removed article</small>
            <span style={{ width: bar(exclusion.originalRemovedLoss) }}>{exclusion.originalRemovedLoss.toFixed(4)}</span>
            <span className="up" style={{ width: bar(exclusion.forkRemovedLoss) }}>
              {exclusion.forkRemovedLoss.toFixed(4)}
            </span>
            <small>{exclusion.controlArticlesCount} control articles</small>
            <span style={{ width: bar(exclusion.originalControlLossMean) }}>
              {exclusion.originalControlLossMean.toFixed(4)}
            </span>
            <span style={{ width: bar(exclusion.forkControlLossMean) }}>{exclusion.forkControlLossMean.toFixed(4)}</span>
          </div>
        </div>
      );
    }
    default:
      return (
        <div className="pv-card">
          {copy.check.ladder.map(step => (
            <div className="pv-row pv-step" key={step.time}>
              <b>{step.time}</b>
              <span>{step.title}</span>
              <small>{step.meta}</small>
            </div>
          ))}
        </div>
      );
  }
};

const Points: React.FC<{ index: number }> = ({ index }) => (
  <ul className="st-points">
    {steps[index].points.map(p => (
      <li key={p}>{p}</li>
    ))}
  </ul>
);

// The six steps drawn as a checkpoint chain: linked nodes across the top
// that fill in as you scroll, with the current step and its preview below.
// On narrow screens and with reduced motion the chain runs down the page.
export const Story: React.FC<StoryProps> = ({ evidence }) => {
  const reduced = usePrefersReducedMotion();
  const narrow = useNarrow(900);
  const pinned = !reduced && !narrow;
  const [active, setActive] = useState(0);

  const ref = useScrollProgress<HTMLElement>(pinned, p => {
    setActive(Math.min(steps.length - 1, Math.floor(p * steps.length)));
  });

  const header = (
    <div className="sh">
      <div className="kicker">{copy.story.kicker}</div>
      <h2>{copy.story.h2}</h2>
      <p>{copy.story.sub}</p>
    </div>
  );

  if (!pinned) {
    return (
      <section id="how" className="story story-stacked">
        <div className="wrap">
          {header}
          <ol className="chain-v">
            {steps.map((step, i) => (
              <li key={step.title}>
                <span className="node" aria-hidden="true" />
                <h3>{step.title}</h3>
                <Points index={i} />
                <div className="st-preview">
                  <Preview index={i} evidence={evidence} />
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>
    );
  }

  return (
    <section id="how" className="story" ref={ref} style={{ height: `${steps.length * 70 + 40}vh` }}>
      <div className="st-sticky">
        <div className="wrap">
          {header}

          <ol className="chain" style={{ '--done': active / (steps.length - 1) } as React.CSSProperties}>
            {steps.map((step, i) => (
              <li key={step.title} className={i === active ? 'on' : i < active ? 'done' : ''}>
                <span className="node" aria-hidden="true" />
                <span className="label">{step.title}</span>
              </li>
            ))}
          </ol>

          <div className="st-body">
            <div className="st-text" aria-live="polite">
              <div className="st-count">
                Checkpoint {String(active + 1).padStart(2, '0')} of {String(steps.length).padStart(2, '0')}
              </div>
              <h3>{steps[active].title}</h3>
              <Points index={active} />
              <a className="btn" href="#check">
                {copy.story.cta} <span aria-hidden="true">→</span>
              </a>
            </div>
            <div className="st-preview" aria-hidden="true">
              <div className="st-preview-in" key={active}>
                <Preview index={active} evidence={evidence} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
