import React, { useState } from 'react';
import { copy } from '../content/copy';
import type { EvidenceData } from '../data/contracts';
import { usePrefersReducedMotion } from '../hooks/usePrefersReducedMotion';
import { useNarrow, useScrollProgress } from '../hooks/useScrollProgress';
import { VerifierWindow } from './VerifierWindow';

interface HeroProps {
  evidence: EvidenceData;
}

// The landing: a short centred headline, and under it a laptop with its lid
// half closed. While the tall hero scrolls, the lid opens and the laptop rises
// and grows until its screen fills the view, while the headline fades.
// On narrow screens and with reduced motion it is a plain stacked layout.
export const Hero: React.FC<HeroProps> = ({ evidence }) => {
  const reduced = usePrefersReducedMotion();
  const narrow = useNarrow(900);
  const animated = !reduced && !narrow;
  const [opened, setOpened] = useState(false);

  const ref = useScrollProgress<HTMLDivElement>(animated, p => {
    if (p > 0.8) setOpened(true);
  });

  const { verifications } = evidence;
  const matched = verifications.coverage.filter(c => c.projectResult === 'MATCH').length;

  return (
    <div className={`hx${animated ? '' : ' hx-static'}`} id="top" ref={ref}>
      <div className="hx-stage">
        <div className="wrap hx-grid">
          <div className="hx-copy">
            <div className="kicker">{copy.hero.kicker}</div>
            <h1>
              {copy.hero.h1Prefix}
              <span className="accent">{copy.hero.h1Accent}</span>
              {copy.hero.h1Suffix}
            </h1>
            <p className="hx-lede">{copy.hero.ledeShort}</p>
            <div className="cta">
              <a className="btn primary" href="#check">
                {copy.hero.verifyCta} <span aria-hidden="true">→</span>
              </a>
              <a className="btn" href="#overview">
                {copy.hero.evidenceCta}
              </a>
            </div>
            <p className="hx-meta">
              <span>
                {matched}/{verifications.coverage.length} segments matched
              </span>
              <span>GPL-3.0</span>
              <a href="https://aossie.org">
                {copy.hero.poweredBy} AOSSIE <span aria-hidden="true">↗</span>
              </a>
            </p>
          </div>
        </div>

        <div className="hx-device">
          <div className="lap">
            <div className="lap-lid">
              <span className="lap-cam" aria-hidden="true" />
              <div className="lap-screen">
                <VerifierWindow evidence={evidence} play={opened || !animated} instant={reduced} />
              </div>
            </div>
            <div className="lap-base" aria-hidden="true" />
          </div>
        </div>
      </div>
    </div>
  );
};
