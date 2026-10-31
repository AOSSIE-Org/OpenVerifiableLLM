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
        <div className="hx-waves" aria-hidden="true">
          <svg
            className="hx-waves-svg"
            viewBox="0 0 1440 800"
            fill="none"
            preserveAspectRatio="none"
          >
            <defs>
              <linearGradient id="wave-grad-1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="var(--s3)" stopOpacity="0" />
                <stop offset="25%" stopColor="var(--s3)" stopOpacity="0.22" />
                <stop offset="65%" stopColor="var(--s2)" stopOpacity="0.32" />
                <stop offset="100%" stopColor="var(--s3)" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="wave-grad-2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="var(--s3)" stopOpacity="0" />
                <stop offset="35%" stopColor="var(--s3)" stopOpacity="0.18" />
                <stop offset="75%" stopColor="var(--s2)" stopOpacity="0.25" />
                <stop offset="100%" stopColor="var(--s3)" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="wave-shade" x1="50%" y1="0%" x2="50%" y2="100%">
                <stop offset="0%" stopColor="var(--s3)" stopOpacity="0.04" />
                <stop offset="100%" stopColor="var(--s3)" stopOpacity="0" />
              </linearGradient>
            </defs>

            {/* Subtle shaded band between wave contours */}
            <path
              className="hx-wave-shade"
              d="M -40,320 C 320,220, 520,440, 840,340 C 1160,240, 1280,380, 1480,290 L 1480,470 C 1280,560, 1160,420, 840,520 C 520,620, 320,400, -40,500 Z"
              fill="url(#wave-shade)"
            />

            {/* Gentle wave contour lines */}
            <path
              className="hx-wave-path"
              d="M -40,240 C 320,150, 500,340, 820,250 C 1140,160, 1260,290, 1480,210"
              stroke="url(#wave-grad-2)"
              strokeWidth="1.2"
            />
            <path
              className="hx-wave-path"
              d="M -40,280 C 320,185, 510,390, 830,295 C 1150,200, 1270,335, 1480,250"
              stroke="url(#wave-grad-1)"
              strokeWidth="1.5"
            />
            <path
              className="hx-wave-path"
              d="M -40,320 C 320,220, 520,440, 840,340 C 1160,240, 1280,380, 1480,290"
              stroke="url(#wave-grad-1)"
              strokeWidth="1.8"
            />
            <path
              className="hx-wave-path"
              d="M -40,365 C 320,260, 530,485, 850,385 C 1170,285, 1290,425, 1480,335"
              stroke="url(#wave-grad-2)"
              strokeWidth="1.4"
            />
            <path
              className="hx-wave-path"
              d="M -40,415 C 320,305, 540,530, 860,430 C 1180,330, 1300,470, 1480,380"
              stroke="url(#wave-grad-2)"
              strokeWidth="1.2"
            />

            {/* Subtle harmonic counter-waves */}
            <path
              className="hx-wave-dash"
              d="M -40,380 C 360,480, 620,280, 920,400 C 1220,520, 1340,360, 1480,440"
              stroke="url(#wave-grad-2)"
              strokeWidth="1.2"
              strokeDasharray="6 4"
            />
            <path
              className="hx-wave-dash"
              d="M -40,430 C 360,530, 620,330, 920,450 C 1220,570, 1340,410, 1480,490"
              stroke="url(#wave-grad-2)"
              strokeWidth="1.0"
            />
          </svg>
        </div>
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
