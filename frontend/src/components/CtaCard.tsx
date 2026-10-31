import React from 'react';
import { copy } from '../content/copy';

// A closing line: what to do next, with the real cost of doing it.
export const CtaCard: React.FC = () => (
  <section className="cta-sec">
    <div className="wrap">
      <div className="cta-card">
        <div>
          <h2>{copy.ctaCard.h2}</h2>
          <p>{copy.ctaCard.desc}</p>
        </div>
        <div className="cta">
          <a className="btn primary" href={copy.ctaCard.primaryHref}>
            {copy.ctaCard.primary} <span aria-hidden="true">→</span>
          </a>
          <a className="btn" href={copy.ctaCard.secondaryHref}>
            {copy.ctaCard.secondary}
          </a>
        </div>
      </div>
    </div>
  </section>
);
