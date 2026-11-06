import React from 'react';
import { claimsFor } from '../content/claims';
import { copy } from '../content/copy';
import type { Facts } from '../data/facts';

export const Faq: React.FC<{ facts: Facts }> = ({ facts }) => (
  <section id="faq" className="sec">
    <div className="sec-head">
      <h2>{copy.faq.h2}</h2>
      <p className="sub">
        {copy.faq.stillTitle} <a href={copy.faq.discordHref}>{copy.faq.discord}</a> or{' '}
        <a href={copy.faq.issueHref}>{copy.faq.issue.toLowerCase()}</a>.
      </p>
    </div>
    <dl className="qa">
      {claimsFor(facts).faq.map(item => (
        <div key={item.q}>
          <dt>{item.q}</dt>
          <dd>{item.a}</dd>
        </div>
      ))}
    </dl>
  </section>
);
