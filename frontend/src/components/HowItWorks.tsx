import React from 'react';
import { claimsFor } from '../content/claims';
import { copy } from '../content/copy';
import type { Facts } from '../data/facts';

// The whole process as one line from data to recomputation. It runs across
// the page on wide screens and down the page on narrow ones.
export const HowItWorks: React.FC<{ facts: Facts }> = ({ facts }) => (
  <section id="how" className="sec">
    <div className="sec-head">
      <h2>{copy.how.title}</h2>
      <p className="sub">{copy.how.intro}</p>
    </div>
    <ol className="flow">
      {claimsFor(facts).how.map(step => (
        <li key={step.title}>
          <h3>{step.title}</h3>
          <p>{step.text}</p>
        </li>
      ))}
    </ol>
  </section>
);
