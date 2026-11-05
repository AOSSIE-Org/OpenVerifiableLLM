import React from 'react';
import { copy } from '../content/copy';

// The whole process as one line from data to recomputation. It runs across
// the page on wide screens and down the page on narrow ones.
export const HowItWorks: React.FC = () => (
  <section id="how" className="sec">
    <div className="sec-head">
      <h2>{copy.how.title}</h2>
      <p className="sub">{copy.how.intro}</p>
    </div>
    <ol className="flow">
      {copy.how.steps.map(step => (
        <li key={step.title}>
          <h3>{step.title}</h3>
          <p>{step.text}</p>
        </li>
      ))}
    </ol>
  </section>
);
