import React from 'react';
import { copy } from '../content/copy';

// The idea behind the project in three steps, under the lead.
export const HowItWorks: React.FC = () => (
  <section className="how" aria-labelledby="how-title">
    <h2 id="how-title">{copy.how.title}</h2>
    <ol>
      {copy.how.steps.map(step => (
        <li key={step.title}>
          <h3>{step.title}</h3>
          <p>{step.text}</p>
        </li>
      ))}
    </ol>
  </section>
);
