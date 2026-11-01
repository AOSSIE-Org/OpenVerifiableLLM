import React from 'react';
import { copy } from '../content/copy';

export const Faq: React.FC = () => (
  <section id="faq">
    <h2>{copy.faq.h2}</h2>
    <dl className="qa">
      {copy.faq.items.map(item => (
        <div key={item.q}>
          <dt>{item.q}</dt>
          <dd>{item.a}</dd>
        </div>
      ))}
    </dl>
    <p>
      {copy.faq.stillTitle} <a href="https://discord.gg/hjUhu33uAn">{copy.faq.discord}</a> or{' '}
      <a href={copy.faq.issueHref}>{copy.faq.issue.toLowerCase()}</a>.
    </p>
  </section>
);
