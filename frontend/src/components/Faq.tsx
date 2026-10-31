import React from 'react';
import { copy } from '../content/copy';

export const Faq: React.FC = () => (
  <section id="faq">
    <h2>{copy.faq.h2}</h2>
    <dl className="qa">
      {copy.faq.items.map(item => (
        <React.Fragment key={item.q}>
          <dt>{item.q}</dt>
          <dd>{item.a}</dd>
        </React.Fragment>
      ))}
    </dl>
    <p>
      {copy.faq.stillTitle} <a href="https://discord.gg/hjUhu33uAn">{copy.faq.discord}</a> or{' '}
      <a href={copy.faq.issueHref}>{copy.faq.issue.toLowerCase()}</a>.
    </p>
  </section>
);

export const ExternalLinks: React.FC = () => (
  <section id="links">
    <h2>External links</h2>
    <ul>
      {[...copy.footer.projectLinks, ...copy.footer.communityLinks].map(link => (
        <li key={link.href}>
          <a href={link.href}>{link.label}</a>
        </li>
      ))}
    </ul>
  </section>
);
