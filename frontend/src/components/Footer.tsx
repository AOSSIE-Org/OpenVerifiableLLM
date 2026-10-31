import React from 'react';
import { copy, links } from '../content/copy';

export const Footer: React.FC<{ revision: string; fetchedAt: string }> = ({ revision, fetchedAt }) => (
  <footer className="site-footer">
    <ul className="footer-links">
      {copy.footer.links.map(l => (
        <li key={l.href}>
          <a href={l.href}>{l.label}</a>
        </li>
      ))}
    </ul>
    <p>
      {copy.footer.snapshot} {fetchedAt.slice(0, 10)} from Hugging Face revision <code>{revision}</code>.
    </p>
    <p>
      {copy.footer.license} <a href={links.aossie}>{copy.site.by}</a>.
    </p>
  </footer>
);
