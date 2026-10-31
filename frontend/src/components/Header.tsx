import React from 'react';
import { copy, links } from '../content/copy';

export const Header: React.FC = () => (
  <header className="topbar">
    <div className="topbar-in">
      <a className="site-name" href="#top">
        <img src="brand/favicon.svg" width="24" height="24" alt="" />
        <span>{copy.site.title}</span>
      </a>
      <span className="site-by">{copy.site.by}</span>
      <nav aria-label="External" className="topbar-links">
        <a href={links.github}>GitHub</a>
        <a href={links.evidence}>Evidence</a>
        <a href={links.release}>Release v1.0.1</a>
      </nav>
    </div>
  </header>
);
