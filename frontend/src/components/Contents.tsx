import React from 'react';
import { copy } from '../content/copy';

// The table of contents: a sidebar on wide screens, a collapsible box on
// narrow ones (the same list, shown twice and toggled by CSS).
export const Contents: React.FC = () => {
  const list = (
    <ol>
      {copy.contents.items.map((item, i) => (
        <li key={item.href}>
          <a href={item.href}>
            <span className="toc-n">{i + 1}</span> {item.label}
          </a>
        </li>
      ))}
    </ol>
  );

  return (
    <>
      <nav className="toc" aria-label={copy.contents.title}>
        <h2>{copy.contents.title}</h2>
        {list}
      </nav>
      <details className="toc-mobile">
        <summary>{copy.contents.title}</summary>
        {list}
      </details>
    </>
  );
};
