import React from 'react';
import { copy } from '../content/copy';

// The table of contents: a bar under the header on wide screens, a
// collapsible box on narrow ones (the same list, shown twice and toggled by CSS).
export const Contents: React.FC = () => {
  const list = (
    <ol>
      {copy.contents.items.map(item => (
        <li key={item.href}>
          <a href={item.href}>{item.label}</a>
        </li>
      ))}
    </ol>
  );

  return (
    <>
      <nav className="toc" aria-label={copy.contents.title}>
        <h2 className="sr-only">{copy.contents.title}</h2>
        {list}
      </nav>
      <details className="toc-mobile">
        <summary>{copy.contents.title}</summary>
        {list}
      </details>
    </>
  );
};
