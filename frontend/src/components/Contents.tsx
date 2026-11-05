import React, { useEffect, useState } from 'react';
import { copy } from '../content/copy';

// The sticky section bar. On a phone it stays one line and scrolls sideways.
export const Contents: React.FC = () => {
  // underline the section being read; plain links where the observer is missing
  const [active, setActive] = useState('');
  useEffect(() => {
    if (!('IntersectionObserver' in window)) return;
    const observer = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActive(`#${entry.target.id}`);
        }
      },
      { rootMargin: '-40% 0px -55% 0px' },
    );
    for (const item of copy.contents.items) {
      const el = document.getElementById(item.href.slice(1));
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <nav className="toc" aria-label={copy.contents.title}>
      <ol>
        {copy.contents.items.map(item => (
          <li key={item.href}>
            <a href={item.href} aria-current={active === item.href ? 'location' : undefined}>
              {item.label}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
};
