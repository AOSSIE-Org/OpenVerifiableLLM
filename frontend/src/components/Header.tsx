import React, { useEffect, useState } from 'react';
import { copy } from '../content/copy';

type Theme = 'light' | 'dark';

// Light by default, like a reference page; the reader's choice is remembered.
function initialTheme(): Theme {
  try {
    const stored = localStorage.getItem('ovl-theme');
    if (stored === 'light' || stored === 'dark') return stored;
  } catch {
    // localStorage unavailable
  }
  return 'light';
}

export const Header: React.FC = () => {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    try {
      if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      localStorage.setItem('ovl-theme', theme);
    } catch {
      // ignore localStorage errors
    }
  }, [theme]);

  return (
    <header>
      <div className="bar">
        <a className="brand" href="#top">
          <img className="logo-l" src="brand/logo-light.svg" width="30" height="30" alt="" />
          <img className="logo-d" src="brand/logo-dark.svg" width="30" height="30" alt="" />
          <span>
            <span className="brand-name">
              {copy.header.title}
              <em>{copy.header.suffix}</em>
            </span>
            <span className="brand-by">{copy.header.tagline}</span>
          </span>
        </a>

        <nav aria-label="Project links" className="top-links">
          {copy.header.links.map(link => (
            <a key={link.href} href={link.href}>
              {link.label}
            </a>
          ))}
        </nav>

        <button
          className="tbtn"
          type="button"
          onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
          aria-label={copy.header.themeToggleAria}
          aria-pressed={theme === 'dark'}
        >
          {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
      </div>
    </header>
  );
};
