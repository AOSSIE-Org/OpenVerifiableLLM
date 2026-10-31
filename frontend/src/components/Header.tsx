import React, { useEffect, useState } from 'react';
import { copy } from '../content/copy';

export const Header: React.FC = () => {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      const stored = localStorage.getItem('ovl-theme');
      return stored === 'light' ? 'light' : 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    try {
      if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
      } else {
        document.documentElement.removeAttribute('data-theme');
      }
      localStorage.setItem('ovl-theme', theme);
    } catch {
      // ignore localStorage errors
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'light' ? 'dark' : 'light'));
  };

  return (
    <header>
      <div className="wrap bar">
        <a className="brand" href="#top" aria-label="OpenVerifiableLLM by AOSSIE, home">
          <img
            className="logo-d"
            src="brand/logo-dark.svg"
            width="44"
            height="44"
            alt=""
          />
          <img
            className="logo-l"
            src="brand/logo-light.svg"
            width="44"
            height="44"
            alt=""
          />
          <span>
            <span className="brand-name">
              {copy.header.title}
              <em>{copy.header.suffix}</em>
            </span>
            <span className="brand-by">{copy.header.tagline}</span>
          </span>
        </a>

        <nav aria-label="Sections">
          {copy.header.nav.map(item => (
            <a key={item.href} href={item.href}>
              {item.label}
            </a>
          ))}
        </nav>

        <button
          className="tbtn"
          id="theme"
          type="button"
          onClick={toggleTheme}
          aria-label={copy.header.themeToggleAria}
        >
          ◐
        </button>
      </div>
    </header>
  );
};
