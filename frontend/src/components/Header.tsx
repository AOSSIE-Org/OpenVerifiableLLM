import React, { useEffect, useState } from 'react';
import { copy } from '../content/copy';

type Theme = 'light' | 'dark';

// Small line icons for the header buttons, drawn with the text colour.
const icons: Record<string, React.ReactNode> = {
  Source: (
    <>
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </>
  ),
  Evidence: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v14c0 1.66 3.58 3 8 3s8-1.34 8-3V5" />
      <path d="M4 12c0 1.66 3.58 3 8 3s8-1.34 8-3" />
    </>
  ),
  dark: <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
  light: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
};

const Icon: React.FC<{ name: string }> = ({ name }) => (
  <svg className="hicon" viewBox="0 0 24 24" aria-hidden="true">
    {icons[name]}
  </svg>
);

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
            <a key={link.href} href={link.href} className="hbtn">
              <Icon name={link.label} />
              {link.label}
            </a>
          ))}
        </nav>

        <button
          className="tbtn hbtn"
          type="button"
          onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
          aria-label={copy.header.themeToggleAria}
          aria-pressed={theme === 'dark'}
        >
          <Icon name={theme === 'dark' ? 'light' : 'dark'} />
          {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
      </div>
    </header>
  );
};
