import { useState } from 'react';
import { L } from './text';
import { useHydrated } from './hydrated';
import { label } from '../content/ui';
import { readTheme, writeTheme, type Theme } from '../theme';

/** Shown only after JavaScript runs, so a no-JS page stays light with no dead control. */
export function ThemeToggle() {
  const hydrated = useHydrated();
  const [theme, setTheme] = useState<Theme>(() => (typeof document === 'undefined' ? 'light' : readTheme()));
  if (!hydrated) return null;
  const next: Theme = theme === 'dark' ? 'light' : 'dark';
  const caption = next === 'dark' ? 'theme.toDark' : 'theme.toLight';
  return (
    <button
      type="button"
      className="theme-btn"
      aria-pressed={theme === 'dark'}
      aria-label={`${label(caption)}. ${label('theme.group')}`}
      data-theme-toggle=""
      onClick={() => {
        writeTheme(next);
        setTheme(next);
      }}
    >
      <L k={caption} />
    </button>
  );
}
