/** Light is the default background. Only these two values are ever written to the document or storage. */
export const THEME_KEY = 'ovl-theme';
export type Theme = 'light' | 'dark';
/** Matches --bg in site.css. One colour, not prefers-color-scheme: dark is a toggle, not the OS. */
export const THEME_COLOR: Record<Theme, string> = { light: '#FBFAF7', dark: '#0F1115' };

export function isTheme(v: unknown): v is Theme {
  return v === 'light' || v === 'dark';
}

export function applyTheme(t: Theme): void {
  if (typeof document === 'undefined') return;
  const html = document.documentElement;
  if (!html) return;
  html.setAttribute('data-theme', t);
  html.style.colorScheme = t;
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', THEME_COLOR[t]);
}

export function readTheme(): Theme {
  try {
    const v = localStorage.getItem(THEME_KEY);
    return isTheme(v) ? v : 'light';
  } catch {
    return 'light';
  }
}

export function writeTheme(t: Theme): void {
  applyTheme(t);
  try {
    localStorage.setItem(THEME_KEY, t);
  } catch {
    /* private mode: theme still applies for this page */
  }
}

/** Call from the client entry before React paints. Ignores any value that is not light or dark. */
export function applyStoredTheme(): void {
  applyTheme(readTheme());
}
