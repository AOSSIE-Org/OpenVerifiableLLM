import { renderToString } from 'react-dom/server';
import { App } from './App';
import { validateStrict } from './data/validate';
import { ROUTES } from './routes';
import Overview from './pages/Overview';
import Check from './pages/Check';
import Verification from './pages/Verification';
import Timeline from './pages/Timeline';
import Exclusion from './pages/Exclusion';
import Downloads from './pages/Downloads';

const PAGES = { overview: Overview, check: Check, verification: Verification, timeline: Timeline, exclusion: Exclusion, downloads: Downloads };

/** Build-time only: render every route to static HTML so core content needs no JavaScript. */
export function renderAll(raw: unknown): { slug: string; key: string; html: string }[] {
  const data = validateStrict(raw);
  const pages = ROUTES.map((r) => ({
    slug: r.slug,
    key: r.key,
    html: renderToString(<App state={{ data, errors: {} }} route={r} Page={PAGES[r.key as keyof typeof PAGES]} />),
  }));
  // GitHub Pages serves dist/404.html for any unknown path.
  return [...pages, { slug: '404.html', key: 'notfound', html: renderToString(<App state={{ data, errors: {} }} route={null} Page={null} />) }];
}
