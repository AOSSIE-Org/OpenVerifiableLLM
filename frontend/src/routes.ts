import type { ComponentType } from 'react';

export interface RouteDef { slug: string; key: string; load: () => Promise<{ default: ComponentType }> }
export const ROUTES: RouteDef[] = [
  { slug: '', key: 'overview', load: () => import('./pages/Overview') },
  { slug: 'check/', key: 'check', load: () => import('./pages/Check') },
  { slug: 'verification/', key: 'verification', load: () => import('./pages/Verification') },
  { slug: 'timeline/', key: 'timeline', load: () => import('./pages/Timeline') },
  { slug: 'exclusion/', key: 'exclusion', load: () => import('./pages/Exclusion') },
  { slug: 'downloads/', key: 'downloads', load: () => import('./pages/Downloads') },
];
export const BASE = import.meta.env.BASE_URL;
export const hrefOf = (r: RouteDef) => `${BASE}${r.slug}`;
export function routeFor(pathname: string): RouteDef | null {
  let rel = pathname.startsWith(BASE) ? pathname.slice(BASE.length) : pathname.replace(/^\//, '');
  rel = rel.replace(/index\.html$/, '');
  if (rel && !rel.endsWith('/')) rel += '/';
  return ROUTES.find((r) => r.slug === rel) ?? null;
}
