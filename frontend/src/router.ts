import { useEffect, useState } from 'react';
export function routeFromHash(hash: string) {
  const raw = hash.replace(/^#/, '') || '/';
  const [path, query = ''] = raw.split('?');
  return { path, params: new URLSearchParams(query) };
}
export function useRoute() {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => { const listener = () => setHash(window.location.hash); window.addEventListener('hashchange', listener); return () => window.removeEventListener('hashchange', listener); }, []);
  return routeFromHash(hash);
}
export function detailHref(id: string) { return '#/evidence/' + encodeURIComponent(id); }
