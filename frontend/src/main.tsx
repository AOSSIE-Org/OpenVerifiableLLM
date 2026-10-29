import { createRoot, hydrateRoot, type Root } from 'react-dom/client';
import type { ComponentType } from 'react';
import { App, type AppProps } from './App';
import { loadData } from './data/load';
import { ROUTES, routeFor } from './routes';
import { label } from './content/ui';
import './styles/site.css';

/** The data file failed (missing, malformed, wrong shape, aborted). Keep the prerendered page and replace
 *  only the evidence that depends on that data, so nothing is shown as checked. */
function scopeFailure() {
  for (const s of el.querySelectorAll('[data-section]')) {
    const box = document.createElement('div');
    box.className = 'fallback';
    box.setAttribute('role', 'alert');
    box.setAttribute('data-fallback', '');
    box.setAttribute('data-fatal', '');
    box.textContent = label('error.section');
    s.replaceWith(box);
  }
  // Anything left that still shows a result (outside a data section) is removed too.
  for (const r of el.querySelectorAll('.tile, [data-result]')) r.remove();
  el.setAttribute('data-failed', '');
}

const el = document.getElementById('root')!;
const prerendered = el.firstElementChild !== null;
let root: Root | null = null;

function show(props: AppProps) {
  const node = <App {...props} navigate={navigate} />;
  if (root) root.render(node);
  // Hydrate only when the client sees exactly what the build rendered (valid data, no broken section).
  else if (prerendered && props.state !== 'loading' && !('error' in props.state) && Object.keys(props.state.errors).length === 0) root = hydrateRoot(el, node);
  else {
    el.textContent = '';
    root = createRoot(el);
    root.render(node);
  }
}

let state: AppProps['state'] = 'loading';
async function render(pathname: string, initial = false) {
  const route = routeFor(pathname);
  let Page: ComponentType | null = null;
  try {
    Page = route ? (await route.load()).default : null;
  } catch {
    state = { error: 'page failed to load' };
  }
  show({ state, route, Page });
  if (!initial) window.scrollTo(0, 0);
  else if (location.hash.length > 1) {
    // Keep claim deep links (#ve-04) working after the client render; an unknown or odd hash is ignored.
    let id = '';
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { /* malformed escape */ }
    if (id) document.getElementById(id)?.scrollIntoView();
  }
}
function navigate(href: string) {
  history.pushState(null, '', href);
  void render(new URL(href, location.href).pathname).then(() => document.getElementById('main')?.focus());
}
window.addEventListener('popstate', () => void render(location.pathname));

void (async () => {
  if (!prerendered) show({ state: 'loading', route: routeFor(location.pathname), Page: null });
  state = await loadData();
  if (prerendered && 'error' in state) return scopeFailure();
  await render(location.pathname, true);
  // Warm the other page chunks when idle, so a later click still works on a flaky or lost connection.
  const idle = window.requestIdleCallback ?? ((f: () => void) => setTimeout(f, 1500));
  idle(() => ROUTES.forEach((r) => void r.load().catch(() => undefined)));
})();
