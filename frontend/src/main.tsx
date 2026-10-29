import { createRoot, hydrateRoot, type Root } from 'react-dom/client';
import type { ComponentType } from 'react';
import { App, type AppProps } from './App';
import { loadData } from './data/load';
import { routeFor } from './routes';
import './styles/site.css';

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
async function render(pathname: string) {
  const route = routeFor(pathname);
  let Page: ComponentType | null = null;
  try {
    Page = route ? (await route.load()).default : null;
  } catch {
    state = { error: 'page failed to load' };
  }
  show({ state, route, Page });
  window.scrollTo(0, 0);
}
function navigate(href: string) {
  history.pushState(null, '', href);
  void render(new URL(href, location.href).pathname).then(() => document.getElementById('main')?.focus());
}
window.addEventListener('popstate', () => void render(location.pathname));

void (async () => {
  if (!prerendered) show({ state: 'loading', route: routeFor(location.pathname), Page: null });
  state = await loadData();
  await render(location.pathname);
})();
