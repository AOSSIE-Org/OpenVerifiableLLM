import { useEffect, useRef, type ComponentType, type MouseEvent } from 'react';
import { DataContext, type DataState } from './data/context';
import { ROUTES, hrefOf, type RouteDef } from './routes';
import { L, V } from './components/text';
import { ErrorBoundary } from './components/Section';

export interface AppProps {
  state: DataState | { error: string } | 'loading';
  route: RouteDef | null;
  Page: ComponentType | null;
  navigate?: (href: string) => void;
}

function Shell({ route, navigate, children, pinned }: { route: RouteDef | null; navigate?: AppProps['navigate']; children: React.ReactNode; pinned: boolean }) {
  const main = useRef<HTMLElement>(null);
  const onNav = (e: MouseEvent<HTMLAnchorElement>, href: string) => {
    if (!navigate || e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    navigate(href);
  };
  useEffect(() => {
    if (route) document.title = `${document.querySelector(`[data-label="nav.${route.key}"]`)?.textContent ?? ''} · OpenVerifiableLLM`;
  }, [route]);
  return (
    <>
      <a className="skip" href="#main"><L k="skip" /></a>
      <header className="top">
        <a className="wordmark" href={hrefOf(ROUTES[0]!)} onClick={(e) => onNav(e, hrefOf(ROUTES[0]!))}>
          <L k="site.title" />
        </a>
        <nav aria-label="Pages">
          <ul>
            {ROUTES.map((r) => (
              <li key={r.key}>
                <a href={hrefOf(r)} aria-current={route?.key === r.key ? 'page' : undefined} onClick={(e) => onNav(e, hrefOf(r))}>
                  <L k={`nav.${r.key}`} />
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </header>
      <main id="main" ref={main} tabIndex={-1}>
        {children}
      </main>
      <footer className="foot">
        <a href="https://aossie.org"><L k="footer" /></a>
        {pinned ? (
          <span className="muted">
            <L k="footer.data" /> <code><V path="revShort" /></code>
          </span>
        ) : null}
      </footer>
    </>
  );
}

export function App({ state, route, Page, navigate }: AppProps) {
  if (state === 'loading') {
    return (
      <Shell route={route} pinned={false} {...(navigate ? { navigate } : {})}>
        <p role="status"><L k="loading" /></p>
      </Shell>
    );
  }
  if ('error' in state) {
    return (
      <Shell route={route} pinned={false} {...(navigate ? { navigate } : {})}>
        <div className="fatal" role="alert" data-fatal="">
          <h1><L k="error.title" /></h1>
          <p><L k="error.body" /></p>
        </div>
      </Shell>
    );
  }
  return (
    <DataContext.Provider value={state}>
      <Shell route={route} pinned {...(navigate ? { navigate } : {})}>
        <ErrorBoundary>
          {Page ? <Page /> : <h1><L k="notfound" /></h1>}
        </ErrorBoundary>
      </Shell>
    </DataContext.Provider>
  );
}
