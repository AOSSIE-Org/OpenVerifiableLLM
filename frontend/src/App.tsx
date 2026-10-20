import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { validateSnapshot, type Snapshot } from './data/contracts';
import { isStale, loadSnapshot, snapshotUrl } from './data/load';
import { useRoute } from './router';
import { EvidenceDetail, EvidenceExplorer, InferenceUnavailable, Overview, Releases, Verification } from './pages';
import { Button, Label, State, formatDate } from './components';
import { ScrollLines, SiteFooter, useScrollReveal } from './chrome';

const DevInference = import.meta.env.DEV && (import.meta.env.VITE_MOCK_INFERENCE === 'true' || window.location.port === '5173')
  ? lazy(() => import('./inference/DevInference')) : null;
type DataState = { status: 'loading' } | { status: 'ready' | 'stale'; snapshot: Snapshot } | { status: 'error'; message: string };
const nav = [['/', 'Overview'], ['/evidence', 'Evidence'], ['/releases', 'Releases'], ['/verification', 'Verification'], ['/inference', 'Inference']];
export function App() {
  const { path, params } = useRoute();
  const [data, setData] = useState<DataState>({ status: 'loading' });
  const [attempt, setAttempt] = useState(0); const [menuOpen, setMenuOpen] = useState(false);
  const main = useRef<HTMLElement>(null);
  const previousPath = useRef(path);
  const configuredMode = import.meta.env.VITE_DATA_MODE || 'fixture';
  useEffect(() => {
    const controller = new AbortController();
    setData({ status: 'loading' });
    async function read() {
      if (configuredMode !== 'fixture' && configuredMode !== 'public-snapshot') throw new Error('Unknown configured data mode.');
      if (import.meta.env.DEV && import.meta.env.VITE_SCENARIO) {
        const { scenario } = await import('./data/scenarios');
        return validateSnapshot(scenario(import.meta.env.VITE_SCENARIO), configuredMode);
      }
      const url = snapshotUrl(import.meta.env.BASE_URL, import.meta.env.VITE_SNAPSHOT_PATH || 'data/snapshot.json', window.location.href);
      return loadSnapshot(url, configuredMode, controller.signal);
    }
    void read().then(snapshot => { if (!controller.signal.aborted) setData({ status: isStale(snapshot) ? 'stale' : 'ready', snapshot }); })
      .catch((e: unknown) => { if (!controller.signal.aborted) setData({ status: 'error', message: e instanceof Error ? e.message.slice(0, 1200) : 'Unknown snapshot error.' }); });
    return () => controller.abort();
  }, [attempt, configuredMode]);
  useEffect(() => {
    setMenuOpen(false);
    if (previousPath.current !== path) { main.current?.focus(); window.scrollTo(0, 0); }
    previousPath.current = path;
    document.title = (path.startsWith('/evidence/') ? 'Evidence detail' : nav.find(([p]) => p === path)?.[1] || 'Page unavailable') + ' / OpenVerifiableLLM';
  }, [path]);
  useScrollReveal([path, data.status]);
  let content;
  if (data.status === 'loading') content = <State title="Loading display snapshot…">Fetching small presentation metadata. No verification is being performed.</State>;
  else if (data.status === 'error') content = <><Label>Data unavailable / no verification conclusion</Label><State title="Snapshot unavailable or invalid" retry={() => setAttempt(n => n + 1)}>{data.message}</State><p>Invalid data is not replaced with synthetic success. Review the configured snapshot and its schema.</p></>;
  else {
    const snapshot = data.snapshot;
    if (path === '/') content = <Overview snapshot={snapshot} />;
    else if (path === '/evidence') content = <EvidenceExplorer snapshot={snapshot} params={params} />;
    else if (path.startsWith('/evidence/')) {
      let id = ''; try { id = decodeURIComponent(path.slice('/evidence/'.length)); } catch { id = 'invalid-route-encoding'; }
      content = <EvidenceDetail snapshot={snapshot} id={id} />;
    } else if (path === '/releases') content = <Releases snapshot={snapshot} />;
    else if (path === '/verification') content = <Verification snapshot={snapshot} />;
    else if (path === '/inference') content = <><InferenceUnavailable />{DevInference && <Suspense fallback={<State title="Loading development UI">No model is being loaded.</State>}><DevInference /></Suspense>}</>;
    else content = <><State title="Page unavailable">This route does not exist.</State><Button href="#/">Return to overview</Button></>;
  }
  return <>
    <ScrollLines />
    <a className="skip-link" href="#main" onClick={e => { e.preventDefault(); main.current?.focus(); }}>Skip to content</a>
    <header className="site-header"><div className="container masthead"><a href="#/" className="brand" aria-label="OpenVerifiableLLM overview"><span className="brand-mark" aria-hidden="true" /><span>OpenVerifiableLLM</span></a>
      <button className="menu-button" aria-expanded={menuOpen} aria-controls="primary-nav" onClick={() => setMenuOpen(v => !v)}>Menu</button>
      <nav id="primary-nav" aria-label="Primary" className={menuOpen ? 'open' : ''}>{nav.map(([p, label]) => <a key={p} href={'#' + p} aria-current={(p === '/' ? path === '/' : path.startsWith(p)) ? 'page' : undefined}>{label}</a>)}</nav>
    </div></header>
    {(data.status === 'stale') && <div className="container stale" role="status">Stale snapshot: generated {formatDate(data.snapshot.generatedAt)}. This display may not reflect current project status. <button onClick={() => setAttempt(n => n + 1)}>Reload metadata</button></div>}
    <main id="main" ref={main} tabIndex={-1} className="container main stack">{content}</main>
    <SiteFooter nav={nav} sample={(data.status === 'ready' || data.status === 'stale') && data.snapshot.mode === 'fixture'} />
  </>;
}
