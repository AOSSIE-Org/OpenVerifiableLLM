import { Link, NavLink, Outlet } from 'react-router-dom';
import { SNAPSHOT_IDS } from '../data/adapters/snapshotAdapter';
import { useSnapSearch } from '../nav';
import { useSnapshotState } from '../snapshot';
import { ClaimStrip } from './ClaimStrip';

const LINKS = [
  { to: '/', label: 'Overview', end: true },
  { to: '/evidence', label: 'Evidence' },
  { to: '/releases', label: 'Release' },
  { to: '/verify', label: 'Verify' },
  { to: '/try', label: 'Inference' },
];

export function AppShell() {
  const { snapshot, error, id } = useSnapshotState();
  const search = useSnapSearch();

  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="mast">
        <div className="mast-inner">
          <Link className="wordmark" to={{ pathname: '/', search }}>
            OpenVerifiable<em>LLM</em>
            <span>AOSSIE · public record</span>
          </Link>
          <nav aria-label="Primary">
            {LINKS.map((link) => (
              <NavLink key={link.to} to={`${link.to}${search}`} end={link.end}>
                {link.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      {snapshot?.mode === 'fixture' ? (
        <p className="tape">Test data — not the real project status</p>
      ) : null}

      {snapshot ? <ClaimStrip snapshot={snapshot} /> : null}

      <main id="main" className="page">
        {error ? (
          <section>
            <p className="kicker">Data error</p>
            <h1>This record cannot be shown as a pass</h1>
            <p className="error">{error.message}</p>
            <p>
              If the saved record is empty or broken, that is an error — not a successful check.
            </p>
          </section>
        ) : (
          <Outlet />
        )}
      </main>

      <footer className="site">
        <div className="foot-inner">
          <p>
            <strong>OpenVerifiableLLM</strong> is an AOSSIE project. This website shows a saved
            public record. It does not train models, sign files, or chat.
          </p>
          <p className="packs">
            Snapshot packs:{' '}
            {SNAPSHOT_IDS.map((item) => (
              <a key={item} href={`#/?snapshot=${item}`}>
                {item}
                {item === id ? ' (current)' : ''}
              </a>
            ))}
          </p>
          <p>
            <a href="https://github.com/AOSSIE-Org/OpenVerifiableLLM">GitHub</a>
            {' · '}
            <a href="https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence">
              Evidence dataset
            </a>
            {' · '}
            <a href="https://aossie.org">aossie.org</a>
            {' · '}
            <a href="https://discord.gg/hjUhu33uAn">Discord</a>
          </p>
        </div>
      </footer>
    </>
  );
}
