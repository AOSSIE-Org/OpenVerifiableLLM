import { Link } from 'react-router-dom';
import { WHY } from '../content/copy';
import { Journey } from './Journey';
import { useSnapSearch } from '../nav';
import { useSnapshot } from '../snapshot';

export function OverviewPage() {
  const snapshot = useSnapshot();
  const search = useSnapSearch();
  const pass = snapshot.checks.filter((c) => c.result === 'PASS').length;
  const fail = snapshot.checks.filter((c) => c.result === 'FAIL').length;
  const pending = snapshot.checks.filter((c) => c.result !== 'PASS' && c.result !== 'FAIL').length;
  const released = snapshot.releases.filter((r) => r.availability === 'available').length;

  return (
    <article>
      <div className="hero">
        <div>
          <p className="kicker">AOSSIE · OpenVerifiableLLM</p>
          <h1>A public record of how a small open model is made.</h1>
          <p className="lede">
            Data, training, checkpoints, and released weights — inspectable, and later
            reproducible. This site shows the record; it does not train or chat.
          </p>
          <p className="actions">
            <Link className="btn btn-gold" to={`/evidence${search}`}>
              Evidence
            </Link>
            <Link className="btn btn-ghost" to={`/verify${search}`}>
              Verification
            </Link>
            <Link className="btn btn-ghost" to={`/try${search}`}>
              Inference
            </Link>
          </p>
        </div>
        <aside className="panel">
          <h2>Current status</h2>
          <p className="hold-line">No model release</p>
          <div className="stats">
            <div className="stat">
              <strong>{pass}</strong>
              <span>passed</span>
            </div>
            <div className="stat">
              <strong>{fail}</strong>
              <span>failed</span>
            </div>
            <div className="stat">
              <strong>{pending}</strong>
              <span>pending</span>
            </div>
            <div className="stat">
              <strong>{released}</strong>
              <span>releases</span>
            </div>
          </div>
        </aside>
      </div>

      <h2>Why</h2>
      <div className="why">
        {WHY.map((item) => (
          <article key={item.title}>
            <h3>{item.title}</h3>
            <p>{item.body}</p>
          </article>
        ))}
      </div>

      <h2>Data → Preparation → Training → Replay → Release → Inference</h2>
      <Journey snapshot={snapshot} />
    </article>
  );
}
