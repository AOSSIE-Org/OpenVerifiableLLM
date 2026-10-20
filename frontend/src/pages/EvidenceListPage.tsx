import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  CHECK_RESULTS,
  SCOPES,
  type CheckResult,
  type EvidenceItem,
} from '../data/contracts';
import { PHASE_LABEL, SCOPE_LABEL } from '../content/copy';
import { StatusMark } from '../layout/StatusMark';
import { useSnapSearch } from '../nav';
import { useSnapshot } from '../snapshot';

const PHASE_ORDER = [
  'source',
  'prepare',
  'train',
  'replay',
  'release',
  'other',
] as const;

const PHASE_TITLE: Record<string, string> = {
  source: PHASE_LABEL.source,
  prepare: PHASE_LABEL.prepare,
  train: PHASE_LABEL.train,
  replay: PHASE_LABEL.replay,
  release: PHASE_LABEL.release,
  other: PHASE_LABEL.other,
};

function formatSize(bytes: number | null): string {
  if (bytes === null) return '—';
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function phaseKey(phase: string): string {
  return (PHASE_ORDER as readonly string[]).includes(phase) ? phase : 'other';
}

function Table({ rows, search }: { rows: EvidenceItem[]; search: string }) {
  if (rows.length === 0) return <p>No rows in this group.</p>;
  return (
    <table className="docket">
      <thead>
        <tr>
          <th>Title</th>
          <th>Kind</th>
          <th>Scope</th>
          <th>Result</th>
          <th>When</th>
          <th>Size</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((item) => (
          <tr key={item.id} className={item.supersededBy ? 'superseded' : undefined}>
            <td>
              <Link className="title" to={`/evidence/${item.id}${search}`}>
                {item.title}
              </Link>
            </td>
            <td>{item.kind}</td>
            <td>{SCOPE_LABEL[item.scope]}</td>
            <td>
              <StatusMark result={item.result as CheckResult} />
            </td>
            <td>{item.timestamp ?? '—'}</td>
            <td>{formatSize(item.sizeBytes)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function EvidenceListPage() {
  const snapshot = useSnapshot();
  const search = useSnapSearch();
  const [result, setResult] = useState('all');
  const [scope, setScope] = useState('all');
  const [kind, setKind] = useState('all');
  const [phase, setPhase] = useState('all');

  const kinds = useMemo(
    () => [...new Set(snapshot.evidence.map((item) => item.kind))].sort(),
    [snapshot.evidence],
  );
  const phases = useMemo(
    () => [...new Set(snapshot.evidence.map((item) => item.phase))].sort(),
    [snapshot.evidence],
  );

  const filtered = useMemo(() => {
    return snapshot.evidence.filter((item) => {
      if (result !== 'all' && item.result !== result) return false;
      if (scope !== 'all' && item.scope !== scope) return false;
      if (kind !== 'all' && item.kind !== kind) return false;
      if (phase !== 'all' && item.phase !== phase) return false;
      return true;
    });
  }, [snapshot.evidence, result, scope, kind, phase]);

  const grouped = PHASE_ORDER.map((key) => ({
    key,
    title: PHASE_TITLE[key],
    rows: filtered.filter((item) => phaseKey(item.phase) === key),
  })).filter((g) => g.rows.length > 0);

  const pass = filtered.filter((i) => i.result === 'PASS').length;

  return (
    <article>
      <p className="kicker">Public record</p>
      <h1>Evidence</h1>
      <p>
        <strong>What.</strong> The public record: files and reports the project published.
      </p>
      <p>
        <strong>Why.</strong> Evidence is what verification later checks. This page only shows
        links; it does not re-run training or generate text.
      </p>
      <p>
        <strong>Current status.</strong> Rows below are this saved record. Open one for who ran the
        check and the technical fields.
      </p>
      <div className="stats" style={{ margin: '1rem 0 1.2rem', maxWidth: '36rem' }}>
        <div className="stat">
          <strong>{filtered.length}</strong>
          <span>items shown</span>
        </div>
        <div className="stat">
          <strong>{pass}</strong>
          <span>PASS in view</span>
        </div>
      </div>
      <div className="filters">
        <label>
          Result
          <select value={result} onChange={(e) => setResult(e.target.value)}>
            <option value="all">All</option>
            {CHECK_RESULTS.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label>
          Scope
          <select value={scope} onChange={(e) => setScope(e.target.value)}>
            <option value="all">All</option>
            {SCOPES.map((item) => (
              <option key={item} value={item}>
                {SCOPE_LABEL[item]}
              </option>
            ))}
          </select>
        </label>
        <label>
          Kind
          <select value={kind} onChange={(e) => setKind(e.target.value)}>
            <option value="all">All</option>
            {kinds.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
        <label>
          Phase
          <select value={phase} onChange={(e) => setPhase(e.target.value)}>
            <option value="all">All</option>
            {phases.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </label>
      </div>
      {filtered.length === 0 ? (
        <p>No rows match these filters. That is not a passed check.</p>
      ) : (
        grouped.map((group) => (
          <section key={group.key}>
            <h2 className="group-h">{group.title}</h2>
            <Table rows={group.rows} search={search} />
          </section>
        ))
      )}
    </article>
  );
}
