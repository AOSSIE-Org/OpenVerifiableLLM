import { useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { PHASE_LABEL, SCOPE_LABEL } from '../content/copy';
import { StatusMark } from '../layout/StatusMark';
import { useSnapSearch } from '../nav';
import { useSnapshot } from '../snapshot';

export function EvidenceDetailPage() {
  const { id } = useParams();
  const snapshot = useSnapshot();
  const search = useSnapSearch();
  const item = snapshot.evidence.find((row) => row.id === id);
  const check = item?.checkId
    ? snapshot.checks.find((row) => row.id === item.checkId)
    : undefined;
  const [copied, setCopied] = useState(false);

  if (!item) {
    return (
      <article>
        <h1>This item is not in the record</h1>
        <p>A missing item is shown as missing — not as a passed check.</p>
        <p>
          <Link to={`/evidence${search}`}>Back to evidence</Link>
        </p>
      </article>
    );
  }

  const digest = item.source.sha256;
  const knownIds = new Set(snapshot.evidence.map((row) => row.id));

  async function copyDigest() {
    if (!digest) return;
    await navigator.clipboard.writeText(digest);
    setCopied(true);
  }

  return (
    <article>
      <p>
        <Link to={`/evidence${search}`}>Evidence</Link>
      </p>
      <h1>{item.title}</h1>
      <p>
        <StatusMark result={item.result} /> · {SCOPE_LABEL[item.scope]} · {item.kind} ·{' '}
        {PHASE_LABEL[item.phase] ?? item.phase}
      </p>
      {item.supersededBy ? (
        <p>
          Replaced by{' '}
          <Link to={`/evidence/${item.supersededBy}${search}`}>{item.supersededBy}</Link>
        </p>
      ) : null}
      {item.supersedes ? (
        <p>
          Replaces <Link to={`/evidence/${item.supersedes}${search}`}>{item.supersedes}</Link>
        </p>
      ) : null}

      <h2>What this sits on</h2>
      <ul className="chain">
        {item.parentIds.length === 0 ? (
          <li>No earlier item linked</li>
        ) : (
          item.parentIds.map((parent) =>
            knownIds.has(parent) ? (
              <li key={parent}>
                Comes from: <Link to={`/evidence/${parent}${search}`}>{parent}</Link>
              </li>
            ) : (
              <li key={parent} className="gap">
                Missing link — {parent} is not in this record
              </li>
            ),
          )
        )}
        <li>
          This item: <strong>{item.id}</strong>
        </li>
        {item.childIds.map((child) => (
          <li key={child}>
            Used by:{' '}
            {knownIds.has(child) ? (
              <Link to={`/evidence/${child}${search}`}>{child}</Link>
            ) : (
              child
            )}
          </li>
        ))}
      </ul>

      <h2>Original location</h2>
      <p>
        <a href={item.source.url} rel="noreferrer">
          {item.source.url}
        </a>
      </p>
      {item.source.path ? <p>Path: {item.source.path}</p> : null}
      {item.source.revision ? (
        <p className="digest">Revision {item.source.revision}</p>
      ) : null}
      {digest ? (
        <p>
          <span className="digest">{digest}</span>{' '}
          <button type="button" onClick={() => void copyDigest()}>
            {copied ? 'Copied digest' : 'Copy digest'}
          </button>
        </p>
      ) : (
        <p>No checksum (SHA-256) is listed in this record.</p>
      )}
      {item.timestamp ? (
        <p className="step-meta">Timestamp {item.timestamp} (UTC as published)</p>
      ) : null}

      {check ? (
        <>
          <h2>Linked check</h2>
          <p>{check.explanation}</p>
          <details>
            <summary>Who ran it (technical)</summary>
            <p className="step-meta">
              Check type: {check.profile}. Run by {check.performedBy ?? 'not listed'}.
              Signed off by {check.attestedBy ?? 'not listed'}. Recomputed in this browser:{' '}
              {check.locallyRecomputed ? 'yes' : 'no'}.
            </p>
          </details>
        </>
      ) : null}

      <details>
        <summary>Raw JSON (display only)</summary>
        <pre className="digest">{JSON.stringify(item, null, 2)}</pre>
      </details>
    </article>
  );
}
