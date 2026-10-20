import { useState, type ReactNode } from 'react';
import { detailHref } from './router';
import { safeUrl, type Source, type Snapshot, type Check, type Result, missingCheckReferences } from './data/contracts';

export const profileNames = {
  'artifact-identity': 'Artifact integrity & publisher identity',
  'data-reconstruction': 'Data reconstruction',
  'sampled-replay': 'Sampled replay',
  'complete-replay': 'Complete end-to-end replay',
  'inference-reproduction': 'Inference reproduction',
};
export const attributionNames = {
  'publisher-report': 'Publisher report',
  'project-operated-replay': 'Project-operated replay',
  'consumer-local-recomputation': 'Reported consumer-local recomputation',
  'independent-third-party': 'Reported independent third-party verification',
  unknown: 'Attribution not supplied',
};
export function Button({ children, href, onClick, disabled = false }: { children: ReactNode; href?: string; onClick?: () => void; disabled?: boolean }) {
  // The connected Figma SDS Button has no code implementation in the supplied repo.
  // Local semantic equivalent preserves its 2 px radius, medium size and colors.
  return href ? <a className="button" href={href}>{children}</a> : <button type="button" className="button" disabled={disabled} onClick={onClick}>{children}</button>;
}
export function Label({ children }: { children: ReactNode }) { return <p className="eyebrow">{children}</p>; }
export function Title({ label, children, intro }: { label: string; children: ReactNode; intro?: string }) {
  return <div className="page-title"><Label>{label}</Label><h1>{children}</h1>{intro && <p className="intro">{intro}</p>}</div>;
}
export function Note({ title, children, danger = false }: { title: string; children: ReactNode; danger?: boolean }) {
  return <aside className={'note' + (danger ? ' danger' : '')}><h2>{title}</h2><div>{children}</div></aside>;
}
export function State({ title, children, retry }: { title: string; children: ReactNode; retry?: () => void }) {
  return <section className="state" role="status"><h2>{title}</h2><p>{children}</p>{retry && <Button onClick={retry}>Retry loading snapshot</Button>}</section>;
}
const resultDescriptions: Record<Result, string> = {
  PASS: 'This specific reported check passed within its scope.',
  FAIL: 'This check ran and did not meet its requirements.',
  NOT_RUN: 'This check has not been performed.',
  UNAVAILABLE: 'Required evidence or resources could not be obtained.',
  UNSUPPORTED: 'This environment or format is not supported by the check.',
};
export function ResultText({ result }: { result: Result }) { return <span className={'result result-' + result} title={resultDescriptions[result]}>{result}</span>; }
export function Copy({ value, label = 'Copy full digest' }: { value: string; label?: string }) {
  const [feedback, setFeedback] = useState('');
  async function copy() {
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(value);
      setFeedback('Copied to clipboard.');
    } catch { setFeedback('Copy unavailable. Select and copy the full text above.'); }
  }
  return <div className="copy-row"><Button onClick={() => void copy()}>{label}</Button><span role="status" aria-live="polite">{feedback}</span></div>;
}
export function External({ url, children }: { url: string | null; children: ReactNode }) {
  if (!url || !safeUrl.safeParse(url).success) return <span className="muted">Source URL not supplied.</span>;
  return <a href={url} target="_blank" rel="noopener noreferrer">{children} <span aria-hidden="true">↗</span><span className="sr-only"> (external, opens in a new tab)</span></a>;
}
export function SourceInfo({ source }: { source: Source }) {
  return <div className="source-info"><External url={source.url}>Original evidence</External><dl>
    <div><dt>Immutable revision</dt><dd className="mono">{source.revision ?? 'Not supplied'}</dd></div>
    <div><dt>Artifact path</dt><dd className="mono">{source.path ?? 'Not supplied'}</dd></div>
  </dl></div>;
}
export function EvidenceLink({ id, snapshot }: { id: string; snapshot: Snapshot }) {
  const e = snapshot.evidence.find(e => e.id === id);
  return <a href={detailHref(id)} className={e ? '' : 'missing'}>{e ? e.title : id + ' — reference missing'} <span aria-hidden="true">↗</span></a>;
}
export function CheckFacts({ check, snapshot }: { check: Check; snapshot: Snapshot }) {
  const missing = missingCheckReferences(snapshot, check);
  return <section className="check-facts">
    <h3>{check.title}</h3><p><ResultText result={check.result} /> <span className="mono"> · {check.scope.toUpperCase()} SCOPE</span></p>
    <p>{check.explanation}</p><dl>
      <div><dt>Profile</dt><dd>{profileNames[check.profile]}</dd></div>
      <div><dt>Attribution</dt><dd>{attributionNames[check.attribution]}</dd></div>
      <div><dt>Performed by</dt><dd>{check.performedBy ?? 'Not supplied'}</dd></div>
      <div><dt>Attested by</dt><dd>{check.attestedBy ?? 'Not supplied'}</dd></div>
      <div><dt>Reported locally recomputed</dt><dd>{check.locallyRecomputed ? 'Yes — in the supplied report' : 'No'}</dd></div>
      <div><dt>Recomputed by this visitor</dt><dd>No — this frontend displays reports.</dd></div>
      <div><dt>Timestamp</dt><dd>{check.timestamp ? formatDate(check.timestamp) : 'Not supplied'}</dd></div>
    </dl>
    <p className="small">Supporting records:</p><ul className="relation-list">{check.evidenceIds.map(id => <li key={id}><EvidenceLink id={id} snapshot={snapshot} /></li>)}</ul>
    {missing.length > 0 && <p className="missing" role="status">Supporting evidence unavailable: {missing.join(', ')}. This reported result cannot be assessed here.</p>}
    {check.independenceEvidence && <SourceInfo source={check.independenceEvidence} />}
  </section>;
}
export function formatDate(date: string) { return new Intl.DateTimeFormat('en-GB', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'UTC' }).format(new Date(date)) + ' UTC'; }
