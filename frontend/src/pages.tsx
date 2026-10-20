import { useMemo } from 'react';
import { type Snapshot, type Evidence, results, scopes, evidenceChecks, immutableFileUrl } from './data/contracts';
import { Button, CheckFacts, Copy, EvidenceLink, External, Label, Note, ResultText, SourceInfo, State, Title, attributionNames, formatDate } from './components';
import { detailHref } from './router';
import { ContainerScroll } from '@/components/ui/container-scroll-animation';
import { GlowCard } from '@/components/ui/spotlight-card';

const stages = ['Public sources', 'Prepared data / tokenizer', 'Initialization', 'Base training', 'Conversational training', 'Complete replay', 'Publication'];
export function EvidenceTable({ snapshot, items }: { snapshot: Snapshot; items: Evidence[] }) {
  if (!items.length) return <State title="No matching evidence">Try another search or reset the filters. No verification result is inferred.</State>;
  return <div className="table-scroll"><table className="evidence-table">
    <caption className="sr-only">Evidence records and their scoped reported check results</caption>
    <thead><tr><th scope="col">Evidence</th><th scope="col">Scope</th><th scope="col">Reported result</th><th scope="col">Attribution</th></tr></thead>
    <tbody>{items.map(e => {
      const checks = evidenceChecks(snapshot, e.id);
      return <tr key={e.id}><td><a className="record-title" href={detailHref(e.id)}>{e.title} <span aria-hidden="true">↗</span></a><span className="record-meta">{e.phase} · {e.kind}</span>{e.supersededBy && <span className="record-meta">Superseded — original preserved</span>}</td>
        <td className="scope-cell">{e.scope}</td>
        <td>{checks.length ? checks.map(c => <div className="table-check" key={c.id}><ResultText result={c.result} /><span className="record-meta">{c.profile}</span></div>) : <span className="muted">No check reported</span>}</td>
        <td>{checks.map(c => <div key={c.id}>{attributionNames[c.attribution]}<span className="record-meta">{c.performedBy ?? 'Operator not supplied'}{c.timestamp && ' · ' + formatDate(c.timestamp)}</span></div>)}{!checks.length && 'Not supplied'}</td>
      </tr>;
    })}</tbody>
  </table></div>;
}
export function Overview({ snapshot }: { snapshot: Snapshot }) {
  const hasReleased = snapshot.releases.some(r => r.availability === 'available');
  return <>
    <Label>Open research / model provenance</Label>
    <section className="hero">
      <div className="hero-copy"><h1>A model with a<br className="desktop-break" /> traceable beginning.</h1>
        <p className="intro">OpenVerifiableLLM explores whether a small language model’s declared inputs, data preparation and training computation can be inspected and reproduced.</p>
        <div className="actions"><Button href="#/evidence">Explore evidence</Button><a href="#/verification">How verification works <span aria-hidden="true">↗</span></a></div>
      </div>
      <GlowCard as="aside" customSize glowColor="blue" className="research-note"><Label>Release availability</Label><h2>{hasReleased ? 'Release records supplied.' : 'Not released yet.'}</h2>
        <p>{hasReleased ? 'Inspect exact identities and scoped reports. Synthetic release examples are not downloadable project models.' : 'Final base and conversational models are not available in this snapshot.'}</p>
        <hr /><div className="mono small"><p>Snapshot: {formatDate(snapshot.generatedAt)}</p><p>{snapshot.mode === 'fixture' ? 'Actual project status: approval pending' : 'Approved public display snapshot'}</p></div>
        <a href="#/releases">View releases <span aria-hidden="true">↗</span></a>
      </GlowCard>
    </section>
    <hr />
    <section className="section stack"><Label>01 / The process</Label><h2>From public inputs to a reproducible release.</h2>
      <ol className="process">{stages.map((s, i) => <li key={s}><Label>{String(i + 1).padStart(2, '0')}</Label><p>{s}</p><span className="mono small muted">Declared stage</span></li>)}</ol>
      <p className="small muted">Process order describes the intended workflow. It does not indicate that every stage is complete.</p>
      <details><summary>Workflow summary (separate from verification)</summary><p className="mono small">Workflow: {snapshot.workflow.state}</p><p>{snapshot.workflow.summary}</p></details>
    </section>
    <hr />
    <section className="section stack"><Label>02 / The evidence</Label><h2>Read the record, not just the verdict.</h2>
      <p className="muted">Every check has a scope. A pilot pass does not establish a production model, and a published report is not a local recomputation.</p>
      {snapshot.checks.length === 0 ? <State title="No checks reported">This snapshot contains no checks. No successful verification is inferred.</State> : <EvidenceTable snapshot={snapshot} items={snapshot.evidence.slice(0, 2)} />}
      <a href="#/evidence">Browse all evidence <span aria-hidden="true">↗</span></a>
      <Note title="What verification establishes"><p>Provenance and replay checks concern declared artifacts and computation. They do not establish answer accuracy or overall model safety.</p></Note>
    </section>
    <details><summary>Snapshot provenance and source references</summary><p className="small">Generated {formatDate(snapshot.generatedAt)}. {snapshot.mode === 'fixture' ? 'These project links supply context; they are not sources for the synthetic check results.' : 'Approved display snapshot; fetching this data is not verification.'}</p>{snapshot.sources.map((s, i) => <SourceInfo source={s} key={i} />)}</details>
  </>;
}
export function EvidenceExplorer({ snapshot, params }: { snapshot: Snapshot; params: URLSearchParams }) {
  const query = params.get('q') ?? '';
  const phase = params.get('phase') ?? ''; const kind = params.get('kind') ?? '';
  const result = params.get('result') ?? ''; const scope = params.get('scope') ?? '';
  const items = useMemo(() => snapshot.evidence.filter(e => {
    const checks = evidenceChecks(snapshot, e.id);
    const search = [e.id, e.title, e.source.sha256 ?? '', e.phase, e.kind].join(' ').toLowerCase();
    return search.includes(query.toLowerCase()) && (!phase || phase === e.phase) && (!kind || kind === e.kind)
      && (!scope || scope === e.scope) && (!result || checks.some(c => c.result === result));
  }), [snapshot, query, phase, kind, result, scope]);
  function update(key: string, value: string) {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value); else next.delete(key);
    const q = next.toString(); window.location.hash = '/evidence' + (q ? '?' + q : '');
  }
  const phases = [...new Set(snapshot.evidence.map(e => e.phase))].sort();
  const kinds = [...new Set(snapshot.evidence.map(e => e.kind))].sort();
  return <>
    <ContainerScroll titleComponent={<Title label={'Evidence index / ' + (snapshot.mode === 'fixture' ? 'synthetic fixture' : 'public snapshot')} intro="Search reports and artifacts by phase, scope and result. Open a record to inspect its sources and limits.">Inspect the evidence.</Title>}>
      <div className="scroll-card-body stack">

    <div className="search"><label className="sr-only" htmlFor="evidence-search">Search evidence</label><input id="evidence-search" type="search" value={query} onChange={e => update('q', e.target.value)} placeholder="Search by title, ID or digest…" /></div>
    <div className="filters">{[
      ['phase', 'Phase', phase, phases], ['kind', 'Evidence kind', kind, kinds],
      ['result', 'Result', result, [...results]], ['scope', 'Scope', scope, [...scopes]],
    ].map(([key, title, value, options]) => <div className="filter" key={String(key)}><label className="eyebrow" htmlFor={'filter-' + key}>{title}</label><select id={'filter-' + key} value={value as string} onChange={e => update(key as string, e.target.value)}><option value="">All {key === 'kind' ? 'kinds' : key === 'phase' ? 'phases' : key === 'scope' ? 'scopes' : 'results'}</option>{(options as string[]).map(v => <option value={v} key={v}>{v}</option>)}</select></div>)}</div>
    <div className="list-summary"><p className="mono small muted" aria-live="polite">{items.length} {snapshot.mode === 'fixture' ? 'example ' : ''}records · URL-preserved filters</p><a href="#/evidence">Reset filters</a></div>
          {snapshot.checks.length === 0 && <State title="No checks reported">The snapshot contains no verification checks. Evidence loading does not establish success.</State>}
    <EvidenceTable snapshot={snapshot} items={items} />
</div>
    </ContainerScroll>
    <Note title="A result applies only to its stated scope."><p>Fixture: synthetic development data. Pilot: a limited experiment. Production: declared production artifacts and computation.</p></Note>
    <details><summary>How absent records are handled</summary><div className="three-columns">
      <div><h3>No checks reported</h3><p>A valid snapshot may contain zero checks. No success is inferred.</p></div>
      <div><h3>Report unavailable</h3><p>Unavailable evidence is shown explicitly. A stored publisher result is not recomputed.</p></div>
      <div><h3>Invalid snapshot</h3><p>Malformed required metadata stops data rendering. Sample success is never a fallback.</p></div>
    </div></details>
  </>;
}
export function EvidenceDetail({ snapshot, id }: { snapshot: Snapshot; id: string }) {
  const e = snapshot.evidence.find(e => e.id === id);
  if (!e) return <><Title label="Evidence / missing reference">Evidence unavailable.</Title><State title="Missing evidence reference">The record “{id}” is not present in this snapshot. No verification conclusion can be drawn from its absence.</State><a href="#/evidence">Return to evidence explorer</a></>;
  const checks = evidenceChecks(snapshot, id);
  const children = snapshot.evidence.filter(child => child.parents.includes(id));
  return <>
    <a href="#/evidence" className="small">Evidence / {e.title}</a>
    <Title label={'Record / ' + e.id}>{e.title}</Title>
    {e.synthetic && <p className="eyebrow">Synthetic display example · {e.scope} scope</p>}
    {e.supersededBy && <Note title="Superseded report"><p>This original record is preserved. Its result has not been overwritten. Replacement: <EvidenceLink id={e.supersededBy} snapshot={snapshot} /></p></Note>}
    <div className="detail-layout"><div className="stack">
      {checks.length ? checks.map(c => <CheckFacts key={c.id} check={c} snapshot={snapshot} />) : <State title="No check reported">This is an evidence artifact with no supplied verification check.</State>}
    </div><Note title="Read within scope."><p>A reported pass is not proof that this browser independently verified the computation.</p><p className="small">Reported relationships and publisher attestations are display metadata.</p></Note></div>
    <section className="stack"><Label>Artifact identity</Label><h2>Digest and source</h2><p className="small muted">SHA-256 / {e.synthetic ? 'synthetic example — not a project artifact' : 'reported artifact digest'}</p>
      {e.source.sha256 ? <><code className="digest" tabIndex={0}>{e.source.sha256}</code><Copy value={e.source.sha256} /></> : <p>Digest not supplied.</p>}
      <SourceInfo source={e.source} />
      <dl><div><dt>Size</dt><dd>{e.sizeBytes === null ? 'Not supplied' : e.sizeBytes.toLocaleString() + ' bytes'}</dd></div><div><dt>Timestamp</dt><dd>{e.timestamp ? formatDate(e.timestamp) : 'Not supplied'}</dd></div></dl>
    </section>
    <hr />
    <section className="stack"><Label>Reported relationships</Label><h2>Keep the chain inspectable.</h2>
      <div><h3>Parents</h3>{e.parents.length ? <ul className="relation-list">{e.parents.map(p => <li key={p}><EvidenceLink id={p} snapshot={snapshot} /></li>)}</ul> : <p className="muted">No parents reported.</p>}</div>
      <div><h3>Children</h3>{children.length ? <ul className="relation-list">{children.map(c => <li key={c.id}><EvidenceLink id={c.id} snapshot={snapshot} /></li>)}</ul> : <p className="muted">No children reported.</p>}</div>
      {e.supersedes && <p>Supersedes: <EvidenceLink id={e.supersedes} snapshot={snapshot} /> — original preserved.</p>}
      <p className="small muted">Relationships are reported metadata, not cryptographic verification.</p>
    </section>
    <details><summary>Technical metadata</summary><p className="small muted">Bounded display preview. This JSON is escaped text; it is not an exact-byte verification of the original artifact.</p><pre>{JSON.stringify(e.metadata, null, 2).slice(0, 8192)}</pre></details>
  </>;
}
export function Releases({ snapshot }: { snapshot: Snapshot }) {
  return <>
    <Title label="Model releases" intro="Exact weights, immutable revisions and scoped verification reports belong together. Availability comes from supplied release metadata.">A release is more than a download.</Title>
    <div className="release-grid">{snapshot.releases.map((r, i) => <GlowCard as="article" customSize glowColor={r.role === 'base' ? 'blue' : 'purple'} className="release-card" key={r.id} id={'release-' + r.id}>
      <Label>Role / {String(i + 1).padStart(2, '0')}</Label><h2>{r.role === 'base' ? 'Base model' : 'Conversational model'}</h2><p className="muted">{r.role === 'base' ? 'The pretrained model for completion.' : 'The conversational model, with an explicit base parent.'}</p><hr />
      <h3>{r.availability === 'not-released' ? 'Not released yet.' : r.availability === 'withdrawn' ? 'Release withdrawn.' : 'Release metadata supplied.'}</h3>
      {r.availability !== 'available' ? <><p className="muted">Downloads and generation become available after genuine release metadata and supported instructions are supplied.</p><button disabled className="unavailable">Download unavailable</button></> : r.identity && <>
        <p className="mono small">{r.id}</p>{snapshot.mode === 'fixture' && <p className="missing">Synthetic future-release example. Downloads remain disabled.</p>}
        <dl><div><dt>Repository</dt><dd className="mono">{r.identity.repository}</dd></div><div><dt>Immutable revision</dt><dd className="mono">{r.identity.revision}</dd></div><div><dt>Release root</dt><dd className="mono">{r.identity.root}</dd></div><div><dt>Supported runtime</dt><dd>{r.identity.runtime}</dd></div></dl>
        <External url={r.identity.inventoryUrl}>File inventory</External><External url={r.identity.licenseUrl}>License</External><External url={r.identity.modelCardUrl}>Model card</External>
        {r.parentReleaseId && <p>Base parent: {snapshot.releases.some(x => x.id === r.parentReleaseId) ? <a href={'#/releases?release=' + r.parentReleaseId} onClick={() => setTimeout(() => document.getElementById('release-' + r.parentReleaseId)?.scrollIntoView(), 0)}>{r.parentReleaseId}</a> : <span className="missing">Missing release reference: {r.parentReleaseId}</span>}</p>}
        <ul className="file-list">{r.identity.files.map(f => <li key={f.path}><code>{f.path}</code><span>{f.sizeBytes.toLocaleString()} bytes</span>{snapshot.mode === 'public-snapshot' ? <External url={immutableFileUrl(r, f.path)}>Download pinned file</External> : <button disabled>Download unavailable (fixture)</button>}</li>)}</ul>
        <h3>Scoped verification reports</h3>{!r.checkIds.length && <p>No checks reported for this release.</p>}{r.checkIds.map(id => { const c = snapshot.checks.find(c => c.id === id); return c ? <CheckFacts key={id} check={c} snapshot={snapshot} /> : <p key={id} className="missing">Missing check reference: {id}</p>; })}
      </>}
    </GlowCard>)}</div>
    <hr /><section className="stack"><Label>When released</Label><h2>The identity you should be able to inspect.</h2><ol className="requirements">{['Exact repository and immutable revision', 'Release root, inventory, file sizes and license', 'Supported runtime and scoped verification reports', 'Base-parent identity for the conversational model'].map(s => <li key={s}>{s}</li>)}</ol></section>
  </>;
}
const profileGuide = [
  ['artifact-identity', 'Artifact integrity & publisher identity', 'Checks declared artifact bytes and publisher identity.', 'Does not establish how training was performed.', 'Artifacts, digests and identity/trust metadata.'],
  ['data-reconstruction', 'Data reconstruction', 'Checks reconstruction of declared prepared data.', 'Does not establish training replay or answer accuracy.', 'Pinned source inputs and preparation recipe.'],
  ['sampled-replay', 'Sampled replay', 'Replays a declared sample of training computation.', 'Does not establish complete end-to-end replay.', 'Selected checkpoints, data and replay environment.'],
  ['complete-replay', 'Complete end-to-end replay', 'Reproduces the declared pipeline through final artifacts.', 'Does not establish overall model safety.', 'Complete inputs, recipe and required compute.'],
  ['inference-reproduction', 'Inference reproduction', 'Reproduces an inference result under supplied settings.', 'Does not establish answer correctness.', 'Exact release, runtime and decoding settings.'],
] as const;
export function Verification({ snapshot }: { snapshot: Snapshot }) {
  return <>
    <Title label="Verification guide" intro="Verification is a set of scoped procedures. It is not one universal green badge.">Choose the check. Know its limits.</Title>
    <div className="profiles">{profileGuide.map(([id, title, establishes, limit, requires], i) => <section className="profile" key={id}>
      <div><Label>Profile / {String(i + 1).padStart(2, '0')}</Label><h2>{title}</h2></div>
      <div className="stack compact"><p>{establishes}</p><p className="muted">{limit}</p><p className="small muted">Requires: {requires}</p>
        {snapshot.commands.filter(c => c.profile === id).map(c => <details key={c.id}><summary>Maintainer-approved instructions</summary><pre>{c.executable}</pre>{snapshot.mode === 'public-snapshot' ? <Copy value={c.executable} label="Copy approved command" /> : <p>Synthetic command examples are not offered for execution.</p>}<ul>{c.prerequisites.map(p => <li key={p}>{p}</li>)}</ul><p>Expected scope: {c.expectedScope}</p><p>Resources ({c.resources.kind}): {c.resources.description}</p><SourceInfo source={c.resources.measurementSource} /><SourceInfo source={c.source} /><ul>{c.validationEvidenceIds.map(e => <li key={e}><EvidenceLink id={e} snapshot={snapshot} /></li>)}</ul></details>)}
      </div>
    </section>)}</div>
    {snapshot.commands.length === 0 && <Note title="Instructions pending release."><p>Executable commands appear only when maintainers supply approved, tested instructions. Full replay requires a suitable environment and compute; it is not a browser action.</p></Note>}
    <section className="stack"><Label>Supplied check results</Label><h2>Reports you can inspect.</h2><p className="muted">Fetching a report does not mean this visitor recomputed verification.</p>{snapshot.checks.length ? <div className="check-grid">{snapshot.checks.map(c => <details key={c.id}><summary><ResultText result={c.result} /> · {c.title} · {c.scope}</summary><CheckFacts check={c} snapshot={snapshot} /></details>)}</div> : <State title="No checks reported">There is no successful verification conclusion for an empty set of checks.</State>}</section>
  </>;
}
export function InferenceUnavailable() {
  return <>
    <Title label="Inference">Generation, when it can be reproduced.</Title>
    <section className="inference-panel"><Label>Public production state</Label><h2>Generation is unavailable.</h2><p className="intro">Generation is unavailable until a verified model release and supported inference backend are connected.</p><hr /><h3>Required before enabling</h3><p className="mono">Exact release identity · Supported backend · Approved decoding settings</p><p className="muted">Base completion and conversation remain unavailable until then.</p></section>
    <p className="small muted">Development mocks belong in a separate development build. No alternate model is silently substituted.</p>
  </>;
}
