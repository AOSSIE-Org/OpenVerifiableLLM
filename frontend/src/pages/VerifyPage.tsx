import type { CheckResult } from '../data/contracts';
import { FAQ, LAYERS, SCOPE_LABEL } from '../content/copy';
import { StatusMark } from '../layout/StatusMark';
import { useSnapshot } from '../snapshot';

const PROFILES: {
  id: string;
  title: string;
  means: string;
  doesNot: string;
  technical: string;
}[] = [
  {
    id: 'artifact-identity',
    title: 'Artifact identity',
    means: 'The published files match their listed checksums, and they were published by the expected project account.',
    doesNot: 'Does not prove how those files were trained.',
    technical: 'artifact and publisher identity',
  },
  {
    id: 'data-reconstruction',
    title: 'Data reconstruction',
    means: 'Starting from the public source files, the same cleaned training text can be produced.',
    doesNot: 'Does not prove the downloaded model is the result of that training.',
    technical: 'data reconstruction',
  },
  {
    id: 'sampled-replay',
    title: 'Sampled replay',
    means: 'Only the listed slice of training was run again in the stated setup and matched.',
    doesNot: 'Does not replace a complete replay.',
    technical: 'sampled replay',
  },
  {
    id: 'full-replay',
    title: 'Complete replay',
    means: 'Every listed step from data through training was rebuilt and run again on the stated computers, and the files matched.',
    doesNot: 'This is not a click in the browser. It needs the stated machines and time. A cost estimate is not a bill.',
    technical: 'complete end-to-end replay',
  },
  {
    id: 'inference-reproduction',
    title: 'Inference reproduction',
    means: 'Using the identified model files and settings, a saved example output comes out again.',
    doesNot: 'A live chat box is not this check, and it does not make answers true.',
    technical: 'inference reproduction',
  },
];

export function VerifyPage() {
  const snapshot = useSnapshot();
  return (
    <article>
      <p className="kicker">Verification</p>
      <h1>What each check means</h1>
      <p>
        Verification is not the same as the project’s evidence record, and not the same as
        inference. Each profile below is a different claim. This page explains them; it does not
        run them. No copy-paste command is shown until the maintainer supplies one that has been
        tested.
      </p>
      {PROFILES.map((profile) => {
        const check = snapshot.checks.find((item) => item.profile === profile.id);
        const result: CheckResult = check?.result ?? 'NOT_RUN';
        return (
          <section className="profile" key={profile.id}>
            <h2>{profile.title}</h2>
            <p>
              <StatusMark result={result} /> ·{' '}
              {check ? SCOPE_LABEL[check.scope] : 'not in this record'}
            </p>
            <p>
              <strong>If this passes:</strong> {profile.means}
            </p>
            <p>
              <strong>It does not mean:</strong> {profile.doesNot}
            </p>
            {check ? <p>{check.explanation}</p> : <p>This record has no check of this kind yet.</p>}
            <details>
              <summary>Technical name</summary>
              <p className="step-meta">{profile.technical}</p>
            </details>
            <p className="step-meta">How to run it: instructions pending — no invented command.</p>
          </section>
        );
      })}

      <h2>Evidence, verification, inference</h2>
      <div className="layers">
        {LAYERS.map((item) => (
          <article className="panel" key={item.title}>
            <h3>{item.title}</h3>
            <p>{item.body}</p>
          </article>
        ))}
      </div>

      <h2>Questions</h2>
      <div className="faq">
        {FAQ.map((item) => (
          <details key={item.q}>
            <summary>{item.q}</summary>
            <p>{item.a}</p>
          </details>
        ))}
      </div>
    </article>
  );
}
