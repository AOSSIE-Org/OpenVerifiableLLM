import { useSnapshot } from '../snapshot';

export function TryPage() {
  const snapshot = useSnapshot();
  const chat = snapshot.releases.find((item) => item.role === 'chat');

  return (
    <article>
      <p className="kicker">Inference</p>
      <h1>Try</h1>
      <p>
        <strong>What.</strong> Inference is using released model files to produce text — including a
        later, locked chat box on this page.
      </p>
      <p>
        <strong>Why.</strong> That is a different claim from evidence (the public record) and from
        verification (rebuilding or re-running training). A live chat would not inspect data or
        checkpoints, and it would not make answers true.
      </p>
      <p>
        <strong>Current status.</strong> Chat model: {chat?.availability === 'available' ? 'listed' : 'not released'}.
        Send is off. This page will not fall back to another model.
      </p>
      <div className="composer" aria-disabled="true">
        <p>
          <strong>Chat</strong>
        </p>
        <label>
          Message
          <textarea rows={5} disabled placeholder="Locked until a checked release exists" />
        </label>
        <button type="button" disabled>
          Send
        </button>
      </div>
      <details>
        <summary>Technical note</summary>
        <p className="step-meta">
          A labelled mock adapter exists in tests only. The public build does not call it.
          Inference reproduction, when it exists, is a saved-example check — not this composer.
        </p>
      </details>
    </article>
  );
}
