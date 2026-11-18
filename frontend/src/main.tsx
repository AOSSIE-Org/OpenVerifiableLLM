import { StrictMode } from 'react';
import { createRoot, hydrateRoot } from 'react-dom/client';
import App from './App';
import { applyTheme, initialTheme } from './theme';
import { validateEvidence } from './data/validate';
import { summarize } from './data/facts';
import type { EvidenceData } from './data/contracts';
import './styles/base.css';

const root = document.getElementById('root');
if (!root) throw new Error('root element missing');

// Before the first paint, so the loading and error states follow the reader's theme too.
applyTheme(initialTheme());

let initialData: { evidence: EvidenceData; facts: ReturnType<typeof summarize> } | undefined;
const embedded = document.getElementById('__OVL_EVIDENCE__');
if (embedded?.textContent) {
  try {
    const evidence = validateEvidence(JSON.parse(embedded.textContent));
    const facts = summarize(evidence);
    initialData = { evidence, facts };
  } catch {
    // If embedded snapshot is invalid or missing, App falls back to client fetch.
  }
}

if (root.hasChildNodes() && initialData) {
  hydrateRoot(
    root,
    <StrictMode>
      <App initialData={initialData} />
    </StrictMode>,
  );
} else {
  createRoot(root).render(
    <StrictMode>
      <App initialData={initialData} />
    </StrictMode>,
  );
}
