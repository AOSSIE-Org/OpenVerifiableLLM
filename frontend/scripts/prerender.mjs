import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import React from 'react';
import { renderToString } from 'react-dom/server';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..');
const distDir = path.resolve(rootDir, 'dist');
const htmlPath = path.resolve(distDir, 'index.html');
const evidencePath = path.resolve(rootDir, 'public/data/evidence.json');

if (!fs.existsSync(htmlPath)) {
  console.error('Error: dist/index.html not found. Run "vite build" first.');
  process.exit(1);
}

if (!fs.existsSync(evidencePath)) {
  console.error('Error: public/data/evidence.json not found.');
  process.exit(1);
}

const server = await createServer({
  root: rootDir,
  server: { middlewareMode: true },
  appType: 'custom',
});

try {
  const { App } = await server.ssrLoadModule('/src/App.tsx');
  const { validateEvidence } = await server.ssrLoadModule('/src/data/validate.ts');
  const { summarize } = await server.ssrLoadModule('/src/data/facts.ts');

  const rawEvidence = JSON.parse(fs.readFileSync(evidencePath, 'utf8'));
  const evidence = validateEvidence(rawEvidence);
  const facts = summarize(evidence);

  const appHtml = renderToString(
    React.createElement(App, { initialData: { evidence, facts } }),
  );

  let template = fs.readFileSync(htmlPath, 'utf8');

  if (!template.includes('<div id="root"></div>')) {
    console.error('Error: <div id="root"></div> placeholder not found in dist/index.html');
    process.exit(1);
  }

  const serialized = JSON.stringify(rawEvidence).replace(/</g, '\\u003c');
  const embeddedScript = `<script id="__OVL_EVIDENCE__" type="application/json">${serialized}</script>`;

  template = template.replace(
    '<div id="root"></div>',
    `<div id="root">${appHtml}</div>\n    ${embeddedScript}`,
  );

  fs.writeFileSync(htmlPath, template, 'utf8');
  console.log(`Prerendered static HTML into dist/index.html (${appHtml.length} chars).`);
} finally {
  await server.close();
}
