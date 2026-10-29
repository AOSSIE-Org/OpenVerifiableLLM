// Copy the claim ledger (owned by Research Lab) and, if present, the plain-language copy file
// (owned by Open Source Ops) into src/content/. The site never edits either; it only renders them.
//   CLAIMS_DIR=/path/to/ovllm-claims npm run claims:sync
import { copyFileSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';

const dir = process.env.CLAIMS_DIR ?? path.resolve('..', '..', 'ovllm-claims');
const ledger = path.join(dir, 'claims.json');
if (!existsSync(ledger)) {
  console.error(`sync-claims: ${ledger} not found (set CLAIMS_DIR)`);
  process.exit(1);
}
const parsed = JSON.parse(readFileSync(ledger, 'utf8'));
if (parsed.schema !== 'ovllm.claims.v1') {
  console.error(`sync-claims: unknown ledger schema ${parsed.schema}`);
  process.exit(1);
}
copyFileSync(ledger, 'src/content/claims.json');
// COPY.md: "## <claim id>" then "- Headline: …" and "- Developer: …" lines.
const copy = path.join(dir, 'COPY.md');
if (existsSync(copy)) {
  const out = {};
  for (const block of readFileSync(copy, 'utf8').split(/^## /m).slice(1)) {
    const id = block.split('\n', 1)[0].trim();
    const headline = /^- Headline: (.*)$/m.exec(block)?.[1]?.trim();
    const developer = /^- Developer: (.*)$/m.exec(block)?.[1]?.trim();
    if (headline) out[id] = { plain: headline, ...(developer ? { developer } : {}) };
  }
  writeFileSync('src/content/copy.json', JSON.stringify(out, null, 1) + '\n');
} else if (!existsSync('src/content/copy.json')) writeFileSync('src/content/copy.json', '{}\n');
console.log(`sync-claims: ${parsed.claims.length} claims from ${ledger}${copy ? `, copy from ${copy}` : ''}`);
