// Build gate: site-data.json must validate strictly (with chain hashes recomputed) and every
// rendered ledger claim must pass the wording and number rules. Any problem exits non-zero.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { validateStrict } from '../src/data/validate';
import { auditClaims } from '../src/content/audit';

const sha = (s: string) => createHash('sha256').update(s).digest('hex');
const raw: unknown = JSON.parse(readFileSync('public/data/site-data.json', 'utf8'));
const data = validateStrict(raw, sha);
const problems = auditClaims(data);
if (problems.length) {
  console.error(`check-data: ${problems.length} ledger problem(s)`);
  for (const p of problems) console.error(' -', p);
  process.exit(1);
}
console.log(`check-data: OK (${data.chain.entries} entries, ${data.verifications.rowCount} check rows, ${data.verifications.tiles.length} tiles)`);
