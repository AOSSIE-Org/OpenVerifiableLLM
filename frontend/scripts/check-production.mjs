import { readFile, readdir } from 'node:fs/promises';
const files = (await readdir('dist/assets')).filter(f => f.endsWith('.js'));
const code = (await Promise.all(files.map(f => readFile('dist/assets/' + f, 'utf8')))).join('\n');
for (const forbidden of ['Generate scripted example', 'createMockAdapter', 'Synthetic backend error', 'Exercise the interface.', 'synthetic-example/not-a-project-release']) {
  if (code.includes(forbidden)) throw new Error('Development mock/scenario leaked into production: ' + forbidden);
}
console.log('PASS: production JS excludes development inference UI, adapter and future-release scenarios.');
