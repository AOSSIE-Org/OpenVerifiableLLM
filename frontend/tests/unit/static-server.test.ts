import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const src = readFileSync(path.resolve(__dirname, '../../scripts/static-server.mjs'), 'utf8');

function contained(root: string, file: string): boolean {
  const rel = path.relative(root, file);
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel));
}

describe('static-server path containment', () => {
  const root = path.resolve('/x/dist');
  it('allows files inside dist', () => {
    expect(contained(root, path.resolve('/x/dist/index.html'))).toBe(true);
    expect(contained(root, root)).toBe(true);
  });
  it('rejects a sibling such as dist-ssr', () => {
    expect(contained(root, path.resolve('/x/dist-ssr/entry.js'))).toBe(false);
    expect(contained(root, path.resolve('/x/dist2/index.html'))).toBe(false);
  });
  it('is the function the server exports', () => {
    expect(src).toContain('export function contained(root, file)');
    expect(src).toContain('path.relative(root, file)');
    expect(src).toMatch(/!rel\.startsWith\('\.\.'\)/);
    expect(src).toContain('fileURLToPath(import.meta.url)');
  });
});
