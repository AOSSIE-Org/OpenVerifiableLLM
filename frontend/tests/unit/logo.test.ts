import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const root = path.resolve(__dirname, '../..');
const mark = 'M8 2H24A6 6 0 0 1 30 8V17.25H23.5V10.45';

describe('approved OpenVerifiableLLM mark', () => {
  it('ships the C-gate in public/logo.svg and favicon.svg', () => {
    for (const name of ['logo.svg', 'favicon.svg']) {
      const svg = readFileSync(path.join(root, 'public', name), 'utf8');
      expect(svg).toContain(mark);
      expect(svg).toContain('#00843D');
      expect(svg).toContain('#FFCD00');
      expect(svg).not.toContain('shape-rendering="crispEdges"');
    }
  });
});
