import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync } from 'node:fs';
import { resolve, join } from 'node:path';

// The site is a reference page: no scroll-linked effects and no animation.
const srcDir = resolve(process.cwd(), 'src');
const files = (dir: string): string[] =>
  readdirSync(dir, { withFileTypes: true }).flatMap(e =>
    e.isDirectory() ? files(join(dir, e.name)) : [join(dir, e.name)],
  );

describe('No motion', () => {
  const all = files(srcDir);
  const code = all.filter(f => /\.(tsx?|css)$/.test(f)).map(f => [f, readFileSync(f, 'utf8')] as const);

  it('has no scroll listeners or animation frames in components', () => {
    for (const [f, text] of code) {
      expect(text, f).not.toMatch(/addEventListener\(\s*['"]scroll/);
      expect(text, f).not.toMatch(/requestAnimationFrame|IntersectionObserver/);
    }
  });

  it('declares no keyframes or transitions in CSS', () => {
    for (const [f, text] of code.filter(([f]) => f.endsWith('.css'))) {
      expect(text, f).not.toMatch(/@keyframes/);
      expect(text, f).not.toMatch(/transition\s*:(?!\s*none)/);
      expect(text, f).not.toMatch(/scroll-behavior\s*:\s*smooth/);
    }
  });
});
