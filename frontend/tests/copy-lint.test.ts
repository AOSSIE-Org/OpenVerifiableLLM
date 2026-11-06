import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('Copy Rules Lint', () => {
  const read = (path: string) => readFileSync(resolve(process.cwd(), path), 'utf8');
  const copyContent = read('src/content/copy.ts');
  const claimsContent = read('src/content/claims.ts');

  // Pinned path exception: the path is real, the word is not a claim
  const ALLOWED_PINNED_PATH = 'release/unlearning-3i-atlas/';

  const BANNED_PATTERNS = [
    { word: 'trustworthy', regex: /\btrustworthy\b/i },
    { word: 'guaranteed', regex: /\bguaranteed\b/i },
    { word: 'secure', regex: /\bsecure\b/i },
    { word: 'revolutionary', regex: /\brevolutionary\b/i },
    { word: 'fully', regex: /\bfully\b/i },
    { word: 'proven-safe', regex: /\bproven-safe\b/i },
    { word: 'unlearning', regex: /\bunlearning\b/i },
    { word: 'one file', regex: /\bone\s+file\b/i },
  ];

  for (const [name, content] of [['copy.ts', copyContent], ['claims.ts', claimsContent]] as const) {
    const sanitized = content.replaceAll(ALLOWED_PINNED_PATH, 'REDACTED_ALLOWED_PATH');
    for (const { word, regex } of BANNED_PATTERNS) {
      it(`does not contain banned term "${word}" in ${name}`, () => {
        expect(sanitized.match(regex), `Found banned word "${word}" in ${name}`).toBeNull();
      });
    }

    // These all come from the snapshot. Typing one here would let the page disagree with it.
    it(`does not type evidence values by hand in ${name}`, () => {
      for (const value of [/\b37\b/, /\b38\b/, /\b46\b/, /\b110\b/, /\b169\b/, /\b30\b/, /\b23\b/, /\b19\b/, /96dc282a/, /20 Oct/, /28 Oct/, /31 Oct/, /v1\.\d+\.\d+/, /RTX \d+/]) {
        expect(content.match(value), `${value} is typed into ${name}`).toBeNull();
      }
    });
  }

  it('does not state a blanket match in static copy', () => {
    expect(copyContent).not.toMatch(/\b(all|every|both)\b[^'"`]*\bmatch/i);
  });
});
