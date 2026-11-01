import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

describe('Copy Rules Lint', () => {
  const copyPath = resolve(process.cwd(), 'src/content/copy.ts');
  const copyContent = readFileSync(copyPath, 'utf8');

  // Pinned path exception
  const ALLOWED_PINNED_PATH = 'release/unlearning-3i-atlas/';

  // Remove the allowed pinned path before checking
  const sanitized = copyContent.replaceAll(ALLOWED_PINNED_PATH, 'REDACTED_ALLOWED_PATH');

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

  for (const { word, regex } of BANNED_PATTERNS) {
    it(`does not contain banned term "${word}" in copy`, () => {
      const match = sanitized.match(regex);
      expect(match, `Found banned word "${word}" in copy.ts`).toBeNull();
    });
  }

  it('contains the authorized pinned evidence path release/unlearning-3i-atlas/', () => {
    expect(copyContent).toContain(ALLOWED_PINNED_PATH);
  });
});
