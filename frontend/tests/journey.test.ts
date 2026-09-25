import { describe, expect, it } from 'vitest';
import { worstResult } from '../src/content/copy';

describe('worstResult', () => {
  it('lets FAIL dominate', () => {
    expect(worstResult(['PASS', 'FAIL', 'NOT_RUN'])).toBe('FAIL');
  });

  it('does not treat mixed PASS and NOT_RUN as a full pass', () => {
    expect(worstResult(['PASS', 'NOT_RUN'])).toBe('NOT_RUN');
  });

  it('returns PASS only when every result is PASS', () => {
    expect(worstResult(['PASS', 'PASS'])).toBe('PASS');
  });
});
