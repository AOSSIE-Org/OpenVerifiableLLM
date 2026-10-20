import { describe, expect, it } from 'vitest';
import { capabilities, createMockAdapter } from '../src/inference/mock';
import { checkedGenerate, type GenerationRequest } from '../src/inference/adapter';
const request: GenerationRequest = { mode: 'base', release: capabilities.releases.base, prompt: 'Public test input', messages: [], settings: { strategy: 'greedy', maxOutputTokens: 128 } };
describe('isolated inference adapter', () => {
  it('returns explicitly mock output with exact identity and settings', async () => {
    const r = await checkedGenerate(createMockAdapter('success', 0), request);
    expect(r.mock).toBe(true); expect(r.release).toEqual(request.release); expect(r.settings).toEqual(request.settings);
  });
  it.each(['error', 'timeout', 'identity-mismatch'] as const)('rejects %s', async scenario => expect(checkedGenerate(createMockAdapter(scenario, 0), request)).rejects.toThrow());
  it('surfaces empty output without replacing it', async () => expect((await checkedGenerate(createMockAdapter('empty', 0), request)).text).toBe(''));
  it('cancels pending and already-aborted requests', async () => {
    const c = new AbortController(); const task = checkedGenerate(createMockAdapter('success', 1000), request, c.signal); c.abort();
    await expect(task).rejects.toMatchObject({ name: 'AbortError' });
    await expect(checkedGenerate(createMockAdapter('success', 0), request, c.signal)).rejects.toMatchObject({ name: 'AbortError' });
  });
  it('rejects unsupported input, mode, settings and selected release', async () => {
    const invalid = [
      { ...request, prompt: '' }, { ...request, prompt: 'x'.repeat(2001) },
      { ...request, mode: 'unsupported' }, { ...request, release: { ...request.release, root: 'f'.repeat(64) } },
      { ...request, settings: { strategy: 'sampling', maxOutputTokens: 128 } },
      { ...request, settings: { strategy: 'greedy', maxOutputTokens: 129 } },
    ];
    for (const r of invalid) await expect(checkedGenerate(createMockAdapter('success', 0), r as GenerationRequest)).rejects.toThrow();
  });
});
