export type ModelIdentity = { id: string; repository: string; revision: string; root: string };
export type Message = { role: 'user' | 'assistant'; content: string };
export type Settings = { strategy: 'greedy'; maxOutputTokens: number };
export type GenerationRequest = { release: ModelIdentity; mode: 'base' | 'conversation'; prompt: string; messages: Message[]; settings: Settings };
export type GenerationResponse = { text: string; release: ModelIdentity; settings: Settings; mock: boolean };
export type Capabilities = {
  mock: boolean; releases: Record<'base' | 'conversation', ModelIdentity>; modes: ('base' | 'conversation')[];
  inputLimit: number; outputLimit: number; strategy: 'greedy';
};
export interface InferenceAdapter {
  capabilities(): Promise<Capabilities>;
  generate(request: GenerationRequest, signal?: AbortSignal): Promise<GenerationResponse>;
}
export function sameIdentity(a: ModelIdentity, b: ModelIdentity) {
  return a.id === b.id && a.repository === b.repository && a.revision === b.revision && a.root === b.root;
}
export async function checkedGenerate(adapter: InferenceAdapter, request: GenerationRequest, signal?: AbortSignal) {
  const c = await adapter.capabilities();
  if (!c.modes.includes(request.mode)) throw new Error('Unsupported generation mode.');
  if (!sameIdentity(c.releases[request.mode], request.release)) throw new Error('Selected release identity is unsupported.');
  if (request.settings.strategy !== c.strategy || !Number.isInteger(request.settings.maxOutputTokens) || request.settings.maxOutputTokens < 1 || request.settings.maxOutputTokens > c.outputLimit) throw new Error('Unsupported decoding settings.');
  const input = request.mode === 'base' ? request.prompt : request.messages.map(m => m.content).join('');
  if (!input.trim() || input.length > c.inputLimit) throw new Error('Input is empty or exceeds the adapter limit.');
  if (request.mode === 'conversation' && request.messages.some(m => !['user', 'assistant'].includes(m.role))) throw new Error('Unsupported message role.');
  const response = await adapter.generate(request, signal);
  if (signal?.aborted) throw new DOMException('Request cancelled', 'AbortError');
  if (!response || typeof response.text !== 'string' || !response.release || !sameIdentity(response.release, request.release)) throw new Error('Returned release identity mismatch.');
  if (!response.settings || response.settings.strategy !== request.settings.strategy || response.settings.maxOutputTokens !== request.settings.maxOutputTokens) throw new Error('Returned decoding settings mismatch.');
  if (response.mock !== c.mock) throw new Error('Adapter response mode mismatch.');
  return response;
}
