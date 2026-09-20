import { GenerationUnavailable, type InferenceAdapter, type InferenceCapabilities } from './types';

export class DisabledAdapter implements InferenceAdapter {
  async capabilities(): Promise<InferenceCapabilities> {
    return {
      mock: false,
      available: false,
      releaseIds: [],
      modes: ['completion', 'conversation'],
      note: 'Generation is unavailable until a verified model release is connected',
    };
  }

  async generate(): Promise<never> {
    throw new GenerationUnavailable();
  }
}
