export type InferenceMode = 'completion' | 'conversation';

export interface InferenceCapabilities {
  mock: boolean;
  available: boolean;
  releaseIds: string[];
  modes: InferenceMode[];
  note: string;
}

export interface GenerationRequest {
  releaseId: string;
  mode: InferenceMode;
  prompt?: string;
  messages?: { role: 'user' | 'assistant'; content: string }[];
}

export interface GenerationResponse {
  text: string;
  releaseId: string;
  disclaimer: string;
}

export class GenerationUnavailable extends Error {
  constructor(message = 'Generation is unavailable until a verified model release is connected') {
    super(message);
    this.name = 'GenerationUnavailable';
  }
}

export class IdentityMismatch extends Error {
  constructor() {
    super('Adapter returned a different release identity than the one selected');
    this.name = 'IdentityMismatch';
  }
}

export interface InferenceAdapter {
  capabilities(): Promise<InferenceCapabilities>;
  generate(request: GenerationRequest, signal?: AbortSignal): Promise<GenerationResponse>;
}
