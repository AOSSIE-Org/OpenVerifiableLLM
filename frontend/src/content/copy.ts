import type { CheckResult, EvidenceScope } from '../data/contracts';

export const SCOPE_LABEL: Record<EvidenceScope, string> = {
  fixture: 'test data',
  pilot: 'practice run',
  production: 'real run',
};

export const PHASE_LABEL: Record<string, string> = {
  source: 'Data',
  prepare: 'Preparation',
  train: 'Training',
  replay: 'Replay',
  release: 'Release',
  other: 'Other',
};

export type JourneyKind = 'evidence' | 'verification' | 'inference';

export interface JourneyStageDef {
  id: string;
  label: string;
  what: string;
  why: string;
  pipelineIds: string[];
  checkProfile: string | null;
  kind: JourneyKind;
}

export const JOURNEY: JourneyStageDef[] = [
  {
    id: 'data',
    label: 'Data',
    what: 'Name the public Wikipedia files used as input.',
    why: 'Later steps cannot be checked if the starting files are unnamed or private. Conversation data is planned for the chat model, not shown as done here.',
    pipelineIds: ['wiki-source', 'wiki-acquire'],
    checkProfile: null,
    kind: 'evidence',
  },
  {
    id: 'preparation',
    label: 'Preparation',
    what: 'Turn those files into the training text and tokens.',
    why: 'Cleaning and tokenizing must be specified, or training cannot be reproduced.',
    pipelineIds: ['extract', 'tokenizer'],
    checkProfile: null,
    kind: 'evidence',
  },
  {
    id: 'training',
    label: 'Training',
    what: 'Train from scratch on that data; save checkpoints.',
    why: 'Released weights should be the output of this run, not an unexplained extra process.',
    pipelineIds: ['init', 'wiki-train', 'chat-train'],
    checkProfile: null,
    kind: 'evidence',
  },
  {
    id: 'replay',
    label: 'Replay',
    what: 'Rebuild the data and run the same training again.',
    why: 'A checksum or signature does not prove the training description is true. This is verification, not the project’s file list.',
    pipelineIds: ['replay'],
    checkProfile: 'full-replay',
    kind: 'verification',
  },
  {
    id: 'release',
    label: 'Release',
    what: 'Publish the model files with checksums.',
    why: 'Without a public folder and version, there are no weights for others to download.',
    pipelineIds: ['publish'],
    checkProfile: null,
    kind: 'evidence',
  },
  {
    id: 'inference',
    label: 'Inference',
    what: 'Reproduce a saved example from released files.',
    why: 'Generating text is not the same as proving how the model was trained. Live chat is not this check.',
    pipelineIds: [],
    checkProfile: 'inference-reproduction',
    kind: 'inference',
  },
];

export function worstResult(results: CheckResult[]): CheckResult {
  if (results.includes('FAIL')) return 'FAIL';
  if (results.includes('UNSUPPORTED')) return 'UNSUPPORTED';
  if (results.includes('UNAVAILABLE')) return 'UNAVAILABLE';
  if (results.length > 0 && results.every((r) => r === 'PASS')) return 'PASS';
  if (results.includes('PASS') && results.some((r) => r !== 'PASS')) return 'NOT_RUN';
  if (results.includes('NOT_RUN') || results.length === 0) return 'NOT_RUN';
  return 'NOT_RUN';
}

export const WHY = [
  {
    title: 'A download is not a proof',
    body: 'A hosted file and a checksum do not show which data was used or how training ran.',
  },
  {
    title: 'A signature is not a replay',
    body: 'A signature names who published a folder. It does not re-run training.',
  },
  {
    title: 'Chat is not verification',
    body: 'Generating text does not inspect data, checkpoints, or training.',
  },
];

export const LAYERS = [
  {
    title: 'Evidence',
    body: 'The public record: files, reports, and links the project published. This website displays that record. It does not re-run training.',
  },
  {
    title: 'Verification',
    body: 'Independent checks of that record — for example rebuilding the data or re-running training. Each check has its own meaning. There is no single green badge.',
  },
  {
    title: 'Inference',
    body: 'Using released weights to produce text. That is not enabled here, and it would not make answers into citations even if it were.',
  },
];

export const FAQ = [
  {
    q: 'What is OpenVerifiableLLM?',
    a: 'A project to train a small language model and keep a public record of how it was made: data, training steps, checkpoints (mid-training saves), and released weights (the downloadable model files). Other people should be able to inspect that record and later reproduce it. It is not a day-to-day chatbot.',
  },
  {
    q: 'Why does that matter?',
    a: 'Most open models give you files and a write-up. You can download the files. You usually cannot check the write-up. If a lab says “trained only on this Wikipedia dump,” you have to trust them. This project tries to make that checkable.',
  },
  {
    q: 'What can I do on this website?',
    a: 'Read the saved public record and follow links to original files. The Verify page explains what each kind of check would mean. This site does not train, sign, or generate text.',
  },
  {
    q: 'What is a practice run versus the real run?',
    a: 'A practice run (pilot) is the team testing the pipeline. The real run (production) is the Wikipedia-to-model release. A pass on a practice run does not finish the real run. Test data (fixture) is synthetic and is not project status.',
  },
  {
    q: 'Is a pass here the same as the old ovllm GREEN verdict?',
    a: 'No. An older command in the repo can print GREEN when file hashes and signatures match. That is artifact identity, not a full rebuild and re-run of Wikipedia training. This site keeps those checks separate.',
  },
  {
    q: 'Who ran the checks shown?',
    a: 'The project team, not an independent outsider, and not your browser. Opening this page does not recompute anything. If a check says it was not recomputed locally, that is correct.',
  },
  {
    q: 'Would a later chat mean the answers are true?',
    a: 'No. Checking how a model was made is not fact-checking what it says, and it does not mean a reply came from a particular Wikipedia article.',
  },
  {
    q: 'Can I run the full check in the browser?',
    a: 'No. Re-running training needs the stated computers and is slow and costly. It will never be a silent click here. A later, labelled checksum of a small published file would only prove that one file’s bytes.',
  },
  {
    q: 'Where are the original files?',
    a: 'On the Evidence page: GitHub and a public Hugging Face dataset. This website only shows those links. It does not host Wikipedia dumps or checkpoints.',
  },
];
