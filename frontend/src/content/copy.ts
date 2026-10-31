// All page text in one place, so the copy lint can check it.
// Numbers come from public/data/evidence.json, not from this file.

const HF = 'https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence';
const GH = 'https://github.com/AOSSIE-Org/OpenVerifiableLLM';

export const links = {
  github: GH,
  release: `${GH}/releases/tag/v1.0.1`,
  evidence: HF,
  ollama: 'https://ollama.com/ryoari/openverifiablellm',
  runpod: 'https://console.runpod.io/deploy?template=mai1ozh35g',
  issues: `${GH}/issues`,
  discord: 'https://discord.gg/hjUhu33uAn',
  aossie: 'https://aossie.org',
  /** Link to a file in the evidence dataset at the pinned revision. */
  evidenceFile: (revision: string, path: string) => `${HF}/blob/${revision}/${path}`,
};

export const copy = {
  site: {
    title: 'OpenVerifiableLLM',
    by: 'An AOSSIE project',
  },
  contents: [
    { id: 'overview', label: 'Overview' },
    { id: 'check', label: 'Check it yourself' },
    { id: 'verification', label: 'Verification' },
    { id: 'timeline', label: 'Training timeline' },
    { id: 'exclusion', label: 'Verifiable exclusion' },
    { id: 'downloads', label: 'Downloads' },
    { id: 'terms', label: 'Terms used on this page' },
    { id: 'references', label: 'References' },
  ],
  lead: {
    tagline: 'From the published evidence record, Hugging Face revision',
    // The bold name is rendered by the page; this follows it.
    body:
      'is a 23M-parameter language model trained in one exact pass over English Wikipedia, followed by a short conversation phase. Its training was recorded in 37 segments. Each segment can be recomputed on the same GPU type and compared bit for bit with the published result.',
    status:
      'All 37 segments have been recomputed and matched, all of them by the project on a second machine. No independent checks have been submitted yet.',
    caution:
      '"Verified" on this page means recomputed bit for bit and matched. It does not mean the model\'s answers are correct; it is a small model that is often wrong about facts.',
  },
  overview: {
    facts: [
      '37 of 37 segments recomputed bit for bit on a second machine, by the project.',
      'Exact data provenance: the prepared data rebuilds byte-identical from the raw Wikipedia dump.',
      'A timestamped release record: the chain head was stamped by two RFC 3161 time-stamping authorities.',
    ],
    limitsTitle: 'Limits',
    limits: [
      'It is a small model and is often wrong about facts.',
      'Checks need the same GPU type (RTX 5090). Other hardware gives different bits, which is expected.',
      'No independent checks yet. Every check so far was run by the project.',
      'Check results are self-reported and unsigned for now.',
    ],
    countsCaption: 'Training data',
    loopNote:
      'The training loop is about 630 lines in simple/ovl.py, plus about 1,000 lines of model, data and initialization helpers.',
  },
  check: {
    intro: 'There are three levels. Only the third recomputes any training.',
    levels: [
      {
        id: 'try',
        title: 'Try it (2 minutes)',
        cost: 'Free, any laptop.',
        proves: 'Nothing. This is a demo.',
        command: 'ollama run ryoari/openverifiablellm',
      },
      {
        id: 'consistency',
        title: 'Consistency check (10 minutes, no GPU)',
        cost: 'Free, CPU only.',
        proves:
          'That the released files match their timestamped record. It does not recompute any training.',
        command: `pip install huggingface_hub torch numpy rfc8785==0.1.4 tokenizers==0.23.2 mwparserfromhell==0.7.2 \\
  "safetensors>=0.4.5,<1.0" "pynacl>=1.5,<2.0"
hf download AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence \\
  --repo-type dataset --include "release/simple-run/*" --exclude "*/ckpt-*" --local-dir ovl
cd ovl/release/simple-run
sha256sum -c --ignore-missing release-hashes.txt
openssl ts -verify -in anchor.digicert.tsr -data anchor.json -CAfile /etc/ssl/certs/ca-certificates.crt
# Verification: OK   (DigiCert stamped anchor.json on 2026-10-20)
git clone -q --depth 1 --branch v1.0.1 https://github.com/AOSSIE-Org/OpenVerifiableLLM ovl-code
OVL_DEVICE=cpu python ovl-code/simple/ovl.py verify --run . --weights conversation-model.pt
# anchor anchor.json: chain head MATCH (38 entries)
# weights conversation-model.pt: MATCH chain entry 37`,
        note: 'Commands copied from the README\'s "Getting started". Use ovl.py from v1.0.1.',
      },
      {
        id: 'actual',
        title: 'The actual verification (about 1 hour, about US$1.50)',
        cost: 'A rented RTX 5090.',
        proves:
          'Recomputes one training segment and compares every byte with the published result.',
        action: { label: 'Open the one-click RunPod template', href: 'https://console.runpod.io/deploy?template=mai1ozh35g' },
      },
    ],
  },
  verification: {
    coverageTitle: 'Coverage map',
    coverageIntro:
      'Segment k runs from boundary k to boundary k + 1. Each tile is one segment, labelled by who checked it.',
    selfReported:
      'Check results are self-reported and unsigned for now. Signed results and segment choice from public randomness are on the roadmap.',
    oddsTitle: 'Detection odds',
    oddsIntro:
      'If one segment were wrong, how likely is it that random independent checks would find it? With N = 37 segments and M independent checks, each choosing a segment at random:',
    oddsFormula: 'P = 1 − ((N − 1) / N)^M',
    hardwareTitle: 'Hardware map',
    hardwareIntro: 'Which hardware reproduces the bits. A mismatch on a different GPU type is expected.',
    checksTitle: 'All recorded checks',
  },
  timeline: {
    intro:
      'The 38 boundaries recorded in the chain. Two events happened during training; both were shown harmless because the segments around them matched.',
    events: [
      { title: 'Host restart', where: 'Segment 10', desc: 'Training resumed on a fresh host. The segment matched.' },
      { title: 'Code change', where: 'Segment 19 onward', desc: 'Each segment ran its recorded code. All of them matched.' },
    ],
    timestampNote:
      'The timestamps show that the record existed by 20 October 2026. They do not show when or how training happened.',
    tableTitle: 'Chain boundaries',
  },
  exclusion: {
    intro:
      'A second model, trained without one article. The article removed was one that training had not reached yet.',
    whyTitle: 'Why the claim holds',
    why: [
      'Checkpoints 0–30 are shared byte for byte with the original run.',
      'The removal is recorded in the chain.',
      'Every other gap in the data is refused.',
      'Every fork segment matched.',
    ],
    limitsTitle: 'Limits',
    limits: [
      'It proves a model was trained without the article. It does not erase what a model already learned.',
      'Removing text the run had already seen means retraining from an earlier save point.',
      'One run, so there is no variance estimate.',
    ],
    folderNote: 'The folder keeps its old name because it is pinned evidence.',
  },
  downloads: {
    intro: 'Check each file against its SHA-256 before use.',
    files: [
      {
        name: 'ovl-23m-chat.gguf',
        source: 'GitHub release v1.0.1',
        url: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/download/v1.0.1/ovl-23m-chat.gguf',
      },
      {
        name: 'Modelfile',
        source: 'GitHub release v1.0.1',
        url: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/download/v1.0.1/Modelfile',
      },
      {
        name: 'conversation-model.pt',
        source: 'Hugging Face',
        hfPath: 'release/simple-run/conversation-model.pt',
      },
      {
        name: 'base-model.pt',
        source: 'Hugging Face',
        hfPath: 'release/simple-run/base-model.pt',
      },
    ],
    ollama: 'Ollama: ryoari/openverifiablellm (built from the GGUF above).',
  },
  terms: [
    { term: 'Verified', def: 'Recomputed bit for bit and matched. It never means the model\'s answers are correct.' },
    { term: 'Consistency check', def: 'The laptop check: the release matches its timestamped record. Nothing is recomputed.' },
    { term: 'Actual verification', def: 'Recomputing a training segment on the same GPU type and comparing it byte for byte.' },
    { term: 'Project / independent', def: 'Who ran a check. "Project" means the project\'s own machines; "independent" means anyone else. There are no independent checks yet.' },
    { term: 'Verifiable exclusion', def: 'A model provably trained without a given document, shown by a recorded fork of the training chain.' },
  ],
  references: {
    intro: 'Every number on this page is read from these files at the pinned revision and validated on load.',
    labels: {
      'release/simple-run/chain.jsonl': 'Training chain: 38 boundaries',
      'release/simple-run/verifications.jsonl': 'Segment check results',
      'release/simple-run/hardware-map.jsonl': 'Hardware map',
      'release/simple-run/release-hashes.txt': 'SHA-256 of every release file',
      'release/simple-run/anchor.json': 'Timestamped chain head',
      'release/simple-run/README.md': 'Wording of claims and limits',
      'release/unlearning-3i-atlas/loo-results.json': 'Verifiable exclusion: loss results',
    } as Record<string, string>,
    preparationLabel: 'Data preparation counts',
  },
  footer: {
    license: 'OpenVerifiableLLM is licensed under GPL-3.0.',
    snapshot: 'Evidence snapshot fetched',
    links: [
      { label: 'Source code', href: GH },
      { label: 'Release v1.0.1', href: `${GH}/releases/tag/v1.0.1` },
      { label: 'Evidence on Hugging Face', href: HF },
      { label: 'Model on Ollama', href: 'https://ollama.com/ryoari/openverifiablellm' },
      { label: 'Report an issue', href: `${GH}/issues` },
      { label: 'Discord', href: 'https://discord.gg/hjUhu33uAn' },
    ],
  },
};
