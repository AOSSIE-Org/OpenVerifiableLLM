// Content copy for OpenVerifiableLLM site
export const copy = {
  header: {
    title: 'OpenVerifiable',
    suffix: 'LLM',
    tagline: 'By AOSSIE',
    links: [
      { label: 'Source', href: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM' },
      { label: 'Evidence', href: 'https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence' },
    ],
    themeToggleAria: 'Switch between dark and light theme',
  },
  contents: {
    title: 'Contents',
    items: [
      { label: "What's proven", href: '#overview' },
      { label: 'Check it yourself', href: '#check' },
      { label: 'Verification', href: '#verification' },
      { label: 'Training timeline', href: '#timeline' },
      { label: 'Verifiable exclusion', href: '#exclusion' },
      { label: 'Downloads', href: '#downloads' },
      { label: 'Common questions', href: '#faq' },
      { label: 'External links', href: '#links' },
    ],
  },
  article: {
    title: 'OpenVerifiableLLM',
    from: 'An AOSSIE project',
    kicker: 'Release v1.0.1 · 28 October 2026 · Record timestamped 20 October 2026',
    lead: [
      'is a language model whose training anyone can recompute. It is a 23-million-parameter model trained on one exact pass over English Wikipedia, followed by a short conversation phase.',
      'Training was split into 37 segments between 38 recorded checkpoints. Each checkpoint stores the model state, the data position and the hash of the checkpoint before it, so the checkpoints form a chain. Because training is deterministic, anyone with the same GPU type can recompute a segment and get exactly the same bytes.',
      'All 37 segments were recomputed on a second machine by the project and matched. No independent checker has submitted a result yet.',
    ],
    factsTitle: 'Key facts',
    factsCta: 'Check a segment yourself →',
    inShort: 'In short',
    infobox: {
      caption: 'OpenVerifiableLLM',
      release: 'v1.0.1 (28 October 2026)',
      parameters: '23 million',
      data: 'English Wikipedia, one exact pass',
      checked: 'by the project',
      independent: 'none yet',
      stamped: '20 October 2026 (DigiCert, FreeTSA)',
      license: 'GPL-3.0',
      source: 'GitHub',
    },
  },
  faq: {
    h2: 'Common questions',
    stillTitle: 'More questions:',
    discord: 'Ask on Discord',
    issue: 'Open an issue',
    issueHref: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM/issues',
    items: [
      {
        q: 'What does "verified" mean here?',
        a: 'Someone recomputed a segment and got exactly the same bytes. It says nothing about whether the answers are correct.',
      },
      {
        q: 'Who has checked it so far?',
        a: 'Only the project. No independent checker has submitted a result yet.',
      },
      {
        q: 'What does the 10-minute check prove?',
        a: 'That the released files match the timestamped record. It recomputes no training.',
      },
      {
        q: 'Why does it need the same GPU type?',
        a: 'Bit-exact results depend on the hardware. Other GPUs give different bits, which is expected.',
      },
    ],
  },
  overview: {
    kicker: "What's proven",
    h2: "What you can check, and what you can't yet.",
    subtitle: 'Recomputed bit for bit on a second RTX 5090 by the project; 0 independent checks so far.',
    facts: [
      {
        value: '37',
        total: ' / 37',
        title: 'Segments recomputed bit for bit',
        chip: 'Checked by the project',
      },
      {
        value: 'Exact',
        total: '',
        title: 'Data rebuilt from the Wikipedia dump',
        chip: '8de1d600…',
      },
      {
        value: '20 Oct',
        total: '',
        title: 'Record timestamped',
        chip: 'DigiCert · FreeTSA',
      },
    ],
    shownTitle: 'Shown',
    shownItems: [
      'The weights come from the recorded run.',
      'The data is exactly the stated dump.',
      'The record is unchanged since 20 October.',
    ],
    notShownTitle: 'Not shown',
    notShownItems: [
      "That the model's answers are right.",
      'Results on other hardware.',
      'Any check from outside the project.',
    ],
    counts: [
      { key: 'articles', label: 'Articles' },
      { key: 'targets', label: 'Wikipedia targets' },
      { key: 'conversations', label: 'Conversations' },
      { key: 'loopLines', label: 'Training loop', value: '~630 lines' },
    ],
  },
  check: {
    kicker: 'Check it yourself',
    h2: 'From a laptop demo to a bit-for-bit recompute.',
    ladder: [
      {
        time: '2 min',
        meta: 'Free · any laptop',
        title: 'Try the model',
        chips: [{ label: '○ Demo, checks nothing', type: 'h' }],
        command: 'ollama run ryoari/openverifiablellm',
      },
      {
        time: '10 min',
        meta: 'Free · no GPU',
        title: 'Consistency check',
        chips: [
          { label: '✓ Release matches its record', type: 'm' },
          { label: '○ No training recomputed', type: 'h' },
        ],
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
      },
      {
        time: '1 hour',
        meta: 'About US$1.50 · RTX 5090',
        title: 'The actual verification',
        isKey: true,
        chips: [{ label: '✓ Recomputes a segment bit for bit', type: 'm' }],
        desc: 'Rebuilds one segment and compares every byte with the published result.',
        action: {
          label: 'Open the RunPod template',
          href: 'https://console.runpod.io/deploy?template=mai1ozh35g',
        },
      },
    ],
  },
  verification: {
    kicker: 'Verification',
    h2: 'Every segment matched. Only the project has checked so far.',
    subtitle: '"Verified" means recomputed and matched, not that the answers are correct.',
    captionNotice: '46 checks across 37 segments (all MATCH). 0 checks from independent checkers.',
    warningNotice: 'Results are self-reported and unsigned for now. Signed results and segment selection from public randomness are on the roadmap.',
    oddsTitle: 'How many independent checks would catch a bad segment?',
    oddsFormula: 'P = 1 − (36/37)^M, with N = 37',
    oddsDescription: 'Detection odds: 95% at 110 random checks, 99% at 169. Current independent checks: 0.',
    hardwareTitle: 'Which hardware reproduces the bits',
    hardwareSubtitle: 'A mismatch on other GPUs is expected.',
  },
  timeline: {
    kicker: 'Training timeline',
    h2: 'A host restart and a code change. Both still matched.',
    restartTitle: 'Host restart',
    restartTag: 'Segment 10',
    restartDesc: 'Resumed on a fresh host.',
    codeChangeTitle: 'Code change',
    codeChangeTag: 'Segment 19 onward',
    codeChangeDesc: 'Each segment ran its recorded code.',
    timestampNote: 'Timestamps show the record existed by 20 October 2026, not when or how training happened.',
  },
  exclusion: {
    kicker: 'Verifiable exclusion',
    h2: 'A second model, provably trained without one article.',
    desc: 'The fork removed 3I/ATLAS before training reached it. Everything else is identical.',
    removedArticleLabel: 'Loss on the removed article',
    removedArticleNote: 'Harder, because the fork never saw it.',
    controlArticlesLabel: 'Loss on 50 control articles',
    controlArticlesNote: 'Practically unchanged.',
    whyHoldsTitle: 'Why the claim holds',
    whyHoldsItems: [
      'Checkpoints 0–30 are shared byte for byte.',
      'The removal is recorded in the chain.',
      'Every other gap in the data is refused.',
      'Every fork segment matched.',
    ],
    limitsTitle: 'Limits',
    limitsItems: [
      'It proves a model was trained without the article.',
      "It doesn't erase what a model already learned.",
      'Text the run had already seen means retraining from an earlier save point.',
      'One run, no variance estimate.',
    ],
    evidenceLabel: 'Evidence:',
    evidencePath: 'release/unlearning-3i-atlas/',
  },
  downloads: {
    kicker: 'Downloads',
    h2: 'Every file ships with its SHA-256.',
    files: [
      {
        name: 'ovl-23m-chat.gguf',
        source: 'GitHub release v1.0.1',
        actionLabel: 'Download',
        url: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/download/v1.0.1/ovl-23m-chat.gguf',
      },
      {
        name: 'Modelfile',
        source: 'GitHub release v1.0.1',
        actionLabel: 'Download',
        url: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/download/v1.0.1/Modelfile',
      },
      {
        name: 'conversation-model.pt',
        source: 'Hugging Face, revision 96dc282a',
        actionLabel: 'Open',
        url: 'https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence/blob/96dc282a/release/simple-run/conversation-model.pt',
      },
      {
        name: 'ryoari/openverifiablellm',
        source: 'Ollama',
        actionLabel: 'Open',
        note: 'Install with ollama run',
        url: 'https://ollama.com/ryoari/openverifiablellm',
      },
    ],
  },
  footer: {
    org: 'An AOSSIE project',
    license: 'OpenVerifiableLLM is licensed under GPL-3.0.',
    evidenceNote: 'Evidence pinned to Hugging Face revision',
    contactEmail: 'contact@aossie.org',
    projectHeading: 'Project',
    communityHeading: 'Community',
    contactHeading: 'Contact',
    projectLinks: [
      { label: 'Source code on GitHub', href: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM' },
      { label: 'Release v1.0.1', href: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/tag/v1.0.1' },
      { label: 'Evidence on Hugging Face', href: 'https://huggingface.co/datasets/AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence' },
      { label: 'Model on Ollama', href: 'https://ollama.com/ryoari/openverifiablellm' },
    ],
    communityLinks: [
      { label: 'Contribution guidelines', href: 'https://github.com/AOSSIE-Org/Info/blob/main/Rules/Contribution.md' },
      { label: 'Code of Conduct', href: 'https://github.com/AOSSIE-Org/Info/blob/main/CODE_OF_CONDUCT.md' },
      { label: 'AI usage policy', href: 'https://github.com/AOSSIE-Org/Info/blob/main/Rules/AI.md' },
      { label: 'About AOSSIE', href: 'https://aossie.org/en/about' },
    ],
  },
};
