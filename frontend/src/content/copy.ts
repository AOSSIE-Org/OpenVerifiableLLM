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
      'is a language model whose training anyone can recompute: 23 million parameters, trained on one pass over English Wikipedia and a short set of conversations.',
      'Training ran in 37 segments between 38 hashed checkpoints that link into a chain. The same code, data and GPU type give back the same bytes.',
    ],
    factsTitle: 'Key facts',
    factsCta: 'Try checking a segment →',
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
    stillTitle: 'Still stuck?',
    discord: 'Ask on Discord',
    issue: 'Open an issue',
    issueHref: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM/issues',
    items: [
      {
        q: 'What does "verified" mean here?',
        a: 'Someone took a segment, recomputed it from its saved starting point, and got exactly the bytes we published. It says nothing about whether the model\'s answers are any good.',
      },
      {
        q: 'Who has checked it so far?',
        a: 'Just us, for now. We\'ve recomputed every segment, but no independent checker has sent in a result yet.',
      },
      {
        q: 'What does the 10-minute check prove?',
        a: 'It confirms that the files you downloaded match the timestamped record. It doesn\'t recompute any training; for that you need the one-hour check.',
      },
      {
        q: 'Why does it need the same GPU type?',
        a: 'Different GPU architectures do floating-point maths in slightly different ways, so the bits drift apart. On an RTX 5090 they match exactly. On an RTX 4090, an H100 or a CPU they don\'t, and that\'s expected.',
      },
    ],
  },
  overview: {
    kicker: "What's proven",
    h2: 'What the evidence supports, and where it stops.',
    subtitle: 'Recomputed bit for bit on a second RTX 5090 by the project; 0 independent checks so far.',
    facts: [
      {
        value: '37',
        total: ' / 37',
        title: 'Training segments recomputed bit for bit',
        chip: 'Checked by the project',
      },
      {
        value: 'Exact',
        total: '',
        title: 'Training data rebuilt from the Wikipedia dump',
        chip: '8de1d600…',
      },
      {
        value: '20 Oct',
        total: '',
        title: 'Release record timestamped',
        chip: 'DigiCert · FreeTSA',
      },
    ],
    shownTitle: 'What this shows',
    shownItems: [
      'The weights came from the recorded run.',
      'The data is exactly the stated Wikipedia dump.',
      'The record is unchanged since 20 October.',
    ],
    notShownTitle: 'What it doesn\'t show',
    notShownItems: [
      'That the answers are right. It\'s small and often wrong.',
      'Results on other GPUs. You need the same type.',
      'Outside agreement. No independent check yet.',
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
    h2: 'Three ways in: a 2-minute demo, a 10-minute release check, or an hour of real recompute.',
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
        desc: 'Rebuilds one segment and compares every byte with ours.',
        action: {
          label: 'Open the RunPod template',
          href: 'https://console.runpod.io/deploy?template=mai1ozh35g',
        },
      },
    ],
  },
  verification: {
    kicker: 'Verification',
    h2: 'Every segment matched, but only we have checked so far.',
    subtitle: '"Verified" means recomputed and matched, not that the answers are right.',
    captionNotice: '46 checks across 37 segments, all matching, none independent.',
    warningNotice: 'Self-reported and not yet signed.',
    oddsTitle: 'How many independent checks would catch a bad segment?',
    oddsFormula: 'P = 1 − (36/37)^M, with N = 37',
    oddsDescription: 'Detection odds: 95% at 110 random checks, 99% at 169. Current independent checks: 0.',
    hardwareTitle: 'Which hardware reproduces the bits',
    hardwareSubtitle: 'Bit-exact results need the same GPU type, so other hardware is expected to differ.',
  },
  timeline: {
    kicker: 'Training timeline',
    h2: 'One machine restart and one code change. Both segments still matched.',
    restartTitle: 'Host restart',
    restartTag: 'Segment 10',
    restartDesc: 'Training picked up on a fresh machine from the saved checkpoint.',
    codeChangeTitle: 'Code change',
    codeChangeTag: 'Segment 19 onward',
    codeChangeDesc: 'Each segment ran the exact code recorded for it.',
    timestampNote: 'The timestamps prove the record existed by 20 October 2026, not when or how training ran.',
  },
  exclusion: {
    kicker: 'Verifiable exclusion',
    h2: 'A second copy, trained with one article left out, and proof it never saw it.',
    desc: 'The fork removed 3I/ATLAS before training reached it. Everything else is identical.',
    removedArticleLabel: 'Loss on the removed article',
    removedArticleNote: 'Higher, as you\'d expect: the fork never saw it.',
    controlArticlesLabel: 'Loss on 50 control articles',
    controlArticlesNote: 'Barely moved.',
    whyHoldsTitle: 'Why we can say this',
    whyHoldsItems: [
      'Checkpoints 0 to 30 are identical byte for byte.',
      'The removal is written into the chain.',
      'Any other data gap is rejected.',
      'Every fork segment matched.',
    ],
    limitsTitle: 'What it can\'t do',
    limitsItems: [
      'It shows training without the article, nothing more.',
      'It can\'t make a model forget what it learned.',
      'Text already seen means retraining from an earlier checkpoint.',
      'One run, so no variance estimate yet.',
    ],
    evidenceLabel: 'Evidence:',
    evidencePath: 'release/unlearning-3i-atlas/',
  },
  downloads: {
    kicker: 'Downloads',
    h2: 'Each file with its SHA-256, so you can check your download.',
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
    about: 'A language model whose training anyone can recompute, built in the open by AOSSIE.',
    license: 'OpenVerifiableLLM is licensed under GPL-3.0.',
    evidenceNote: 'Evidence pinned to Hugging Face revision',
    contactEmail: 'contact@aossie.org',
    projectHeading: 'Project',
    communityHeading: 'Community',
    contactHeading: 'Contact',
    helpHeading: 'Get help',
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
