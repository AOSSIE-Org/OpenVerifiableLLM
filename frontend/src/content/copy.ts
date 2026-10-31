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
      'is a language model whose training anyone can recompute. It\'s small, 23 million parameters, and it was trained on a single pass over English Wikipedia, then on a short set of conversations.',
      'We split training into 37 segments and saved a checkpoint at each of the 38 boundaries. Every checkpoint records the model\'s state, where it was in the data, and the hash of the checkpoint before it, so the checkpoints link into a chain. Training is deterministic: give someone the same code, data and type of GPU, and they get the same bytes back.',
      'So far we\'ve recomputed all 37 segments ourselves on a second machine, and every one matched. Nobody outside the project has checked yet, and we\'d like that to change.',
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
    h2: 'Here\'s what the evidence supports today, and where it stops.',
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
      'The released weights came out of the recorded run.',
      'The training data is exactly the Wikipedia dump we say it is.',
      'The record hasn\'t changed since it was stamped on 20 October.',
    ],
    notShownTitle: 'What it doesn\'t show',
    notShownItems: [
      'That the model\'s answers are right. It\'s small and often gets facts wrong.',
      'That it reproduces on other hardware. You need the same type of GPU.',
      'That anyone outside the project agrees. No one has checked independently yet.',
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
    h2: 'You can go as deep as you like: a two-minute demo, a ten-minute check of the release, or an hour recomputing real training.',
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
        desc: 'This rebuilds one segment from its saved starting point and compares every byte with what we published.',
        action: {
          label: 'Open the RunPod template',
          href: 'https://console.runpod.io/deploy?template=mai1ozh35g',
        },
      },
    ],
  },
  verification: {
    kicker: 'Verification',
    h2: 'Every segment matched when we recomputed it, but so far we\'re the only ones who have checked.',
    subtitle: 'When we say "verified" we mean recomputed and matched. It doesn\'t mean the model\'s answers are right.',
    captionNotice: 'In total there are 46 checks across 37 segments, all matching, and none of them independent.',
    warningNotice: 'These results are self-reported and not yet signed. We plan to add signed results and to choose which segments get checked using public randomness.',
    oddsTitle: 'How many independent checks would catch a bad segment?',
    oddsFormula: 'P = 1 − (36/37)^M, with N = 37',
    oddsDescription: 'Detection odds: 95% at 110 random checks, 99% at 169. Current independent checks: 0.',
    hardwareTitle: 'Which hardware reproduces the bits',
    hardwareSubtitle: 'Bit-exact results need the same kind of GPU, so a mismatch on other hardware is expected rather than a problem.',
  },
  timeline: {
    kicker: 'Training timeline',
    h2: 'Training didn\'t run in one clean go: the machine restarted once, and the code changed partway through. Both segments still matched.',
    restartTitle: 'Host restart',
    restartTag: 'Segment 10',
    restartDesc: 'Training picked up on a fresh machine from the saved checkpoint.',
    codeChangeTitle: 'Code change',
    codeChangeTag: 'Segment 19 onward',
    codeChangeDesc: 'Each segment ran the exact code recorded for it.',
    timestampNote: 'The timestamps prove the record existed by 20 October 2026. They don\'t tell you when or how the training itself ran.',
  },
  exclusion: {
    kicker: 'Verifiable exclusion',
    h2: 'We trained a second copy of the model with one article left out, and we can show it never saw that article.',
    desc: 'The fork removed 3I/ATLAS before training reached it. Everything else is identical.',
    removedArticleLabel: 'Loss on the removed article',
    removedArticleNote: 'Higher, as you\'d expect: the fork never saw it.',
    controlArticlesLabel: 'Loss on 50 control articles',
    controlArticlesNote: 'Barely moved.',
    whyHoldsTitle: 'Why we can say this',
    whyHoldsItems: [
      'The first checkpoints, 0 to 30, are identical byte for byte.',
      'The removal itself is written into the chain.',
      'Any other gap in the data is rejected.',
      'Every segment of the fork matched when recomputed.',
    ],
    limitsTitle: 'What it can\'t do',
    limitsItems: [
      'It shows this model was trained without the article. Nothing more.',
      'It can\'t make a model forget something it has already learned.',
      'If a run has already seen the text, you have to retrain from an earlier checkpoint.',
      'It\'s a single run, so there\'s no estimate of the variance yet.',
    ],
    evidenceLabel: 'Evidence:',
    evidencePath: 'release/unlearning-3i-atlas/',
  },
  downloads: {
    kicker: 'Downloads',
    h2: 'Each file is listed with its SHA-256, so you can check that what you downloaded is what we released.',
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
