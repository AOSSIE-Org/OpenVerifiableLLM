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
      'is a language model whose training anyone can recompute. It\'s a small one, 23 million parameters, trained on English Wikipedia and then on a short set of conversations. The point isn\'t how clever it is. The point is that you don\'t have to take our word for how it was made.',
    ],
    factsTitle: 'Key facts',
    factsCta: 'Try checking a segment →',
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
  how: {
    title: 'How it works',
    steps: [
      {
        title: '1. Training is cut into pieces',
        text: 'We trained the model in 37 segments and saved a checkpoint after each one, so there are 38 snapshots of the model along the way.',
      },
      {
        title: '2. Each checkpoint points back',
        text: 'Every checkpoint stores the hash of the one before it. Change any of them and the chain no longer adds up.',
      },
      {
        title: '3. Anyone can redo a piece',
        text: 'Start from a saved checkpoint, run the same code and data on the same kind of GPU, and you should get the exact same bytes we did.',
      },
    ],
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
        a: 'Someone recomputed a segment and got exactly the bytes we published. It doesn\'t say anything about whether the model\'s answers are any good.',
      },
      {
        q: 'Who has checked it so far?',
        a: 'Only us so far. We\'ve recomputed every segment on a second machine, but nobody outside the project has sent in a result yet.',
      },
      {
        q: 'What does the 10-minute check prove?',
        a: 'That the files you downloaded match the record we timestamped on 20 October. It doesn\'t redo any training. The one-hour check does that.',
      },
      {
        q: 'Why does it need the same GPU type?',
        a: 'Different GPUs round floating-point numbers in slightly different ways, so the bits drift. On an RTX 5090 they match exactly. On a 4090, an H100 or a CPU they won\'t, and that\'s normal.',
      },
      {
        q: 'Are its answers any good?',
        a: 'Not especially. It\'s tiny and gets facts wrong a lot. This project is about proving how a model was trained, not about building a smart one.',
      },
      {
        q: "Why should I check if you already did?",
        a: 'Because a project checking its own work doesn\'t count for much. One result from someone outside would mean more than all 46 of ours.',
      },
    ],
  },
  overview: {
    kicker: "What's proven",
    subtitle: 'Recomputed bit for bit on a second RTX 5090 by the project; 0 independent checks so far.',
    explain: 'A segment is about a thirty-seventh of the training run. To recompute one, you start from the checkpoint before it, run the same code on the same data, and compare the checkpoint you end up with against ours. If a single bit is different, it counts as a mismatch.',
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
      'That it matches on other kinds of GPU.',
      'That anyone outside the project agrees yet.',
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
    intro: 'Pick how much time you have. The first two run on any laptop. Only the last one actually redoes training, and for that you rent an RTX 5090 for about an hour.',
    ladder: [
      {
        time: '2 min',
        meta: 'Free · any laptop',
        title: 'Try the model',
        chips: [{ label: '○ Demo, checks nothing', type: 'h' }],
        desc: 'Run the model on your own machine. It\'s fun to poke at, but it doesn\'t prove anything.',
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
        desc: 'Download the release and check it against the record we timestamped. Nothing gets retrained.',
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
        desc: 'Rebuilds one segment from its saved checkpoint and compares every byte with ours.',
        action: {
          label: 'Open the RunPod template',
          href: 'https://console.runpod.io/deploy?template=mai1ozh35g',
        },
      },
    ],
  },
  verification: {
    kicker: 'Verification',
    intro: 'Every segment matched when we recomputed it. The gap is that we\'re the only ones who have checked so far, and that\'s the part we most want help with.',
    explain: 'Say someone had tampered with one of the 37 segments. If people check segments at random, 110 checks give a 95% chance of landing on the bad one, and 169 checks give 99%. Right now there are no independent checks, which is why the curve starts at zero.',
    captionNotice: '46 checks across 37 segments, all matching, none independent.',
    warningNotice: 'Self-reported and not yet signed.',
    oddsTitle: 'How many independent checks would catch a bad segment?',
    oddsFormula: 'P = 1 − (36/37)^M, with N = 37',
    oddsDescription: 'Detection odds: 95% at 110 random checks, 99% at 169. Current independent checks: 0.',
    hardwareTitle: 'Which hardware reproduces the bits',
    hardwareNote: 'Only the same GPU model gives identical bits. Other hardware rounds numbers a little differently, so a mismatch there is normal.',
  },
  timeline: {
    kicker: 'Training timeline',
    intro: 'Training didn\'t run in one go. The machine restarted once, and the code was updated partway through. We recomputed the segments around both, and they still matched.',
    h2: 'One machine restart and one code change. Both segments still matched.',
    restartTitle: 'Host restart',
    restartTag: 'Segment 10',
    codeChangeTitle: 'Code change',
    codeChangeTag: 'Segment 19 onward',
    restartDesc: 'Picked up from the saved checkpoint on a new machine.',
    codeChangeDesc: 'Each segment ran the code recorded for it.',
    timestampNote: 'Timestamps prove the record existed by 20 October 2026, not when training ran.',
  },
  exclusion: {
    kicker: 'Verifiable exclusion',
    h2: 'We trained a second copy of the model with one article taken out, and we can show it never saw it.',
    desc: 'The fork removed 3I/ATLAS before training reached it. Everything else is identical.',
    removedArticleLabel: 'Loss on the removed article',
    removedArticleNote: 'Higher, as you\'d expect: the fork never saw it.',
    controlArticlesLabel: 'Loss on 50 control articles',
    controlArticlesNote: 'Barely moved.',
    whyHoldsTitle: 'Why we can say this',
    whyHoldsItems: [
      'Checkpoints 0 to 30 are identical in both copies.',
      'The removal is recorded in the chain.',
      'Every segment of the second copy matched.',
    ],
    limitsTitle: 'What it can\'t do',
    limitsItems: [
      'It shows this model never saw the article. Nothing beyond that.',
      'It can\'t make a model forget something it already learned.',
      'It\'s one run, so there\'s no variance estimate yet.',
    ],
    evidenceLabel: 'Evidence:',
    explain: 'People often ask whether a model can be made to forget something. This doesn\'t do that. It shows something narrower: if you know early enough that a text shouldn\'t be in training, you can leave it out and prove that you did.',
    evidencePath: 'release/unlearning-3i-atlas/',
  },
  downloads: {
    kicker: 'Downloads',
    intro: 'Each file comes with its SHA-256 hash. After downloading, run sha256sum on it and check the result matches the hash below.',
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
