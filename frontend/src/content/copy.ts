// Static copy for the OpenVerifiableLLM site. Anything that states an evidence
// result, count, date or revision lives in claims.ts instead, worked out from the data.
export const copy = {
  header: {
    title: 'OpenVerifiable',
    suffix: 'LLM',
    tagline: 'By AOSSIE',
    source: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM',
  },
  contents: {
    title: 'Contents',
    items: [
      { label: 'How it works', href: '#how' },
      { label: 'Results', href: '#overview' },
      { label: 'Training', href: '#timeline' },
      { label: 'Reproduce', href: '#check' },
      { label: 'Release', href: '#downloads' },
      { label: 'FAQ', href: '#faq' },
    ],
  },
  article: {
    title: 'OpenVerifiableLLM',
    from: 'An AOSSIE project',
    lead: [
      'is a language model whose training anyone can recompute.',
      "It's small, 23 million parameters, trained on English Wikipedia and a few conversations. You don't have to take our word for how it was made.",
    ],
    factsTitle: 'Key facts',
    primaryAction: 'Explore the results',
    secondaryAction: 'Source on GitHub',
    evidenceLine: 'Evidence pinned to Hugging Face revision',
  },
  how: {
    title: 'How it works',
    intro: 'Training is cut into pieces that anyone can rerun and compare.',
  },
  faq: {
    h2: 'Common questions',
    stillTitle: 'Still stuck?',
    discord: 'Ask on Discord',
    discordHref: 'https://discord.gg/hjUhu33uAn',
    issue: 'Open an issue',
    issueHref: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM/issues',
  },
  overview: {
    kicker: 'Results',
    dataTitle: 'Training data',
    dataRebuilt: 'Rebuilt exactly from the Wikipedia dump. Prepared data',
    shownTitle: 'What this shows',
    notShownTitle: "What it doesn't show",
  },
  check: {
    kicker: 'Check it yourself',
    intro: 'Pick how much time you have. Only the last one redoes training.',
  },
  verification: {
    coverageTitle: 'Who has checked each segment',
    warningNotice: 'Self-reported and not yet signed.',
    oddsTitle: 'How many checks would catch a bad segment',
    hardwareTitle: 'Which hardware reproduces the bits',
    hardwareNote: 'Only the same GPU model gives identical bits; a mismatch elsewhere is normal.',
  },
  timeline: {
    kicker: 'Training history',
    baseTitle: 'Base model finished',
    finalTitle: 'Final model',
    restartTitle: 'Host failure',
    restartDesc: 'Training resumed from the saved checkpoint.',
    codeChangeTitle: 'Code change',
    codeChangeDesc: 'The CPU loader and thread pool changed. Later segments were rechecked.',
  },
  exclusion: {
    kicker: 'Verifiable exclusion',
    h2: 'We trained a second copy of the model with one article taken out, and we can show it never saw it.',
    removedArticleLabel: 'Loss on the removed article',
    whyHoldsTitle: 'Why we can say this',
    limitsTitle: "What it can't do",
    limitsItems: [
      'It shows this model never saw the article. Nothing beyond that.',
      "It can't make a model forget something it already learned.",
      "It's one run, so there's no variance estimate yet.",
    ],
    evidenceLabel: 'Evidence:',
    explain: "This doesn't make a model forget anything. It shows that if you leave a text out early enough, you can prove you did.",
  },
  downloads: {
    kicker: 'Release',
    intro: 'Run sha256sum on each file and compare it with the hash below.',
  },
  footer: {
    org: 'An AOSSIE project',
    license: 'OpenVerifiableLLM is licensed under GPL-3.0.',
    contactEmail: 'contact@aossie.org',
    projectHeading: 'Project',
    communityHeading: 'Community',
    helpHeading: 'Get help',
    communityLinks: [
      { label: 'Contribution guidelines', href: 'https://github.com/AOSSIE-Org/Info/blob/main/Rules/Contribution.md' },
      { label: 'Code of Conduct', href: 'https://github.com/AOSSIE-Org/Info/blob/main/CODE_OF_CONDUCT.md' },
      { label: 'AI usage policy', href: 'https://github.com/AOSSIE-Org/Info/blob/main/Rules/AI.md' },
      { label: 'About AOSSIE', href: 'https://aossie.org/en/about' },
    ],
  },
};
