// Every sentence that states an evidence result or count. Each one is worked out
// from the records, so the page can't claim more than the data says.
import type { CheckOutcome } from '../data/contracts';
import type { Facts } from '../data/facts';

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

export interface LadderStep {
  time: string;
  meta: string;
  title: string;
  desc: string;
  chips: { label: string; type: 'm' | 'h' }[];
  isKey?: boolean;
  command?: string;
  needs?: string;
  action?: { label: string; href: string };
}

export function claimsFor(f: Facts) {
  const independentSoFar = `${plural(f.independentChecks, 'independent check')} so far`;
  const noneIndependent = f.independentChecks === 0;
  const unmatched = f.projectMismatched + f.projectUnchecked;
  const anyMismatchingChecks = f.projectChecksMismatched + f.independentChecksMismatched;
  const mismatchBits = [
    f.projectMismatched > 0 ? `${f.projectMismatched} mismatched` : '',
    f.projectUnchecked > 0 ? `${f.projectUnchecked} not checked` : '',
  ].filter(Boolean);

  return {
    lead: `It's small, ${f.modelMillions} million parameters, trained on English Wikipedia and a few conversations. You don't have to take our word for how it was made.`,
    hero: {
      kicker: `Release ${f.releaseTag} · ${f.releaseDate} · Record timestamped ${f.stampDate}`,
      recomputed: 'segments recomputed, by the project',
      independent: `independent ${f.independentChecks === 1 ? 'check' : 'checks'} so far`,
      checks:
        f.projectChecksMismatched === 0
          ? 'project checks, all matching'
          : `project checks, ${f.projectChecksMismatched} mismatching`,
      stampedLabel: `record timestamped by ${f.authorities.join(' and ')}`,
    },
    how: [
      { title: 'Data', text: 'An exact Wikipedia dump. The project rebuilt the training data from it, byte for byte.' },
      { title: 'Training', text: `Run in ${f.segments} segments, with a checkpoint saved after each.` },
      { title: 'Chain', text: 'Each record links to the hash of the one before, so any change breaks it.' },
      { title: 'Timestamp', text: `${f.authorities.join(' and ')} stamped the record on ${f.stampDate}.` },
      { title: 'Recompute', text: `Rerun a segment on an ${f.checkGpu} and compare the bytes.` },
    ],
    results: {
      subtitle:
        unmatched === 0
          ? `Recomputed bit for bit on a second ${f.checkGpu} by the project; ${independentSoFar}.`
          : `${f.projectMatched} of ${f.segments} segments recomputed bit for bit on a second ${f.checkGpu} by the project; ${independentSoFar}.`,
      explain: `A segment is one of ${f.segments} slices of training. Start from the checkpoint before it, rerun it, and compare the result with ours. One different bit is a mismatch.`,
      projectSummary: `Project: ${f.projectMatched} / ${f.segments} matched`,
      projectExtra: mismatchBits.length > 0 ? ` (${mismatchBits.join(', ')})` : '',
      independentSummary: `Independent: ${f.independentChecked} / ${f.segments} checked`,
      independentExtra: f.independentMismatched > 0 ? ` (${f.independentMismatched} mismatched)` : '',
      mapLabel:
        `${f.segments} segments. Project checks: ${f.projectMatched} matched, ${f.projectMismatched} mismatched, ${f.projectUnchecked} not checked. ` +
        `Independent checks: ${f.independentMatched} matched, ${f.independentMismatched} mismatched, ${f.segments - f.independentChecked} not checked.`,
      notice: [
        plural(f.totalChecks, 'check'),
        `across ${plural(f.segments, 'segment')},`,
        anyMismatchingChecks === 0 ? 'all matching,' : `${anyMismatchingChecks} mismatching,`,
        noneIndependent ? 'none independent.' : `${f.independentChecks} independent.`,
      ].join(' '),
      noticeClose:
        unmatched > 0
          ? `${unmatched === 1 ? '1 segment has' : `${unmatched} segments have`} no matching result. The map shows which.`
          : noneIndependent
            ? "Every segment matched in the project's own checks, but nobody outside has checked. That's where we need help."
            : 'Every project check matched, and outside checks have started. More are needed.',
      shown: [
        f.allProjectMatched
          ? `The project reran all ${f.segments} segments and got the recorded bytes.`
          : `Not all of it: ${unmatched === 1 ? '1 segment has' : `${unmatched} segments have`} no matching result.`,
        'The project rebuilt the training data from the Wikipedia dump and got the same bytes.',
        `${f.authorities.join(' and ')} timestamped the record on ${f.stampDate}, so it existed by then.`,
      ],
      notShown: [
        "That the answers are right. It's small and often wrong.",
        'That it matches on other kinds of GPU.',
        noneIndependent ? 'That anyone outside the project agrees yet.' : 'That outside checkers cover every segment.',
      ],
    },
    odds: {
      checksToday: `${f.independentChecks === 1 ? 'check' : 'checks'} today`,
      explain: `If one segment had been tampered with, ${f.checksFor95} independent checks would catch it 95% of the time and ${f.checksFor99} would catch it 99%. That assumes each checker picks a segment at random and reports honestly. ${
        noneIndependent
          ? 'There are no independent checks yet, so the curve starts at zero.'
          : `There ${f.independentChecks === 1 ? 'is' : 'are'} ${plural(f.independentChecks, 'independent check')} so far.`
      }`,
      formula: `P = 1 − (${f.segments - 1}/${f.segments})^M, with N = ${f.segments}`,
      description: `Detection odds: 95% at ${f.checksFor95} random checks, 99% at ${f.checksFor99}. Current independent checks: ${f.independentChecks}.`,
      today: `Today: ${plural(f.independentChecks, 'independent check')}`,
    },
    faq: [
      {
        q: 'What does "verified" mean here?',
        a: 'Someone recomputed a segment and got exactly our bytes. It says nothing about answer quality.',
      },
      {
        q: 'Who has checked it so far?',
        a: noneIndependent
          ? 'Only us. Nobody outside the project has sent a result yet.'
          : `Us, plus ${plural(f.independentChecks, 'independent check')} across ${plural(f.independentChecked, 'segment')}.`,
      },
      {
        q: 'What does the 10-minute check prove?',
        a: `That your files match the record stamped on ${f.stampDate}. It doesn't redo training.`,
      },
      {
        q: 'Why does it need the same GPU type?',
        a: `GPUs round numbers slightly differently. On an ${f.checkGpu} the bits match; on other hardware they drift, and that is normal.`,
      },
      {
        q: 'Are its answers any good?',
        a: "Not really. It's tiny and often wrong. The project is about how it was trained.",
      },
      {
        q: 'Why should I check if you already did?',
        a: noneIndependent
          ? `One outside result would mean more than all ${f.projectChecks} of ours.`
          : `Each outside result means more than any of our ${f.projectChecks}.`,
      },
    ],
    snapshotNote: 'This page checks that its snapshot is consistent with itself, not that it is authentic. To verify the evidence, run the checks below.',
    timestampNote: `Timestamps prove the record existed by ${f.stampDate}, not when training ran.`,
    ladder: ladderFor(f),
    forkWhy: [
      `Entries 0 to ${f.forkFromEntry} of its record are identical to the original.`,
      'The removal is recorded in the chain.',
      f.forkSegmentsMatched === f.forkSegments
        ? `All ${f.forkSegments} segments from the split onward matched, recomputed by the project.`
        : `${f.forkSegmentsMatched} of the ${f.forkSegments} segments from the split onward matched, recomputed by the project.`,
    ],
    forkLimits: f.forkAnchored ? [] : ['Its record has no timestamp of its own at this revision.'],
    downloads: [
      {
        name: 'ovl-23m-chat.gguf',
        source: `GitHub release ${f.releaseTag}`,
        url: `https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/download/${f.releaseTag}/ovl-23m-chat.gguf`,
      },
      {
        name: 'Modelfile',
        source: `GitHub release ${f.releaseTag}`,
        url: `https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/download/${f.releaseTag}/Modelfile`,
      },
      {
        name: 'conversation-model.pt',
        source: `Hugging Face, revision ${f.revision}`,
        url: `https://huggingface.co/datasets/${f.dataset}/blob/${f.revision}/release/simple-run/conversation-model.pt`,
      },
      {
        name: 'ryoari/openverifiablellm',
        source: 'Ollama',
        note: 'Install with ollama run',
        url: 'https://ollama.com/ryoari/openverifiablellm',
      },
    ],
    footerLinks: [
      { label: 'Source code on GitHub', href: 'https://github.com/AOSSIE-Org/OpenVerifiableLLM' },
      { label: `Release ${f.releaseTag}`, href: `https://github.com/AOSSIE-Org/OpenVerifiableLLM/releases/tag/${f.releaseTag}` },
      { label: 'Evidence on Hugging Face', href: f.evidenceUrl },
      { label: 'Model on Ollama', href: 'https://ollama.com/ryoari/openverifiablellm' },
    ],
  };
}

function ladderFor(f: Facts): LadderStep[] {
  return [
    {
      time: '2 min',
      meta: 'Free · any laptop',
      title: 'Try the model',
      chips: [{ label: '○ Demo, checks nothing', type: 'h' }],
      desc: 'Chat with it on your own machine.',
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
      desc: 'Check the release against the stamped record.',
      // Pinned to the revision on this page, because the dataset's main branch has moved on.
      command: `pip install huggingface_hub torch numpy
hf download ${f.dataset} \\
  --repo-type dataset --revision ${f.revisionFull} \\
  --include "release/simple-run/*" --exclude "*/ckpt-*" --local-dir ovl
cd ovl/release/simple-run
sha256sum -c --ignore-missing release-hashes.txt
git clone -q --depth 1 --branch ${f.releaseTag} https://github.com/AOSSIE-Org/OpenVerifiableLLM ovl-code
OVL_DEVICE=cpu python ovl-code/simple/ovl.py verify --run . --weights conversation-model.pt`,
      needs: 'Needs Python 3.12 or newer, git, openssl and sha256sum. It passes only if the last line starts with verify PASS. On Windows, verify finds no CA bundle for the timestamps, so use Linux or macOS.',
    },
    {
      time: '1 hour',
      meta: `About US$1–1.50 · ${f.checkGpu}`,
      title: 'The actual verification',
      isKey: true,
      chips: [{ label: '✓ Recomputes a segment bit for bit', type: 'm' }],
      desc: 'Rebuild one segment and compare every byte with ours.',
      action: {
        label: 'Open the RunPod template',
        href: 'https://console.runpod.io/deploy?template=mai1ozh35g',
      },
    },
  ];
}

export function timelineClaims(f: Facts, outcomes: { failure: CheckOutcome; codeChange: CheckOutcome }) {
  const bothMatched = outcomes.failure === 'MATCH' && outcomes.codeChange === 'MATCH';
  return {
    intro: `The host failed once and the code changed once. ${
      bothMatched ? 'The segments they touched still matched.' : 'Not every segment they touched matched.'
    }`,
    caption: `${plural(f.segments, 'segment')}, ${
      f.allProjectMatched
        ? 'all matched by the project'
        : `${f.projectMatched} matched by the project, ${f.projectMismatched} mismatched, ${f.projectUnchecked} not checked`
    }. Evenly spaced, not to time.`,
  };
}

export const removedLossNote = (delta: number): string =>
  delta > 0 ? "Higher, as you'd expect: the fork never saw it." : 'Not higher, so this run does not show the expected effect.';

export const controlLossNote = (delta: number, removedDelta: number): string =>
  Math.abs(delta) < Math.abs(removedDelta) ? 'Barely moved.' : 'Moved as much as the removed article.';
