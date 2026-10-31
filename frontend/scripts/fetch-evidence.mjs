// Manual script: node scripts/fetch-evidence.mjs
// Pinned to Hugging Face dataset revision 96dc282a.
// Writes public/data/evidence.json with source hashes and validated numbers.

import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const DATASET = 'AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence';
const REVISION = '96dc282a';
const BASE_URL = `https://huggingface.co/datasets/${DATASET}/raw/${REVISION}/`;

const PREP_PATH = 'prepared/8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68/preparation.json';

const FILES_TO_FETCH = [
  'release/simple-run/chain.jsonl',
  'release/simple-run/verifications.jsonl',
  'release/simple-run/hardware-map.jsonl',
  'release/simple-run/release-hashes.txt',
  'release/simple-run/anchor.json',
  'release/simple-run/README.md',
  'release/unlearning-3i-atlas/loo-results.json',
  PREP_PATH,
];

function sha256(content) {
  return createHash('sha256').update(content).digest('hex');
}

async function fetchFile(relPath) {
  const url = `${BASE_URL}${relPath}`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${res.status} ${res.statusText}`);
  }
  const text = await res.text();
  return {
    path: relPath,
    text,
    sha256: sha256(text),
  };
}

export async function buildEvidenceData() {
  console.log(`Fetching pinned evidence from Hugging Face (${DATASET} @ ${REVISION})...`);
  const fetched = {};
  for (const relPath of FILES_TO_FETCH) {
    process.stdout.write(`  Fetching ${relPath}... `);
    const file = await fetchFile(relPath);
    fetched[relPath] = file;
    console.log(`OK (${file.sha256.slice(0, 8)})`);
  }

  // 1. Parse chain.jsonl
  const chainRawLines = fetched['release/simple-run/chain.jsonl'].text.trim().split('\n');
  const chain = chainRawLines.map((line) => {
    const entry = JSON.parse(line);
    const entryHash = sha256(line);
    return {
      index: entry.index,
      kind: entry.kind,
      control: entry.control,
      state: entry.state,
      model: entry.model,
      previous: entry.previous,
      time: entry.time,
      entryHash,
    };
  });

  if (chain.length !== 38) {
    throw new Error(`Expected 38 chain entries, found ${chain.length}`);
  }
  for (let i = 1; i < chain.length; i++) {
    if (chain[i].previous !== chain[i - 1].entryHash) {
      throw new Error(`Broken chain previous link at index ${i}`);
    }
  }

  // 2. Parse verifications.jsonl
  const verificationsRawLines = fetched['release/simple-run/verifications.jsonl'].text.trim().split('\n');
  const rawVerifications = verificationsRawLines.map(line => JSON.parse(line));

  // Segment coverage
  const segmentCoverage = [];
  for (let s = 0; s < 37; s++) {
    const checksForSegment = rawVerifications.filter(v => v.segment === s);
    const projectMatch = checksForSegment.some(c => c.checker.startsWith('project') && c.result === 'MATCH');
    const independentChecks = checksForSegment.filter(c => !c.checker.startsWith('project'));

    segmentCoverage.push({
      segment: s,
      projectResult: projectMatch ? 'MATCH' : 'MISMATCH',
      independentResult: independentChecks.length > 0 ? (independentChecks.every(c => c.result === 'MATCH') ? 'MATCH' : 'MISMATCH') : null,
      checks: checksForSegment,
    });
  }

  // 3. Parse hardware-map.jsonl
  const hwRawLines = fetched['release/simple-run/hardware-map.jsonl'].text.trim().split('\n');
  const hardwareMap = hwRawLines.map(line => JSON.parse(line));

  // 4. Parse release-hashes.txt
  const releaseHashesText = fetched['release/simple-run/release-hashes.txt'].text.trim().split('\n');
  const releaseHashes = {};
  for (const line of releaseHashesText) {
    const parts = line.trim().split(/\s+/);
    if (parts.length >= 2) {
      releaseHashes[parts[1]] = parts[0];
    }
  }

  // Also include the GitHub release v1.0.1 model assets
  releaseHashes['ovl-23m-chat.gguf'] = '541d49c97b87cd4ed0acb90b4327535ee20b8c11c452e2583d460154e9d980de';
  releaseHashes['Modelfile'] = 'df4dacfc3191e56b1dc1195764ea66ca1a7695856be04ecec17d8dde4c66fc8b';

  // 5. Parse anchor.json
  const anchor = JSON.parse(fetched['release/simple-run/anchor.json'].text);

  // 6. Parse preparation.json
  const prep = JSON.parse(fetched[PREP_PATH].text);
  const preparation = {
    articles: prep.corpus.counts.included,
    wikipediaTargets: prep.streams.wikipedia.targets,
    conversations: prep.conversation_selection.conversations.train,
    trainingLoopLines: '~630 lines',
    preparedDataId: '8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68',
  };

  // 7. Parse loo-results.json
  const loo = JSON.parse(fetched['release/unlearning-3i-atlas/loo-results.json'].text);
  const exclusion = {
    removedArticle: '3I/ATLAS',
    originalRemovedLoss: Number(loo.original_base.removed_loss[0].toFixed(4)),
    forkRemovedLoss: Number(loo.fork_base.removed_loss[0].toFixed(4)),
    removedLossDelta: Number((loo.fork_base.removed_loss[0] - loo.original_base.removed_loss[0]).toFixed(4)),
    originalControlLossMean: Number(loo.original_base.control_loss_mean.toFixed(4)),
    forkControlLossMean: Number(loo.fork_base.control_loss_mean.toFixed(4)),
    controlLossDelta: Number((loo.fork_base.control_loss_mean - loo.original_base.control_loss_mean).toFixed(4)),
    controlArticlesCount: loo.control_documents,
    evidencePath: 'release/unlearning-3i-atlas/',
  };

  const evidenceData = {
    schemaVersion: 1,
    dataset: DATASET,
    revision: REVISION,
    fetchedAt: new Date().toISOString(),
    sourceHashes: Object.fromEntries(
      Object.entries(fetched).map(([path, f]) => [path, f.sha256])
    ),
    preparation,
    chain,
    verifications: {
      totalChecks: rawVerifications.length,
      uniqueSegments: 37,
      projectChecksCount: rawVerifications.filter(v => v.checker.startsWith('project')).length,
      independentChecksCount: rawVerifications.filter(v => !v.checker.startsWith('project')).length,
      coverage: segmentCoverage,
      rawChecks: rawVerifications,
    },
    hardwareMap,
    anchor: {
      ...anchor,
      timestamp: '2026-10-20 14:38:43 UTC',
      authorities: ['DigiCert', 'FreeTSA (RFC 3161)'],
    },
    exclusion,
    releaseHashes,
  };

  return evidenceData;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const data = await buildEvidenceData();
    const outPath = join(__dirname, '../public/data/evidence.json');
    await mkdir(dirname(outPath), { recursive: true });
    await writeFile(outPath, JSON.stringify(data, null, 2) + '\n', 'utf8');
    console.log(`\nWrote evidence data to ${outPath}`);
  } catch (err) {
    console.error('Failed to fetch evidence:', err);
    process.exit(1);
  }
}
