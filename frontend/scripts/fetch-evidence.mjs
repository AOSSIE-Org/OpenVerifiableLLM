// Manual script: node scripts/fetch-evidence.mjs
// Builds public/data/evidence.json from the pinned Hugging Face revision and the
// pinned GitHub release. Only the pins below are typed by hand: every count, hash,
// date and result comes from the fetched files, and two runs give identical output.
// `npm run validate` then checks the output against itself, and the build fails if
// it doesn't hold. This reads the evidence; it doesn't verify the timestamp signatures
// or recompute training. `simple/ovl.py verify` and the segment checks do that.

import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import checkers from '../src/data/checkers.json' with { type: 'json' };

const __dirname = dirname(fileURLToPath(import.meta.url));

// The pins
const DATASET = 'AOSSIE/openverifiable-enwiki-20260901-20260918-r1-evidence';
const REVISION = '96dc282a';
const REPOSITORY = 'AOSSIE-Org/OpenVerifiableLLM';
const RELEASE_TAG = 'v1.1.0';
const RELEASE_ASSETS = ['ovl-23m-chat.gguf', 'Modelfile'];
const PREPARED_ID = '8de1d60065b1e8e084c31c202e47367f28c8b89f11b4e895873f775bc0f7ba68';
const AUTHORITIES = { digicert: 'DigiCert', freetsa: 'FreeTSA' };

const RUN = 'release/simple-run';
const FORK = 'release/unlearning-3i-atlas';
const PREP_PATH = `prepared/${PREPARED_ID}/preparation.json`;
const STAMP_FILES = Object.keys(AUTHORITIES).map(name => `anchor.${name}.tsr`);

const FILES_TO_FETCH = [
  `${RUN}/chain.jsonl`,
  `${RUN}/verifications.jsonl`,
  `${RUN}/hardware-map.jsonl`,
  `${RUN}/release-hashes.txt`,
  `${RUN}/anchor.json`,
  `${RUN}/README.md`,
  `${RUN}/code/simple/ovl.py`,
  ...STAMP_FILES.map(file => `${RUN}/${file}`),
  `${FORK}/loo-results.json`,
  `${FORK}/verifications.jsonl`,
  `${FORK}/chain.jsonl`,
  `${FORK}/README.md`,
  PREP_PATH,
];

const sha256 = content => createHash('sha256').update(content).digest('hex');

function must(condition, message) {
  if (!condition) throw new Error(message);
}

async function fetchFile(relPath) {
  const url = `https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/${relPath}`;
  const res = await fetch(url);
  must(res.ok, `Failed to fetch ${url}: HTTP ${res.status} ${res.statusText}`);
  const bytes = Buffer.from(await res.arrayBuffer());
  return { path: relPath, bytes, text: bytes.toString('utf8'), sha256: sha256(bytes) };
}

async function fetchJson(url) {
  const headers = { 'User-Agent': 'openverifiablellm-fetch-evidence' };
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  const res = await fetch(url, { headers });
  must(res.ok, `Failed to fetch ${url}: HTTP ${res.status} ${res.statusText}`);
  return res.json();
}

const textLines = text => text.trim().split('\n');
const jsonLines = text => textLines(text).map(line => JSON.parse(line));

// The file's own digest, worked out from its bytes, must equal the digest GitHub publishes.
async function checkAssetDigest(asset) {
  const res = await fetch(asset.browser_download_url);
  must(res.ok, `Failed to fetch ${asset.browser_download_url}: HTTP ${res.status} ${res.statusText}`);
  const hash = createHash('sha256');
  for await (const chunk of res.body) hash.update(chunk);
  const digest = `sha256:${hash.digest('hex')}`;
  must(digest === asset.digest, `${asset.name} hashes to ${digest}, but GitHub publishes ${asset.digest}`);
}

// The fork's own record names the entry it split from. Its first entries must be the
// run's, byte for byte, and its fork entry must change nothing but declare the exclusion.
function readFork(runLines, forkLines) {
  const entries = forkLines.map(line => JSON.parse(line));
  const forkEntries = entries.filter(entry => entry.kind === 'fork');
  must(forkEntries.length === 1, `Expected one fork entry in the fork's chain, found ${forkEntries.length}`);
  const entry = forkEntries[0];
  const fromEntry = entry.fork?.from_entry;
  must(Number.isInteger(fromEntry), 'The fork entry does not record from_entry');
  let sharedEntries = 0;
  while (sharedEntries < forkLines.length && forkLines[sharedEntries] === runLines[sharedEntries]) sharedEntries++;
  must(sharedEntries === fromEntry + 1, `The fork shares ${sharedEntries} entries with the run, but says it split from entry ${fromEntry}`);
  must(entry.control?.exclude, 'The fork entry does not record what it excludes');
  return {
    fromEntry,
    entryIndex: entry.index,
    entries: entries.length,
    sharedEntries,
    previous: entry.previous,
    state: entry.state,
    excluded: entry.control.exclude,
  };
}

// A segment passes only if every check on it matched. No checks means no result.
function verdict(checks) {
  if (checks.length === 0) return null;
  return checks.every(c => c.result === 'MATCH') ? 'MATCH' : 'MISMATCH';
}

// The first GeneralizedTime in an RFC 3161 reply is the stamp's genTime. This reads
// the time for display only; `ovl.py verify` is what checks the signature.
function stampTime(bytes, name) {
  const found = /\x18[\x0f-\x18](\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?:\.\d+)?Z/.exec(bytes.toString('latin1'));
  must(found, `No timestamp found in ${name}`);
  const [, y, mo, d, h, mi, s] = found;
  return `${y}-${mo}-${d}T${h}:${mi}:${s}Z`;
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
  // The fork at this revision may or may not carry its own anchor
  const forkAnchor = await fetch(`https://huggingface.co/datasets/${DATASET}/resolve/${REVISION}/${FORK}/anchor.json`, { method: 'HEAD' });
  // The training events are only written down in the project's document, at the release tag
  const docsUrl = `https://raw.githubusercontent.com/${REPOSITORY}/${RELEASE_TAG}/docs/how-it-works.md`;
  const docsRes = await fetch(docsUrl);
  must(docsRes.ok, `Failed to fetch ${docsUrl}: HTTP ${docsRes.status}`);
  const docs = await docsRes.text();
  const eventNumber = (pattern, name) => {
    const found = pattern.exec(docs)?.[1];
    must(found, `Could not read the ${name} from docs/how-it-works.md at ${RELEASE_TAG}`);
    return Number(found);
  };
  const trainingEvents = {
    hostFailureSegment: eventNumber(/Training host fails;[^|]*resumed from checkpoint (\d+)/, 'host failure'),
    codeChangeBoundary: eventNumber(/\| From boundary (\d+),/, 'code change'),
  };
  const revisionInfo = await fetchJson(`https://huggingface.co/api/datasets/${DATASET}/revision/${REVISION}`);
  must(revisionInfo.sha.startsWith(REVISION), `Revision ${REVISION} resolved to ${revisionInfo.sha}`);

  // Chain: the number of segments comes from here and nowhere else
  // Each record is kept as published, because its hash covers those exact bytes.
  const runChainLines = textLines(fetched[`${RUN}/chain.jsonl`].text);
  const chain = runChainLines.map(line => {
    const entry = JSON.parse(line);
    return {
      record: line,
      index: entry.index,
      kind: entry.kind,
      control: entry.control,
      state: entry.state,
      model: entry.model,
      previous: entry.previous,
      time: entry.time,
      entryHash: sha256(line),
    };
  });
  for (let i = 1; i < chain.length; i++) {
    must(chain[i].previous === chain[i - 1].entryHash, `Broken chain previous link at index ${i}`);
  }
  const segmentCount = chain.length - 1;

  // Project and independent results stay apart. A checker nobody has classified stops the script.
  const rawChecks = jsonLines(fetched[`${RUN}/verifications.jsonl`].text);
  const category = check => {
    must(Object.hasOwn(checkers, check.checker), `Unclassified checker: "${check.checker}". Add it to src/data/checkers.json.`);
    return checkers[check.checker];
  };
  const coverage = Array.from({ length: segmentCount }, (_, segment) => {
    const checks = rawChecks.filter(c => c.segment === segment);
    return {
      segment,
      projectResult: verdict(checks.filter(c => category(c) === 'project')),
      independentResult: verdict(checks.filter(c => category(c) === 'independent')),
      checks,
    };
  });

  const hardwareMap = jsonLines(fetched[`${RUN}/hardware-map.jsonl`].text);

  // Release hashes: the record's own list, then the asset digests GitHub reports
  const releaseHashes = {};
  for (const line of fetched[`${RUN}/release-hashes.txt`].text.trim().split('\n')) {
    const [hash, name] = line.trim().split(/\s+/);
    if (hash && name) releaseHashes[name] = hash;
  }
  const release = await fetchJson(`https://api.github.com/repos/${REPOSITORY}/releases/tags/${RELEASE_TAG}`);
  for (const name of RELEASE_ASSETS) {
    const asset = release.assets.find(a => a.name === name);
    must(asset?.digest?.startsWith('sha256:'), `Release ${RELEASE_TAG} has no SHA-256 digest for ${name}`);
    process.stdout.write(`  Checking the digest of ${name}... `);
    await checkAssetDigest(asset);
    console.log('OK');
    releaseHashes[name] = asset.digest.slice('sha256:'.length);
  }

  const anchor = JSON.parse(fetched[`${RUN}/anchor.json`].text);
  must(anchor.dataset === DATASET, `The anchor names dataset ${anchor.dataset}, not the pinned ${DATASET}`);
  must(anchor.path === RUN, `The anchor names path ${anchor.path}, not ${RUN}`);
  const timestamps = Object.entries(AUTHORITIES).map(([name, authority]) => {
    const file = `anchor.${name}.tsr`;
    return { authority, file, time: stampTime(fetched[`${RUN}/${file}`].bytes, file) };
  });

  const prepFile = fetched[PREP_PATH];
  must(prepFile.sha256 === PREPARED_ID, `preparation.json hashes to ${prepFile.sha256}, not ${PREPARED_ID}`);
  must(fetched[`${RUN}/README.md`].text.includes(PREPARED_ID), 'The release README does not name the pinned prepared-data ID');
  const prep = JSON.parse(prepFile.text);
  const millions = /(\d+)-million-parameter/.exec(fetched[`${RUN}/README.md`].text)?.[1];
  must(millions, `Could not read the model size from ${RUN}/README.md`);
  const preparation = {
    articles: prep.corpus.counts.included,
    wikipediaTargets: prep.streams.wikipedia.targets,
    conversations: prep.conversation_selection.conversations.train,
    modelParametersMillions: Number(millions),
    trainingCodeLines: fetched[`${RUN}/code/simple/ovl.py`].text.trimEnd().split('\n').length,
    preparedDataId: prepFile.sha256,
  };

  const loo = JSON.parse(fetched[`${FORK}/loo-results.json`].text);
  const removedArticle = /without the "([^"]+)" article/.exec(fetched[`${FORK}/README.md`].text)?.[1];
  must(removedArticle, `Could not read the removed article from ${FORK}/README.md`);
  const forkChecks = jsonLines(fetched[`${FORK}/verifications.jsonl`].text);
  forkChecks.forEach(category);
  const exclusion = {
    removedArticle,
    originalRemovedLoss: Number(loo.original_base.removed_loss[0].toFixed(4)),
    forkRemovedLoss: Number(loo.fork_base.removed_loss[0].toFixed(4)),
    removedLossDelta: Number((loo.fork_base.removed_loss[0] - loo.original_base.removed_loss[0]).toFixed(4)),
    originalControlLossMean: Number(loo.original_base.control_loss_mean.toFixed(4)),
    forkControlLossMean: Number(loo.fork_base.control_loss_mean.toFixed(4)),
    controlLossDelta: Number((loo.fork_base.control_loss_mean - loo.original_base.control_loss_mean).toFixed(4)),
    controlArticlesCount: loo.control_documents,
    evidencePath: `${FORK}/`,
    fork: { ...readFork(runChainLines, textLines(fetched[`${FORK}/chain.jsonl`].text)), anchored: forkAnchor.ok },
    forkChecks,
  };

  return {
    schemaVersion: 4,
    dataset: DATASET,
    revision: REVISION,
    revisionFull: revisionInfo.sha,
    sourceHashes: {
      ...Object.fromEntries(Object.entries(fetched).map(([path, f]) => [path, f.sha256])),
      [`github:${RELEASE_TAG}/docs/how-it-works.md`]: sha256(docs),
    },
    release: { tag: RELEASE_TAG, publishedAt: release.published_at },
    preparation,
    trainingEvents,
    chain,
    verifications: {
      totalChecks: rawChecks.length,
      uniqueSegments: segmentCount,
      projectChecksCount: rawChecks.filter(c => category(c) === 'project').length,
      independentChecksCount: rawChecks.filter(c => category(c) === 'independent').length,
      coverage,
      rawChecks,
    },
    hardwareMap,
    anchor: { ...anchor, timestamps },
    exclusion,
    releaseHashes,
  };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    const data = await buildEvidenceData();
    const outPath = join(__dirname, '../public/data/evidence.json');
    await mkdir(dirname(outPath), { recursive: true });
    await writeFile(outPath, JSON.stringify(data, null, 2) + '\n', 'utf8');
    console.log(`\nWrote evidence data to ${outPath}. Run npm run validate next.`);
  } catch (err) {
    console.error('Failed to fetch evidence:', err);
    process.exit(1);
  }
}
