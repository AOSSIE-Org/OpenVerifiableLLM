import { describe, it, expect } from 'vitest';
import type { EvidenceData } from '../src/data/contracts';
import { sha256Hex } from '../src/data/sha256';
import { validateEvidence, SnapshotError } from '../src/data/validate';
import { loadEvidence, withMismatch, withoutChecks } from './evidence-cases';

// Corrupts a copy of the real snapshot. Fields are loose here because the whole point
// is to hand the validator things the contract doesn't allow.
type Loose = Record<string, any>;
const corrupt = (change: (e: Loose) => void): unknown => {
  const copy = loadEvidence() as unknown as Loose;
  change(copy);
  return copy;
};
const rejects = (change: (e: Loose) => void, message: RegExp) => {
  const bad = corrupt(change);
  expect(() => validateEvidence(bad)).toThrow(SnapshotError);
  expect(() => validateEvidence(bad)).toThrow(message);
};

describe('Evidence Validator', () => {
  it('validates official pinned evidence.json successfully', () => {
    const validated: EvidenceData = validateEvidence(loadEvidence());
    expect(validated.revision).toBe('96dc282a');
    expect(validated.chain.length).toBe(38);
    expect(validated.verifications.uniqueSegments).toBe(37);
    expect(validated.verifications.totalChecks).toBe(46);
    expect(validated.preparation.articles).toBeGreaterThan(0);
    expect(validated.preparation.wikipediaTargets).toBeGreaterThan(0);
    expect(validated.preparation.conversations).toBeGreaterThan(0);
    expect(validated.exclusion.removedArticle).toBe('3I/ATLAS');
    expect(validated.exclusion.forkRemovedLoss).toBeGreaterThan(validated.exclusion.originalRemovedLoss);
  });

  it('accepts a mismatch and a missing check, because the page shows them honestly', () => {
    expect(() => validateEvidence(withMismatch(5))).not.toThrow();
    expect(() => validateEvidence(withoutChecks(7))).not.toThrow();
  });

  it('fails on null or non-object payloads', () => {
    expect(() => validateEvidence(null)).toThrow(SnapshotError);
    expect(() => validateEvidence('not-an-object')).toThrow(SnapshotError);
    expect(() => validateEvidence([])).toThrow(SnapshotError);
  });

  describe('chain', () => {
    it('fails if chain length is less than 2 boundaries', () => {
      rejects(e => { e.chain = [e.chain[0]]; }, /Chain must have at least 2 entries/);
    });

    it('fails if a boundary is dropped, because the segment count comes from the chain', () => {
      rejects(e => { e.chain.pop(); }, /Segment count mismatch/);
    });

    it('fails if unique segment count does not match derived count from chain.jsonl', () => {
      rejects(e => { e.verifications.uniqueSegments = 36; }, /Segment count mismatch/);
    });

    it('fails if previous link in chain is broken', () => {
      rejects(e => { e.chain[10].previous = '0'.repeat(64); }, /Broken previous link at boundary 10/);
    });

    it('fails on a missing or unusable time, which would otherwise break rendering', () => {
      rejects(e => { delete e.chain[3].time; }, /Invalid time at boundary 3/);
      rejects(e => { e.chain[3].time = 'yesterday'; }, /Invalid time at boundary 3/);
      rejects(e => { e.chain[3].time = e.chain[2].time - 100; }, /Time goes backwards at boundary 3/);
    });

    it('fails on missing control data or an unknown kind', () => {
      rejects(e => { delete e.chain[4].control; }, /control at boundary 4/);
      rejects(e => { delete e.chain[4].control.global_step; }, /Invalid global_step at boundary 4/);
      rejects(e => { e.chain[4].control.global_step = 1; }, /Training step goes backwards at boundary 4/);
      rejects(e => { delete e.chain[4].control.phase; }, /Invalid phase at boundary 4/);
      rejects(e => { e.chain[4].kind = 'made-up'; }, /Invalid kind at boundary 4/);
    });
  });

  describe('chain records and hashes', () => {
    // The field changes, and the stored hash stays as it was
    it('catches a changed field whose stored hash is unchanged', () => {
      rejects(e => { e.chain[1].time += 1; }, /time at boundary 1 does not match its record/);
      rejects(e => { e.chain[1].state = 'a'.repeat(64); }, /state at boundary 1 does not match its record/);
      rejects(e => { e.chain[1].model = 'a'.repeat(64); }, /model at boundary 1 does not match its record/);
      rejects(e => { e.chain[1].control.cursor += 1; }, /control at boundary 1 does not match its record/);
      rejects(e => { e.chain[1].control.phase = 'other'; }, /control at boundary 1 does not match its record/);
      rejects(e => { e.chain[1].kind = 'epoch'; }, /kind at boundary 1 does not match its record/);
    });

    it('catches a changed record whose stored hash is unchanged', () => {
      rejects(e => { e.chain[2].record = e.chain[2].record.replace('"time": ', '"time":  '); }, /Entry hash at boundary 2 does not match its record/);
    });

    it('catches a changed stored hash', () => {
      rejects(e => { e.chain[5].entryHash = 'a'.repeat(64); }, /Broken previous link at boundary 6|Entry hash at boundary 5/);
      rejects(e => { e.chain[37].entryHash = 'a'.repeat(64); }, /Entry hash at boundary 37/);
    });

    it('catches a missing or broken record', () => {
      rejects(e => { delete e.chain[3].record; }, /Missing record at boundary 3/);
      rejects(e => { e.chain[3].record = 'not json'; e.chain[3].entryHash = sha256Hex('not json'); e.chain[4].previous = e.chain[3].entryHash; }, /Record at boundary 3 is not valid JSON/);
    });

    it('catches a record rewritten with its hash, because the link to the next entry breaks', () => {
      rejects(e => {
        const entry = JSON.parse(e.chain[4].record);
        entry.time += 1;
        const record = JSON.stringify(entry);
        e.chain[4].record = record;
        e.chain[4].entryHash = sha256Hex(record);
        e.chain[4].time = entry.time;
      }, /Broken previous link at boundary 5/);
    });

    it('catches a whole chain rewritten without its anchor', () => {
      rejects(e => {
        const entry = JSON.parse(e.chain[37].record);
        entry.state = 'a'.repeat(64);
        const record = JSON.stringify(entry);
        e.chain[37].record = record;
        e.chain[37].state = entry.state;
        e.chain[37].entryHash = sha256Hex(record);
      }, /Anchor/);
    });

    it('catches records that no longer rebuild the file the anchor names', () => {
      rejects(e => {
        const hash = 'a'.repeat(64);
        e.anchor.chain_jsonl_sha256 = hash;
        e.sourceHashes['release/simple-run/chain.jsonl'] = hash;
        e.releaseHashes['chain.jsonl'] = hash;
      }, /do not hash to the chain\.jsonl the anchor names/);
    });

    it('still passes a rewrite of everything, which is why this is not verification', () => {
      // The honest limit of this validator, kept as a test so the README can't drift from it.
      const copy = loadEvidence() as unknown as Loose;
      copy.chain = copy.chain.map((entry: Loose) => {
        const { record, ...rest } = entry;
        return { record, ...rest };
      });
      const last = copy.chain[37];
      const parsed = JSON.parse(last.record);
      parsed.time += 1;
      last.record = JSON.stringify(parsed);
      last.time = parsed.time;
      last.entryHash = sha256Hex(last.record);
      const file = `${copy.chain.map((e: Loose) => e.record).join('\n')}\n`;
      copy.anchor.chain_head_sha256 = last.entryHash;
      copy.anchor.chain_jsonl_sha256 = sha256Hex(file);
      copy.sourceHashes['release/simple-run/chain.jsonl'] = copy.anchor.chain_jsonl_sha256;
      copy.releaseHashes['chain.jsonl'] = copy.anchor.chain_jsonl_sha256;
      expect(() => validateEvidence(copy)).not.toThrow();
    });
  });

  describe('values that reach links and commands', () => {
    const badDatasets = [
      'owner/name; rm -rf /',
      'owner/name`whoami`',
      'owner/$(whoami)',
      'owner/name\nrm -rf /',
      'owner/name\n',
      ' owner/name',
      'owner/name ',
      'owner name/data',
      'owner',
      'owner/',
      '/name',
      'a/b/c',
      '../name',
      'owner/..',
      'owner/na..me',
      'owner/-name',
      'owner/name&&ls',
      'owner/name|ls',
      "owner/name'",
      'owner/name"',
      'https://evil.example/x',
      '',
    ];
    for (const dataset of badDatasets) {
      it(`rejects the dataset ${JSON.stringify(dataset)}`, () => {
        rejects(e => { e.dataset = dataset; }, /Invalid dataset identifier/);
      });
    }

    const badTags = [
      'v1.1.0; rm -rf /',
      'v1.1.0`whoami`',
      'v$(whoami)',
      'v1.1.0\nrm -rf /',
      'v1.1.0\n',
      ' v1.1.0',
      'v1.1',
      '1.1.0',
      'v1.1.0-rc1',
      'latest',
      'main',
      '--upload-pack=x',
      '',
    ];
    for (const tag of badTags) {
      it(`rejects the release tag ${JSON.stringify(tag)}`, () => {
        rejects(e => { e.release.tag = tag; }, /Invalid release tag/);
      });
    }

    it('rejects a dataset or tag that is not a string', () => {
      rejects(e => { e.dataset = 42; }, /Invalid dataset identifier/);
      rejects(e => { e.release.tag = { toString: 'v1.1.0' }; }, /Invalid release tag/);
    });

    it('rejects a revision with shell characters', () => {
      rejects(e => { e.revision = '96dc282a; ls'; }, /Invalid evidence revision format/);
      rejects(e => { e.revisionFull = `${e.revisionFull}\n`; }, /is not the full form of revision/);
    });

    it('checks that the anchor names the same dataset', () => {
      rejects(e => { e.dataset = 'someone/else'; }, /Anchor names dataset/);
    });
  });

  describe('fork point', () => {
    it('reads the split from the fork record', () => {
      const { fork } = loadEvidence().exclusion;
      expect(fork).toMatchObject({ fromEntry: 30, entryIndex: 31, entries: 39, sharedEntries: 31 });
    });

    it('rejects a fork point that does not fit the evidence', () => {
      rejects(e => { delete e.exclusion.fork; }, /Missing exclusion fork section/);
      rejects(e => { e.exclusion.fork.fromEntry = 29; }, /Fork entry 31 does not follow from entry 29/);
      rejects(e => { e.exclusion.fork.fromEntry = 999; }, /Invalid fork point: from entry 999/);
      rejects(e => { e.exclusion.fork.fromEntry = -1; }, /Invalid fork point/);
      rejects(e => { e.exclusion.fork.entryIndex = 32; }, /Fork entry 32 does not follow from entry 30/);
      rejects(e => { e.exclusion.fork.sharedEntries = 5; }, /Fork shares 5 entries but splits from entry 30/);
      rejects(e => { e.exclusion.fork.entries = 31; }, /too few for fork entry 31/);
    });

    it('rejects a fork that does not link to the run, or changes its state', () => {
      rejects(e => { e.exclusion.fork.previous = 'a'.repeat(64); }, /Fork does not link to entry 30/);
      rejects(e => { e.exclusion.fork.state = 'a'.repeat(64); }, /Fork entry changes the state of entry 30/);
    });

    it('rejects a fork check outside the segments after the split', () => {
      rejects(e => { e.exclusion.forkChecks[0].segment = 29; }, /Fork check #0 points at invalid segment 29; must be in \[30, 37\]/);
      rejects(e => { e.exclusion.forkChecks[0].segment = 38; }, /Fork check #0 points at invalid segment 38/);
    });
  });

  describe('checks and coverage', () => {
    it('fails if any row in verifications points at an invalid segment', () => {
      rejects(e => { e.verifications.rawChecks[0].segment = 999; }, /points at invalid segment 999/);
    });

    it('fails if checker string is unknown / unauthorized', () => {
      rejects(
        e => { e.verifications.rawChecks.push({ ...e.verifications.rawChecks[0], checker: 'rogue anonymous checker' }); },
        /Unknown or unauthorized checker string/,
      );
    });

    it('does not accept a checker name that merely looks like the project', () => {
      rejects(e => { e.verifications.rawChecks[0].checker = 'project impostor'; }, /Unknown or unauthorized checker string/);
    });

    it('fails if the coverage summary hides a mismatch in the raw checks', () => {
      rejects(e => { e.verifications.rawChecks[5].result = 'MISMATCH'; }, /Coverage disagrees with raw checks: segment 5/);
    });

    it('fails if the coverage summary turns a missing check into a pass', () => {
      rejects(
        e => { e.verifications.rawChecks = e.verifications.rawChecks.filter((c: Loose) => c.segment !== 7); },
        /Coverage disagrees with raw checks: segment 7 says project MATCH, the checks say null/,
      );
    });

    it('fails if the coverage summary claims an independent check nobody made', () => {
      rejects(e => { e.verifications.coverage[2].independentResult = 'MATCH'; }, /segment 2 says independent MATCH/);
    });

    it('fails if a result is anything but MATCH or MISMATCH', () => {
      rejects(e => { e.verifications.rawChecks[0].result = 'PASS'; }, /Invalid verification result/);
    });

    it('fails if the published counts differ from the raw rows', () => {
      rejects(e => { e.verifications.totalChecks = 99; }, /totalChecks \(99\)/);
      rejects(e => { e.verifications.independentChecksCount = 5; }, /independentChecksCount \(5\)/);
      rejects(e => { e.verifications.projectChecksCount = 5; }, /projectChecksCount \(5\)/);
    });
  });

  describe('hashes and anchor', () => {
    it('fails if a source SHA-256 hash is corrupted or invalid', () => {
      rejects(e => { e.sourceHashes['release/simple-run/chain.jsonl'] = 'corrupted_hash_not_64_hex'; }, /Invalid or corrupted SHA-256 hash/);
    });

    it('fails if the fetched chain is not the one release-hashes.txt lists', () => {
      rejects(e => { e.releaseHashes['chain.jsonl'] = 'a'.repeat(64); }, /chain\.jsonl does not match its hash in release-hashes\.txt/);
    });

    it('fails if a checkpoint is missing from the hash list', () => {
      rejects(e => { delete e.releaseHashes['ckpt-00012.pt']; }, /lists 37 checkpoints for 38 boundaries/);
    });

    it('fails if a file the page offers has no hash', () => {
      rejects(e => { delete e.releaseHashes['ovl-23m-chat.gguf']; }, /no hash for ovl-23m-chat\.gguf/);
    });

    it('fails if the anchor does not match the chain it stamped', () => {
      rejects(e => { e.anchor.chain_head_sha256 = 'a'.repeat(64); }, /Anchor chain head/);
      rejects(e => { e.anchor.chain_entries = 12; }, /Anchor records 12 chain entries/);
      rejects(e => { e.anchor.final_model = 'a'.repeat(64); }, /Anchor final model/);
      rejects(e => { e.anchor.final_state = 'a'.repeat(64); }, /Anchor final state/);
      rejects(e => { e.anchor.chain_jsonl_sha256 = 'a'.repeat(64); }, /Anchor does not match the hash of chain\.jsonl/);
      rejects(e => { e.anchor.release_hashes_sha256 = 'a'.repeat(64); }, /release-hashes\.txt/);
      rejects(e => { e.anchor.dataset = 'someone/else'; }, /Anchor names dataset/);
    });

    it('fails without a usable timestamp', () => {
      rejects(e => { e.anchor.timestamps = []; }, /at least one timestamp/);
      rejects(e => { e.anchor.timestamps[0].time = 'soon'; }, /Anchor timestamp 0 has an invalid time/);
      rejects(e => { delete e.anchor.timestamps[1].authority; }, /Anchor timestamp 1 has no authority/);
    });

    it('fails if the prepared data id is not the hash of its file', () => {
      rejects(e => { e.preparation.preparedDataId = 'a'.repeat(64); }, /preparedDataId is not the hash/);
    });
  });

  describe('other displayed fields', () => {
    it('fails on a broken revision or release', () => {
      rejects(e => { e.revision = 'main'; }, /Invalid evidence revision format/);
      rejects(e => { e.revisionFull = 'a'.repeat(40); }, /is not the full form of revision/);
      rejects(e => { delete e.release; }, /Missing release section/);
      rejects(e => { e.release.publishedAt = 'someday'; }, /Invalid release date/);
    });

    it('fails on a broken hardware map', () => {
      rejects(e => { e.hardwareMap = []; }, /hardwareMap must be a non-empty array/);
      rejects(e => { e.hardwareMap[0].result = 'OK'; }, /Invalid result for hardware probe 0/);
      rejects(e => { e.hardwareMap[0].segment = 500; }, /Hardware probe 0 points at invalid segment 500/);
      rejects(e => { e.hardwareMap[4].segment = 'all 12'; }, /Hardware probe 4 points at invalid segment all 12/);
    });

    it('fails on missing preparation counts', () => {
      rejects(e => { delete e.preparation; }, /Missing preparation section/);
      rejects(e => { e.preparation.articles = 0; }, /articles count must be a positive integer/);
      rejects(e => { e.preparation.trainingCodeLines = '~630 lines'; }, /trainingCodeLines must be a positive integer/);
    });

    it('fails if verifiable exclusion loss delta is mathematically inconsistent', () => {
      rejects(e => { e.exclusion.removedLossDelta = 9.9999; }, /Exclusion removed loss delta inconsistent/);
      rejects(e => { e.exclusion.controlLossDelta = 9.9999; }, /Exclusion control loss delta inconsistent/);
    });

    it('fails if the fork has no checks, or an unknown checker', () => {
      rejects(e => { e.exclusion.forkChecks = []; }, /Exclusion has no fork checks/);
      rejects(e => { e.exclusion.forkChecks[0].checker = 'rogue'; }, /Unknown or unauthorized checker string/);
    });
  });
});
