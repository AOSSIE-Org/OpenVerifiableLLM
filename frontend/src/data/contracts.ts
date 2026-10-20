import { z } from 'zod';

export const results = ['PASS', 'FAIL', 'NOT_RUN', 'UNAVAILABLE', 'UNSUPPORTED'] as const;
export const scopes = ['fixture', 'pilot', 'production'] as const;
export const profiles = ['artifact-identity', 'data-reconstruction', 'sampled-replay', 'complete-replay', 'inference-reproduction'] as const;
const text = z.string().trim().min(1).max(4096);
const id = z.string().regex(/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/);
export const sha256 = z.string().regex(/^[a-f0-9]{64}$/);
const date = z.iso.datetime({ offset: true });
export const safeUrl = z.string().max(2048).refine(value => {
  try { const u = new URL(value); return u.protocol === 'https:' && !u.username && !u.password; } catch { return false; }
}, 'Expected an absolute HTTPS URL without credentials');
const path = z.string().max(512).refine(p => p.length > 0 && !p.startsWith('/') && !p.includes('\\') && !p.split('/').some(s => s === '..' || s === '.' || s === ''), 'Expected a confined relative artifact path');
const revision = z.string().regex(/^[a-f0-9]{40,64}$/);
export const SourceSchema = z.object({
  url: safeUrl.nullable(), revision: text.nullable(), path: path.nullable(), sha256: sha256.nullable(),
}).strict();
export const EvidenceSchema = z.object({
  id, title: text, kind: text, phase: text, scope: z.enum(scopes), synthetic: z.boolean(),
  source: SourceSchema, parents: z.array(id).max(100),
  sizeBytes: z.number().int().nonnegative().safe().nullable(), timestamp: date.nullable(),
  supersedes: id.nullable(), supersededBy: id.nullable(),
  metadata: z.record(z.string().max(128), z.union([z.string().max(2048), z.number().finite(), z.boolean(), z.null()])).refine(v => Object.keys(v).length <= 40),
}).strict();
export const CheckSchema = z.object({
  id, title: text, profile: z.enum(profiles), phase: text, scope: z.enum(scopes), result: z.enum(results),
  explanation: text, performedBy: text.nullable(), attestedBy: text.nullable(), locallyRecomputed: z.boolean(),
  attribution: z.enum(['publisher-report', 'project-operated-replay', 'consumer-local-recomputation', 'independent-third-party', 'unknown']),
  independenceEvidence: SourceSchema.nullable(), evidenceIds: z.array(id).max(100), timestamp: date.nullable(),
}).strict().superRefine((c, ctx) => {
  if (c.result === 'PASS' && c.evidenceIds.length === 0) ctx.addIssue({ code: 'custom', message: 'A reported pass requires evidence references' });
  if (c.attribution === 'independent-third-party' && !c.independenceEvidence?.url) ctx.addIssue({ code: 'custom', message: 'Independent attribution requires explicit supporting evidence' });
});
const FileSchema = z.object({ path, sizeBytes: z.number().int().nonnegative().safe(), sha256 }).strict();
const IdentitySchema = z.object({
  repository: z.string().regex(/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/), revision, root: sha256,
  files: z.array(FileSchema).min(1).max(100), inventoryUrl: safeUrl, licenseUrl: safeUrl,
  runtime: text, modelCardUrl: safeUrl,
}).strict();
export const ReleaseSchema = z.object({
  id, role: z.enum(['base', 'chat']), availability: z.enum(['not-released', 'available', 'withdrawn']),
  identity: IdentitySchema.nullable(), parentReleaseId: id.nullable(), checkIds: z.array(id).max(100),
}).strict().superRefine((r, ctx) => {
  if (r.availability === 'available' && !r.identity) ctx.addIssue({ code: 'custom', message: 'An available release requires exact identity' });
  if (r.availability === 'available' && r.role === 'chat' && !r.parentReleaseId) ctx.addIssue({ code: 'custom', message: 'A chat release requires a base parent' });
});
export const CommandSchema = z.object({
  id, profile: z.enum(profiles), approved: z.literal(true), executable: text,
  prerequisites: z.array(text).min(1).max(20), expectedScope: text, source: SourceSchema,
  validationEvidenceIds: z.array(id).min(1).max(20),
  resources: z.object({ description: text, kind: z.enum(['measured', 'estimate']), measurementSource: SourceSchema }).strict(),
}).strict();
export const SnapshotSchema = z.object({
  schemaVersion: z.literal(1), generatedAt: date, mode: z.enum(['fixture', 'public-snapshot']),
  approval: z.object({ approvedBy: text, approvedAt: date, revision }).strict().nullable(),
  sources: z.array(SourceSchema).min(1).max(30),
  workflow: z.object({ summary: text, state: z.enum(['pending', 'running', 'complete']) }).strict(),
  releases: z.array(ReleaseSchema).length(2), checks: z.array(CheckSchema).max(2000),
  evidence: z.array(EvidenceSchema).max(2000), commands: z.array(CommandSchema).max(30),
}).strict().superRefine((s, ctx) => {
  for (const key of ['releases', 'checks', 'evidence', 'commands'] as const) {
    const keys = s[key].map(x => x.id);
    if (new Set(keys).size !== keys.length) ctx.addIssue({ code: 'custom', message: 'Duplicate IDs in ' + key });
  }
  if (new Set(s.releases.map(r => r.role)).size !== 2) ctx.addIssue({ code: 'custom', message: 'Expected one base and one chat release' });
  const base = s.releases.find(r => r.role === 'base');
  for (const r of s.releases) {
    if (r.role === 'chat' && r.availability === 'available' && r.parentReleaseId !== base?.id)
      ctx.addIssue({ code: 'custom', message: 'Chat release parent must reference the base release' });
  }
  if (s.mode === 'fixture' && s.evidence.some(e => !e.synthetic)) ctx.addIssue({ code: 'custom', message: 'Fixture evidence must be labelled synthetic' });
  if (s.mode === 'public-snapshot' && s.evidence.some(e => e.synthetic))
    ctx.addIssue({ code: 'custom', message: 'Public snapshots cannot contain synthetic evidence' });
  if (s.mode === 'public-snapshot' && (!s.approval || !s.sources.some(src => src.url && src.revision === s.approval?.revision)))
    ctx.addIssue({ code: 'custom', message: 'Public snapshot requires approval and matching pinned source revision' });
  for (const c of s.commands) {
    if (!c.source.url || !c.source.revision || !c.resources.measurementSource.url || c.validationEvidenceIds.some(id => !s.evidence.some(e => e.id === id)))
      ctx.addIssue({ code: 'custom', message: 'Approved command requires pinned source, sourced resources and present validation evidence' });
  }
});
export type Snapshot = z.infer<typeof SnapshotSchema>;
export type Evidence = z.infer<typeof EvidenceSchema>;
export type Check = z.infer<typeof CheckSchema>;
export type Release = z.infer<typeof ReleaseSchema>;
export type Source = z.infer<typeof SourceSchema>;
export type Profile = typeof profiles[number];
export type Result = typeof results[number];

export function validateSnapshot(input: unknown, expectedMode?: Snapshot['mode']): Snapshot {
  const s = SnapshotSchema.parse(input);
  if (expectedMode && s.mode !== expectedMode) throw new Error('Snapshot mode does not match the configured data mode.');
  return s;
}
export function immutableFileUrl(release: Release, filePath: string): string | null {
  if (release.availability !== 'available' || !release.identity || !release.identity.files.some(f => f.path === filePath)) return null;
  const r = release.identity;
  return 'https://huggingface.co/' + r.repository + '/resolve/' + r.revision + '/' + filePath.split('/').map(encodeURIComponent).join('/');
}
export function evidenceChecks(s: Snapshot, id: string) { return s.checks.filter(c => c.evidenceIds.includes(id)); }
export function missingCheckReferences(s: Snapshot, c: Check) { return c.evidenceIds.filter(id => !s.evidence.some(e => e.id === id)); }
