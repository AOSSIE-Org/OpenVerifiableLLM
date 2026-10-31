import { readFileSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import type { SiteData } from '../../src/data/types';

export const sha = (s: string) => createHash('sha256').update(s).digest('hex');
const RAW = readFileSync(path.resolve(__dirname, '../../public/data/site-data.json'), 'utf8');
/** A fresh deep copy of the real, committed site-data.json for each test. */
export const fresh = (): SiteData => JSON.parse(RAW) as SiteData;
