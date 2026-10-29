import { ClaimBlock } from '../components/ClaimBlock';
import { CoverageMap } from '../components/CoverageMap';
import { HardwareMap } from '../components/HardwareMap';
import { OddsBar } from '../components/OddsBar';
import { RowGroups } from '../components/RowGroups';
import { Section } from '../components/Section';
import { L } from '../components/text';
import { claimsOn, claim } from '../content/ledger';
import { OVERVIEW_LIMITS } from './Overview';

/** Limit claims for this page, each distinct limit once: a ledger limit whose developer line is
 *  "see X" is a duplicate of claim X and is dropped when X is already on this page. */
export const VERIFICATION_IDS = ['VE-01', 'VE-02', 'VE-03', 'VE-04', 'VE-06', 'VE-07', 'VE-08', 'VE-09', 'VE-10', 'VE-11', 'VE-12', 'OV-07', 'OV-08'];
export const duplicateOf = (id: string) => /^see ([A-Z]+-\d+)$/.exec(claim(id).developer.trim())?.[1] ?? null;
export function distinctLimits(): string[] {
  return claimsOn('limits')
    .filter((c) => !OVERVIEW_LIMITS.includes(c.id))
    .filter((c) => {
      const d = duplicateOf(c.id);
      return !(d && VERIFICATION_IDS.includes(d));
    })
    .map((c) => c.id);
}

export default function Verification() {
  return (
    <>
      <h1><L k="h1.verification" /></h1>
      <h2><L k="h2.map" /></h2>
      <Section needs={['verifications', 'events']}>
        <ClaimBlock id="VE-01">
          <CoverageMap interactive />
        </ClaimBlock>
      </Section>
      <ClaimBlock id="VE-02" />
      <h2><L k="h2.rows" /></h2>
      <Section needs={['verifications']}>
        <ClaimBlock id="VE-03">
          <RowGroups base="verifications" />
        </ClaimBlock>
        <ClaimBlock id="VE-04" />
      </Section>
      <h2><L k="h2.odds" /></h2>
      <Section needs={['odds']}>
        <ClaimBlock id="VE-06">
          <OddsBar />
        </ClaimBlock>
        <ClaimBlock id="VE-07" />
        <ClaimBlock id="VE-08" />
        <ClaimBlock id="VE-09" />
      </Section>
      <h2><L k="h2.hardware" /></h2>
      <Section needs={['hardware']}>
        <HardwareMap />
        {claimsOn('verification', 'VE-HW').map((c) => <ClaimBlock key={c.id} id={c.id} />)}
      </Section>
      <h2><L k="h2.trust" /></h2>
      <ClaimBlock id="VE-10" />
      <ClaimBlock id="VE-11" />
      <ClaimBlock id="VE-12" />
      <h2><L k="h2.meaning" /></h2>
      <ClaimBlock id="OV-08" />
      <ClaimBlock id="OV-07" />
      <h2><L k="h2.limits" /></h2>
      <div className="limits-list">
        {distinctLimits().map((id) => <ClaimBlock key={id} id={id} hideLimit asLimit />)}
      </div>
    </>
  );
}
