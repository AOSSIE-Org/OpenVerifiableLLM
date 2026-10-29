import { ClaimBlock } from '../components/ClaimBlock';
import { LossTable } from '../components/LossTable';
import { RowGroups } from '../components/RowGroups';
import { Section } from '../components/Section';
import { L } from '../components/text';

export default function Exclusion() {
  return (
    <>
      <h1><L k="h1.exclusion" /></h1>
      <h2><L k="h2.removed" /></h2>
      <ClaimBlock id="EX-01" />
      <h2><L k="h2.why" /></h2>
      <ClaimBlock id="EX-02" />
      <ClaimBlock id="EX-03" />
      <ClaimBlock id="EX-04" />
      <Section needs={['exclusion']}>
        <ClaimBlock id="EX-05">
          <RowGroups base="exclusion" />
        </ClaimBlock>
      </Section>
      <h2><L k="h2.loss" /></h2>
      <Section needs={['exclusion']}>
        <ClaimBlock id="EX-06">
          <LossTable />
        </ClaimBlock>
        <ClaimBlock id="EX-07" />
      </Section>
      <h2><L k="h2.limits" /></h2>
      <ClaimBlock id="EX-08" />
      <ClaimBlock id="EX-09" />
      <ClaimBlock id="EX-10" />
    </>
  );
}
