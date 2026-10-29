import { ClaimBlock } from '../components/ClaimBlock';
import { Section } from '../components/Section';
import { L } from '../components/text';

export default function Check() {
  return (
    <>
      <h1><L k="h1.check" /></h1>
      <h2><L k="h2.try" /></h2>
      <ClaimBlock id="CK-01" />
      <h2><L k="h2.consistency" /></h2>
      <Section needs={['check']}>
        <ClaimBlock id="CK-02" />
        <ClaimBlock id="CK-03" />
        <ClaimBlock id="CK-04" />
      </Section>
      <h2><L k="h2.actual" /></h2>
      <Section needs={['check']}>
        <ClaimBlock id="CK-05" />
        <ClaimBlock id="CK-06" />
        <ClaimBlock id="CK-07" />
      </Section>
      <h2><L k="h2.timestamps" /></h2>
      <ClaimBlock id="CK-08" />
    </>
  );
}
