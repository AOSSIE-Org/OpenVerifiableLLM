import { ClaimBlock } from '../components/ClaimBlock';
import { Section } from '../components/Section';
import { Timeline as Strip } from '../components/Timeline';
import { L } from '../components/text';

export default function Timeline() {
  return (
    <>
      <h1><L k="h1.timeline" /></h1>
      <h2><L k="h2.record" /></h2>
      <Section needs={['chain', 'events']}>
        <ClaimBlock id="TL-01">
          <Strip />
        </ClaimBlock>
      </Section>
      <ClaimBlock id="TL-06" />
      <ClaimBlock id="TL-02" />
      <ClaimBlock id="TL-03" />
      <h2><L k="h2.events" /></h2>
      <Section needs={['events']}>
        <ClaimBlock id="TL-04" />
        <ClaimBlock id="TL-05" />
      </Section>
      <ClaimBlock id="TL-07" />
      <h2><L k="h2.trained" /></h2>
      <Section needs={['preparation', 'model']}>
        {['OV-01', 'OV-02', 'OV-03', 'OV-09', 'OV-10', 'OV-11'].map((id) => <ClaimBlock key={id} id={id} />)}
      </Section>
    </>
  );
}
