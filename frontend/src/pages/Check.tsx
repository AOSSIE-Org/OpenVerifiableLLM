import { ClaimBlock } from '../components/ClaimBlock';
import { Section } from '../components/Section';
import { L } from '../components/text';

export default function Check() {
  return (
    <>
      <h1><L k="h1.check" /></h1>
      <p className="lead"><L k="check.lead" /></p>
      <div className="ladder">
        <section className="step">
          <p className="step-time"><L k="h2.try" /></p>
          <div className="step-body">
            <h2><L k="step.try" /></h2>
            <p className="step-meta"><L k="step.meta.try" /></p>
            <ClaimBlock id="CK-01" />
          </div>
        </section>
        <section className="step">
          <p className="step-time"><L k="h2.consistency" /></p>
          <div className="step-body">
            <h2><L k="step.consistency" /></h2>
            <p className="step-meta"><L k="step.meta.consistency" /></p>
            <Section needs={['check']}>
              <ClaimBlock id="CK-02" />
              <ClaimBlock id="CK-03" />
              <ClaimBlock id="CK-04" />
            </Section>
          </div>
        </section>
        <section className="step step-key">
          <p className="step-time"><L k="h2.actual" /></p>
          <div className="step-body">
            <h2><L k="step.actual" /></h2>
            <p className="step-meta"><L k="step.meta.actual" /></p>
            <Section needs={['check']}>
              <ClaimBlock id="CK-05" />
              <ClaimBlock id="CK-06" />
              <ClaimBlock id="CK-07" />
            </Section>
          </div>
        </section>
        <section className="step">
          <p className="step-time"><L k="h2.timestamps" /></p>
          <div className="step-body">
            <h2><L k="step.timestamps" /></h2>
            <ClaimBlock id="CK-08" />
          </div>
        </section>
      </div>
    </>
  );
}
