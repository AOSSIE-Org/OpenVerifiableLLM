import { ClaimBlock } from '../components/ClaimBlock';
import { CoverageMap } from '../components/CoverageMap';
import { Section } from '../components/Section';
import { L, V } from '../components/text';
import { claim } from '../content/ledger';
import { ROUTES, hrefOf } from '../routes';

/** Overview: title, scoreboard with the independent count, three headline facts, three limits, links. */
export const OVERVIEW_FACTS = ['OV-04', 'OV-05', 'OV-06'];
export const OVERVIEW_LIMITS = ['OV-13', 'OV-12', 'LI-01'];
/** What was trained, from ledger plain lines only (OV-01 size, OV-02 Wikipedia pass, OV-03 conversation phase). */
export const TRAINED_IDS = ['OV-01', 'OV-02', 'OV-03'];

export default function Overview() {
  return (
    <>
      <div className="overview-top">
        <h1><L k="h1.overview" /></h1>
        <p className="trained" data-trained="">
          {TRAINED_IDS.map((id, i) => (
            <span key={id} data-claim-ref={id} data-claim-part="plain">{i ? ' ' : ''}{claim(id).plain}</span>
          ))}
        </p>
        <p className="indep" data-independent="">
          <L k="independent.label" /> <strong><V path="verifications.independentChecks" /></strong>
        </p>
        <Section needs={['verifications', 'chain']}>
          <CoverageMap interactive={false} strip />
        </Section>
      </div>
      <h2><L k="h2.headline" /></h2>
      <Section needs={['verifications', 'preparation', 'anchor']}>
        {OVERVIEW_FACTS.map((id) => <ClaimBlock key={id} id={id} />)}
      </Section>
      <h2><L k="h2.limits" /></h2>
      <div className="limits-list">
        {OVERVIEW_LIMITS.map((id) => <ClaimBlock key={id} id={id} hideLimit asLimit />)}
      </div>
      <h2><L k="h2.more" /></h2>
      <ul className="more-links">
        {ROUTES.filter((r) => r.key !== 'overview').map((r) => (
          <li key={r.key}>
            <a href={hrefOf(r)}>
              <L k={`nav.${r.key}`} /> <span className="muted"><L k={`more.${r.key}`} /></span>
            </a>
          </li>
        ))}
      </ul>
    </>
  );
}
