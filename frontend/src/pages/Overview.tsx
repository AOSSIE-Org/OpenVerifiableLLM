import { ClaimBlock } from '../components/ClaimBlock';
import { CoverageMap } from '../components/CoverageMap';
import { OddsBar } from '../components/OddsBar';
import { Section } from '../components/Section';
import { L, V } from '../components/text';
import { claim } from '../content/ledger';
import { ROUTES, hrefOf } from '../routes';

/** Overview: headline, scoreboard with the independent count, one plain line per claim, links.
 *  OV-02/OV-03 live on Timeline, OV-05 on Downloads, OV-06 on Verification; OV-12 is covered by LI-02. */
export const OVERVIEW_FACTS = ['OV-04'];
export const OVERVIEW_LIMITS = ['OV-13', 'LI-01'];
/** What was trained, from the ledger plain line only (OV-01 size). */
export const TRAINED_IDS = ['OV-01'];

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
        <div className="scoreboard">
          <Section needs={['verifications', 'chain']}>
            <p className="score">
              <strong>
                <V path="verifications.segmentsMatched" />
                {' / '}
                <V path="chain.segments" />
              </strong>
              {' '}
              <L k="score.matched" />
            </p>
          </Section>
          <p className="indep" data-independent="">
            <L k="independent.label" /> <strong><V path="verifications.independentChecks" /></strong>
          </p>
        </div>
        <Section needs={['verifications', 'chain']}>
          <CoverageMap interactive={false} strip />
        </Section>
      </div>
      {/* The detection-odds figure from Verification, full width: it shows what 0 independent checks means. */}
      <section className="overview-odds">
        <h2><L k="h2.odds" /></h2>
        <Section needs={['odds']}>
          <OddsBar who compact atN={false} />
        </Section>
      </section>
      <div className="split">
        <div>
          <h2><L k="h2.headline" /></h2>
          <Section needs={['verifications']}>
            <div className="facts">
              {OVERVIEW_FACTS.map((id) => <ClaimBlock key={id} id={id} />)}
            </div>
          </Section>
        </div>
        <div>
          <h2><L k="h2.limits" /></h2>
          <div className="limits-list">
            {OVERVIEW_LIMITS.map((id) => <ClaimBlock key={id} id={id} hideLimit asLimit />)}
          </div>
        </div>
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
