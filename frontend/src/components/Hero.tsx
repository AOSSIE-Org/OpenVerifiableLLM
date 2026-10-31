import React from 'react';
import { copy } from '../content/copy';

export const Hero: React.FC = () => {
  return (
    <div className="wrap hero" id="top">
      <div>
        <div className="kicker">{copy.hero.kicker}</div>
        <h1>
          {copy.hero.h1Prefix}
          <span>{copy.hero.h1Accent}</span>
          {copy.hero.h1Suffix}
        </h1>
        <p className="lede">{copy.hero.lede}</p>
        <div className="cta">
          <a className="btn primary" href="#check">
            {copy.hero.verifyCta}
          </a>
          <a className="btn" href="#overview">
            {copy.hero.evidenceCta}
          </a>
        </div>
      </div>

      <figure className="term" aria-label="Output of the consistency check">
        <div className="tt">
          <b>{copy.hero.termTitle}</b>
          <span>{copy.hero.termMeta}</span>
        </div>
        <pre tabIndex={0} role="region" aria-label="Consistency check terminal commands">
          <span className="p">$</span> openssl ts -verify -in anchor.digicert.tsr -data anchor.json \
          {'\n'}    -CAfile /etc/ssl/certs/ca-certificates.crt{'\n'}
          <span className="ok">Verification: OK</span>
          {'\n\n'}
          <span className="p">$</span> OVL_DEVICE=cpu python ovl-code/simple/ovl.py verify \
          {'\n'}    --run . --weights conversation-model.pt{'\n'}
          anchor anchor.json: chain head <span className="ok">MATCH</span> (38 entries){'\n'}
          weights conversation-model.pt: <span className="ok">MATCH</span> chain entry 37
        </pre>
        <figcaption>{copy.hero.termCaption}</figcaption>
      </figure>
    </div>
  );
};
