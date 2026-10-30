import React from 'react';
import { copy } from '../content/copy';
import { CopyButton } from './CopyButton';

export const CheckLadder: React.FC = () => {
  return (
    <section id="check">
      <div className="wrap">
        <div className="sh">
          <div className="kicker">{copy.check.kicker}</div>
          <h2>{copy.check.h2}</h2>
        </div>

        <div className="ladder">
          {copy.check.ladder.map((step, idx) => (
            <div key={idx} className={`step ${step.isKey ? 'key' : ''}`}>
              <div>
                <div className="time">{step.time}</div>
                <div className="meta">{step.meta}</div>
              </div>

              <div>
                <h3>{step.title}</h3>
                <div className="chips">
                  {step.chips.map((chip, cIdx) => (
                    <span key={cIdx} className={`chip ${chip.type}`}>
                      {chip.label}
                    </span>
                  ))}
                </div>

                {step.desc && <p>{step.desc}</p>}

                {step.command && (
                  <figure className="term">
                    <pre tabIndex={0} role="region" aria-label="Terminal command">{step.command}</pre>
                    <CopyButton textToCopy={step.command} />
                  </figure>
                )}

                {step.action && (
                  <a className="btn primary" href={step.action.href}>
                    {step.action.label}
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
