import React from 'react';
import { copy } from '../content/copy';
import { InShort } from './InShort';
import { CopyButton } from './CopyButton';

export const CheckLadder: React.FC = () => (
  <section id="check">
    <h2>{copy.check.kicker}</h2>
    <InShort>
      {copy.check.h2}
    </InShort>

    {copy.check.ladder.map(step => (
      <div key={step.time} className="level">
        <h3>
          {step.title}{' '}
          <span className="level-meta">
            ({step.time} · {step.meta})
          </span>
        </h3>
        <p className="tags">
          {step.chips.map(chip => (
            <span key={chip.label} className={`tag ${chip.type}`}>
              {chip.label}
            </span>
          ))}
        </p>

        {step.desc && <p>{step.desc}</p>}

        {step.command && (
          <div className="code">
            <pre tabIndex={0} role="region" aria-label={`${step.title} commands`}>
              {step.command}
            </pre>
            <CopyButton textToCopy={step.command} />
          </div>
        )}

        {step.action && (
          <p>
            <a href={step.action.href}>{step.action.label}</a>
          </p>
        )}
      </div>
    ))}
  </section>
);
