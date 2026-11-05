import React from 'react';
import { copy } from '../content/copy';
import { CopyButton } from './CopyButton';

// Three ways to check, on one scale of effort from a few minutes to an hour.
// A one-line command shows in place; a longer script folds away.
export const CheckLadder: React.FC = () => (
  <section id="check" className="sec">
    <div className="sec-head">
      <h2>{copy.check.kicker}</h2>
      <p className="sub">{copy.check.intro}</p>
    </div>

    <ol className="steps">
      {copy.check.ladder.map(step => (
        <li key={step.time} className={step.isKey ? 'step key' : 'step'}>
          <p className="step-time">{step.time}</p>
          <p className="step-meta">{step.meta}</p>
          <h3>{step.title}</h3>
          <p>{step.desc}</p>
          <ul className="step-proves">
            {step.chips.map(chip => (
              <li key={chip.label} className={chip.type}>
                {chip.label}
              </li>
            ))}
          </ul>

          {step.command && !step.command.includes('\n') && (
            <div className="code">
              <pre tabIndex={0} role="region" aria-label={`${step.title} command`}>
                {step.command}
              </pre>
              <CopyButton textToCopy={step.command} />
            </div>
          )}

          {step.command && step.command.includes('\n') && (
            <details className="more">
              <summary>Show the commands</summary>
              <div className="code">
                <pre tabIndex={0} role="region" aria-label={`${step.title} commands`}>
                  {step.command}
                </pre>
                <CopyButton textToCopy={step.command} />
              </div>
            </details>
          )}

          {step.action && (
            <p className="step-action">
              <a href={step.action.href}>{step.action.label} →</a>
            </p>
          )}
        </li>
      ))}
    </ol>
  </section>
);
