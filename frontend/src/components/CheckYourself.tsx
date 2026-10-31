import React from 'react';
import { copy } from '../content/copy';
import { SectionHeading } from './Contents';
import { CopyButton } from './CopyButton';
import { Ref } from './Ref';

export const CheckYourself: React.FC = () => (
  <section id="check" aria-labelledby="check-h">
    <SectionHeading id="check" />
    <p>{copy.check.intro}</p>
    {copy.check.levels.map((level, i) => (
      <div key={level.id} id={`check-${level.id}`}>
        <h3>
          <span className="secnum">2.{i + 1}</span> {level.title}
        </h3>
        <dl className="facts">
          <dt>Needs</dt>
          <dd>{level.cost}</dd>
          <dt>Shows</dt>
          <dd>{level.proves}</dd>
        </dl>
        {'command' in level && level.command && (
          <div className="code">
            <pre tabIndex={0} aria-label={`Commands: ${level.title}`}>
              <code>{level.command}</code>
            </pre>
            <CopyButton textToCopy={level.command} />
          </div>
        )}
        {'note' in level && level.note && (
          <p className="note">
            {level.note}
            <Ref k="readme" />
          </p>
        )}
        {'action' in level && level.action && (
          <p>
            <a href={level.action.href}>{level.action.label}</a>
          </p>
        )}
      </div>
    ))}
  </section>
);
