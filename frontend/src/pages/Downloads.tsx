import { ClaimBlock } from '../components/ClaimBlock';
import { DownloadsList } from '../components/Downloads';
import { Hash } from '../components/Hash';
import { Section } from '../components/Section';
import { L } from '../components/text';
import { claimsOn, ledger } from '../content/ledger';

const ledgerHash = (id: string) => {
  const v = ledger.claims.find((c) => c.id === id)?.value;
  return typeof v === 'string' && /^[0-9a-f]{64}$/.test(v) ? v : null;
};

export default function Downloads() {
  const hfFiles = claimsOn('downloads').filter((c) => c.id.startsWith('DL-') && c.sources.every((s) => s.rev.startsWith('hf:')));
  return (
    <>
      <h1><L k="h1.downloads" /></h1>
      <p className="lead"><L k="downloads.lead" /></p>
      <Section needs={['preparation']}>
        <ClaimBlock id="OV-05" />
      </Section>
      <h2><L k="h2.hf" /></h2>
      {hfFiles.map((c) => <ClaimBlock key={c.id} id={c.id} />)}
      <Section needs={['downloads']}>
        <DownloadsList />
      </Section>
      <h2><L k="h2.github" /></h2>
      {['DL-gguf', 'DL-modelfile'].map((id) => {
        const h = ledgerHash(id);
        return (
          <ClaimBlock key={id} id={id}>
            {h ? <Hash value={h} /> : null}
          </ClaimBlock>
        );
      })}
      <h2><L k="h2.elsewhere" /></h2>
      <ClaimBlock id="DL-hf" />
      <ClaimBlock id="DL-ollama" />
    </>
  );
}
