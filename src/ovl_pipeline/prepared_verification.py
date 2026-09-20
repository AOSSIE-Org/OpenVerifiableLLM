"""Complete prepared-artifact integrity/accounting checks, not raw reconstruction.

The expected roots come from separately trusted registration/source policy. This
validator walks every ledger/article and all stream files; it never samples or
claims to have repeated extraction, tokenization or training.
"""
from collections import Counter
from pathlib import Path

from .canonical import EvidenceError,Merkle,canonical,confined,digest,read_json,require_digest,sha256,verify_inventory
from .data import rows,validate_stream
from .schema import fields,integer

STAGES={'corpus':'corpus','tokenizer':'tokenizer','conversation-selection':'conversation_selection'}
FILES={'corpus':'corpus.json','tokenizer':'tokenizer-manifest.json','conversation-selection':'selection.json',
       'wikipedia':'stream.json','conversation':'stream.json','conversation-validation':'stream.json'}


def corpus_accounting(directory, manifest):
    fields(manifest,'schema policy sources counts record_count ledger_root files','corpus manifest')
    if manifest['schema']!='ovl.corpus.v1' or manifest['policy']!='main-nonredirect-stripcode-v1':
        raise EvidenceError('unsupported corpus policy')
    if type(manifest['sources']) is not list or len(manifest['sources'])!=1:
        raise EvidenceError('invalid corpus source list')
    integer(manifest['record_count'],1,2**53-1,'corpus record count')
    if type(manifest['counts']) is not dict:raise EvidenceError('invalid corpus counts')
    for count in manifest['counts'].values():integer(count,1,2**53-1,'corpus reason count')
    for root in manifest['sources']:require_digest(root)
    counts=Counter();ordinals=Counter();root=Merkle();source_order=0
    articles=iter(rows(directory/'articles.jsonl'));included=0
    reasons={'included','redirect','non_main_namespace','non_wikitext_model','empty_revision_text','empty_extracted_text'}
    for row in rows(directory/'ledger.jsonl'):
        reason=row.get('reason')
        if reason not in reasons:raise EvidenceError('unsupported corpus exclusion')
        expected='source source_order ordinal page_id revision_id title namespace timestamp attribution_url revision_url history_url raw_text_sha256 reason'
        fields(row,expected+(' text_sha256' if reason=='included' else ''),'corpus ledger record')
        index=row['source_order']
        if type(index) is not int or not 0<=index<len(manifest['sources']):raise EvidenceError('invalid ledger source ordinal')
        if index<source_order or index>source_order+1 or row['source']!=manifest['sources'][index]:raise EvidenceError('ledger source order/identity differs')
        if type(row['ordinal']) is not int or row['ordinal']!=ordinals[index]:raise EvidenceError('ledger ordinal gap/duplicate')
        source_order=index;ordinals[index]+=1
        require_digest(row['raw_text_sha256'])
        if reason=='included':
            article=next(articles,None)
            if article is None or type(article.get('text')) is not str or not article['text']:
                raise EvidenceError('included ledger entry lacks nonempty article')
            if sha256(article['text'].encode())!=row['text_sha256']:
                raise EvidenceError('article text differs from ledger digest')
            if {k:v for k,v in article.items() if k!='text'}!={k:v for k,v in row.items() if k!='reason'}:
                raise EvidenceError('article lineage differs from ledger')
            included+=1
        counts[reason]+=1;root.add(canonical(row))
    if next(articles,None) is not None:raise EvidenceError('unaccounted extra article')
    if (not included or dict(counts)!=manifest['counts'] or sum(counts.values())!=manifest['record_count']
            or root.root()!=manifest['ledger_root']):raise EvidenceError('corpus count/root reconciliation failed')
    return {'records':sum(counts.values()),'included':included,'counts':dict(counts),'ledger_root':root.root()}


def verify_prepared(directory:Path,expected_sha256,source_sha256):
    require_digest(expected_sha256);require_digest(source_sha256)
    if directory.is_symlink() or not directory.is_dir():raise EvidenceError('prepared root must be a regular directory')
    value=read_json(confined(directory,'preparation.json'))
    if digest(value)!=expected_sha256:raise EvidenceError('prepared root differs from external registration')
    fields(value,'schema source_commitment_sha256 corpus tokenizer conversation_selection streams code environment validation_used_for_training','complete preparation')
    if (value['schema']!='ovl.complete-preparation.v1' or value['source_commitment_sha256']!=source_sha256
            or value['validation_used_for_training'] is not False):raise EvidenceError('preparation source/split policy differs')
    if set(value['streams'])!={'wikipedia','conversation','conversation-validation'}:raise EvidenceError('all prepared splits required')
    allowed={'preparation.json'}
    for stage,filename in FILES.items():
        manifest=value[STAGES[stage]] if stage in STAGES else value['streams'][stage]
        folder=confined(directory,stage)
        if read_json(confined(folder,filename))!=manifest:raise EvidenceError('stage manifest differs from preparation root')
        verify_inventory(folder,manifest['files'])
        expected_names={e['path'] for e in manifest['files']}|{filename}
        actual_names=set()
        for path in folder.rglob('*'):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):raise EvidenceError('nonregular prepared artifact')
            if path.is_file():actual_names.add(path.relative_to(folder).as_posix())
        if actual_names!=expected_names:raise EvidenceError('unexpected/missing prepared stage artifact')
        allowed.add(stage)
    if {p.name for p in directory.iterdir()}!=allowed:raise EvidenceError('unexpected/missing prepared root artifact')
    checked=corpus_accounting(directory/'corpus',value['corpus'])
    tokenizer=value['tokenizer'];tokenizer_file=next(e['sha256'] for e in tokenizer['files'] if e['path']=='tokenizer.json')
    for name,stream in value['streams'].items():
        if stream['tokenizer_sha256']!=tokenizer_file:raise EvidenceError('stream tokenizer parent differs')
        if stream['phase']!=('wikipedia' if name=='wikipedia' else 'conversation'):raise EvidenceError('stream phase differs')
        validate_stream(directory/name,stream)
    if value['streams']['wikipedia']['documents']!=checked['included']:raise EvidenceError('article/stream document count differs')
    articles=iter(rows(directory/'corpus/articles.jsonl'))
    for document in rows(directory/'wikipedia/documents.jsonl'):
        article=next(articles,None)
        if article is None or document['identity']!=[article['source'],article['page_id'],article['revision_id']]:
            raise EvidenceError('Wikipedia stream document ancestry differs')
    if next(articles,None) is not None:raise EvidenceError('Wikipedia stream omits articles')
    for split,name in [('train','conversation'),('validation','conversation-validation')]:
        selected=iter(rows(directory/'conversation-selection'/f'{split}.jsonl'));count=0
        for document in rows(directory/name/'documents.jsonl'):
            conversation=next(selected,None)
            if conversation is None or document['identity']!=conversation['identity'] or conversation['identity'][0]!=split:
                raise EvidenceError('conversation stream split/membership differs')
            count+=1
        if next(selected,None) is not None or count!=value['conversation_selection']['conversations'][split]:
            raise EvidenceError('conversation stream omits selected conversations')
    return {'schema':'ovl.prepared-integrity-verification.v1','result':'PASS','preparation_sha256':expected_sha256,
            'source_commitment_sha256':source_sha256,'corpus':checked,'scope':'complete-prepared-file-integrity-and-accounting',
            'locally_recomputed':['all_stage_inventories','entire_corpus_ledger_and_articles','all_stream_hashes_and_layout'],
            'raw_transformations_reconstructed':'NOT_RUN','training_replay':'NOT_RUN','production_admission':'NOT_RUN'}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--expected-preparation-sha256',required=True)
    p.add_argument('--expected-source-sha256',required=True)
    a=p.parse_args()
    try:result=verify_prepared(a.prepared,a.expected_preparation_sha256,a.expected_source_sha256)
    except Exception as error:
        print(canonical({'result':'FAIL','scope':'prepared-integrity-only','reason':str(error)}).decode());return 1
    print(canonical(result).decode());return 0


if __name__=='__main__':raise SystemExit(main())
