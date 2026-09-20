"""Development-only prefix comparison; full production preparation remains mandatory."""
from pathlib import Path
import bz2,copy,time
import xml.etree.ElementTree as ET
from defusedxml.ElementTree import iterparse
from ovl_pipeline.data import extract_wikipedia,rows
from ovl_pipeline.canonical import write_json,file_hash,digest


def main():
    root=Path(__file__).parent;source=Path('.ovllm-cache/wikipedia/20260901/enwiki-20260901-pages-articles.xml.bz2')
    with bz2.open(source,'rb') as f:
        parser=iterparse(f,events=('start','end'),forbid_dtd=True,forbid_entities=True,forbid_external=True)
        _,doc=next(parser);ns=doc.tag.split('}')[0]+'}';sample=ET.Element(doc.tag,doc.attrib);eligible=0;pages=0
        for event,page in parser:
            if event!='end' or page.tag!=ns+'page':continue
            sample.append(copy.deepcopy(page));pages+=1
            rev=page.find(ns+'revision')
            if (page.findtext(ns+'ns')=='0' and page.find(ns+'redirect') is None
                and rev.findtext(ns+'model')=='wikitext' and rev.findtext(ns+'text')):eligible+=1
            page.clear();doc.clear()
            if eligible==1000:break
    sliced=root/'prefix.xml.bz2';sliced.write_bytes(bz2.compress(ET.tostring(sample,encoding='utf-8')))
    results=[]
    for workers in [1,8]:
        start=time.monotonic_ns();manifest=extract_wikipedia([sliced],root/f'workers-{workers}',workers=workers)
        results.append({'workers':workers,'elapsed_ms':(time.monotonic_ns()-start)//1000000,'manifest':manifest})
    assert results[0]['manifest']==results[1]['manifest']
    a=list(rows(root/'workers-8/articles.jsonl'));b=list(rows(Path('.ovllm-cache/extraction-timing-v1/partial-output/articles.jsonl')))
    assert len(a)==len(b)==1000
    for x,y in zip(a,b):
        x.pop('source');y.pop('source');assert x==y
    write_json(root/'comparison.json',{'result':'PASS','scope':'development first1000 eligible source articles; not full-corpus preparation/reconstruction',
        'original_source_sha256':'859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116',
        'prefix_source_sha256':file_hash(sliced),'pages':pages,'eligible_articles':1000,
        'serial_parallel_outputs_equal':True,'historical_serial_article_fields_equal_except_slice_source_identity':True,
        'runs':results,'code':{str(p):file_hash(p) for p in [Path(__file__),Path('src/ovl_pipeline/data.py'),Path('src/ovl_pipeline/extraction_workers.py')]}})
    print({'result':'PASS','timings_ms':[(r['workers'],r['elapsed_ms']) for r in results]})

if __name__=='__main__':main()
