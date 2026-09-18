"""Exploratory first-1000 eligible article timing; no full-corpus gate credit."""
from pathlib import Path
import time
import mwparserfromhell
from ovl_pipeline.data import extract_wikipedia
from ovl_pipeline.canonical import write_json,file_hash
root=Path(__file__).parent
original=mwparserfromhell.parse
elapsed=[]
class LimitReached(Exception):pass

def measured(*a,**kw):
    if len(elapsed)>=1000:raise LimitReached()
    start=time.monotonic_ns();result=original(*a,**kw);elapsed.append(time.monotonic_ns()-start)
    return result
mwparserfromhell.parse=measured
started=time.monotonic_ns()
try:
    extract_wikipedia([Path('.ovllm-cache/wikipedia/20260901/enwiki-20260901-pages-articles.xml.bz2')],root/'partial-output')
except LimitReached:pass
else:raise RuntimeError('expected exploratory limit')
write_json(root/'timing.json',{'schema':'ovl.extraction-timing.v1','scope':'exploratory first1000 eligible source articles only; no production preparation or full reconstruction credit','sample_count':len(elapsed),'total_elapsed_ms':(time.monotonic_ns()-started)//1000000,'parser_elapsed_ms':sum(elapsed)//1000000,'script_sha256':file_hash(Path(__file__)),'source_sha256':'859cf6cc1d13007d165025247cacaf463ff8a4ad700b19e476f7626e24169116','data_code_sha256':file_hash(Path('src/ovl_pipeline/data.py')),'partial_artifacts_preserved':True})
print((root/'timing.json').read_text())
