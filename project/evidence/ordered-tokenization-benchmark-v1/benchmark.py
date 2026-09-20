"""Ordered bounded-prefix benchmark only; no reconstruction/acceptance credit.
Runs the unchanged prepare_stream serialization and all text_ids checks. The
candidate only supplies encode() results from ordered encode_batch() calls.
"""
import argparse,hashlib,itertools,json,os,resource,time
from pathlib import Path
import tokenizers
import ovl_pipeline.data as data
from ovl_pipeline.canonical import file_hash,inventory,write_json
p=argparse.ArgumentParser();p.add_argument('--articles',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--documents',type=int,default=4096);p.add_argument('--batch-size',type=int,default=1);a=p.parse_args()
assert 1<=a.documents<=8192 and 1<=a.batch_size<=512
assert not a.output.exists();a.output.mkdir()
documents=list(itertools.islice(data.wikipedia_documents(a.articles),a.documents));assert len(documents)==a.documents
source=hashlib.sha256()
for d in documents:source.update(json.dumps(d,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()+b'\n')
real=tokenizers.Tokenizer.from_file(str(a.tokenizer))
class Ordered:
 def __init__(self):self.next=0;self.pending=[]
 def encode(self,text,add_special_tokens):
  assert add_special_tokens is False and documents[self.next]['text']==text
  if not self.pending:
   group=[];size=0
   for d in documents[self.next:self.next+a.batch_size]:
    n=len(d['text'].encode())
    if group and size+n>8*1024**2:break
    group.append(d['text']);size+=n
   self.pending=iter(real.encode_batch(group,add_special_tokens=False))
   self.remaining=len(group)
  result=next(self.pending);self.remaining-=1;self.next+=1
  if not self.remaining:self.pending=[]
  return result
 def decode(self,*args,**kwargs):return real.decode(*args,**kwargs)
ordered=Ordered()
if a.batch_size>1:
 class Factory:
  @staticmethod
  def from_file(path):assert path==str(a.tokenizer);return ordered
 data.Tokenizer=Factory
start=time.perf_counter();manifest=data.prepare_stream(iter(documents),a.tokenizer,a.output/'stream',phase='wikipedia');elapsed=time.perf_counter()-start
if a.batch_size>1:assert ordered.next==a.documents and not ordered.pending
report={'schema':'ovl.ordered-tokenization-benchmark.v1','scope':'ordered bounded prefix only; not full reconstruction, gate or corpus throughput forecast','documents':a.documents,'prefix_sha256':source.hexdigest(),'tokenizer_sha256':file_hash(a.tokenizer),'data_code_sha256':file_hash(Path(data.__file__)),'benchmark_sha256':file_hash(Path(__file__)),'batch_size':a.batch_size,'batch_byte_bound':8*1024**2,'oversized_single_document':'whole document preserved','tokenizers_version':tokenizers.__version__,'tokenizers_parallelism':os.environ.get('TOKENIZERS_PARALLELISM'),'rayon_num_threads':os.environ.get('RAYON_NUM_THREADS'),'elapsed_seconds':format(elapsed,'.9f'),'targets':manifest['targets'],'targets_per_second':format(manifest['targets']/elapsed,'.3f'),'maximum_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'files':inventory(a.output/'stream',['documents.jsonl','mask.u8','stream.json','tokens.u16']),'checks':'all unchanged text roundtrip, uint16 bound, target count, ordered index serialization, stream hashing retained'}
write_json(a.output/'measurement.json',report);print(json.dumps(report))
