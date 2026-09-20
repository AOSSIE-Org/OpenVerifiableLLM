"""Development-only 16-process trial; candidate override not production authority."""
from pathlib import Path
import time
from ovl_pipeline import data
from ovl_pipeline.extraction_workers import OrderedExtraction
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash
class Candidate16(OrderedExtraction):
    def __init__(self,emit,*,workers):
        super().__init__(emit,workers=8)
        self.workers=16 # Explicit trial override of worker cap; all transform code unchanged.
def main():
    root=Path(__file__).parent;data.OrderedExtraction=Candidate16
    source=Path('.ovllm-cache/extraction-parallel-comparison-v1/prefix.xml.bz2')
    before=time.monotonic_ns();v=data.extract_wikipedia([source],root/'output')
    elapsed=(time.monotonic_ns()-before)//1000000
    baseline=read_json(Path('.ovllm-cache/extraction-parallel-comparison-v2/comparison.json'))
    assert v==baseline['runs'][0]['manifest']
    write_json(root/'comparison.json',dict(result='PASS',scope='candidate16 development prefix only; explicit worker-count override, not retained or production credit',workers=16,elapsed_ms=elapsed,prefix_sha256=file_hash(source),baseline_sha256=digest(baseline),manifest=v,harness_sha256=file_hash(Path(__file__))))
    print({'workers':16,'elapsed_ms':elapsed,'same_manifest':True})
if __name__=='__main__':main()
