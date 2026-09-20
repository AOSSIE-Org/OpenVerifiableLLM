"""Once-only unpaid continuation after the existing full upload succeeds."""
from pathlib import Path
import re,sys,time
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import digest,file_hash,read_json,verify_inventory,write_json

root=Path.cwd();base=root/'.ovllm-cache/full-prepared-publication-v1'
pins=read_json(base/'download-helper-pins-v1.json');verify_inventory(root,pins)
plan=read_json(base/'plan.json');expected='87c1544b5b3786fbf23eaddd812f58b5cffb5605c8050553d0134b579fa1a730'
assert file_hash(base/'plan.json')==expected
intent=read_json(base/'upload/intent.json');receipt=read_json(base/'upload/upload.json')
assert intent['plan_sha256']==expected and intent['plan']==plan
assert receipt['schema']=='ovl.evidence-publication.v1' and receipt['result']=='UPLOADED_NOT_DOWNLOAD_VERIFIED'
assert receipt['intent_sha256']==digest(intent) and receipt['repo']==plan['repo'] and receipt['prefix']==plan['prefix']
assert re.fullmatch('[0-9a-f]{40}',receipt['revision'])
fence=base/'download-launch-v1.json';output=base/'public-stream-download-v1'
assert not fence.exists() and not output.exists(),'Preserve the prior once-only launch; inspect it instead of retrying.'
# Exclusive creation before hashing/network work, including restart after failure.
with fence.open('xb') as stream:
    from ovl_pipeline.canonical import canonical
    import os
    stream.write(canonical({'schema':'ovl.prepared-public-download-launch.v1','observed_epoch':int(time.time()),
        'upload_receipt_sha256':digest(receipt),'plan_sha256':expected,'revision':receipt['revision'],
        'helper_inventory_sha256':digest(pins),'source_driver_sha256':file_hash(Path(__file__)),
        'scope':'single complete anonymous response verification after selected upload; no GPU or HF mutation'}))
    stream.flush();os.fsync(stream.fileno())
from verify_prepared_public_stream import verify
result=verify(base/'plan.json',expected,receipt['revision'],root/'.ovllm-cache/production-preparation-v1',output)
print('Complete public prepared response verification',result['result'],digest(result),flush=True)
