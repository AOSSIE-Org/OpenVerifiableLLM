"""Once-only complete anonymous range verification after preserved short response."""
from pathlib import Path
import os,sys,time
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import canonical,digest,file_hash,read_json,verify_inventory,write_json
root=Path.cwd();base=root/'.ovllm-cache/full-prepared-publication-v1'
intent=read_json(base/'range-download-intent-v3.json')
verify_inventory(root,intent['source_files']);assert file_hash(Path(__file__))==intent['driver_sha256']
assert read_json(base/'public-range-download-v2/failure.json')==intent['predecessor_failure']
plan=read_json(base/'plan.json');receipt=read_json(base/'upload/upload.json');upload=read_json(base/'upload/intent.json')
assert file_hash(base/'plan.json')==intent['plan_sha256'] and upload['plan']==plan
assert receipt['intent_sha256']==digest(upload) and receipt['result']=='UPLOADED_NOT_DOWNLOAD_VERIFIED'
assert receipt['revision']==intent['revision'] and receipt['repo']==plan['repo'] and receipt['prefix']==plan['prefix']
output=base/'public-range-download-v3';assert not output.exists()
with (base/'range-download-launch-v3.json').open('xb') as f:
 f.write(canonical({'intent_sha256':digest(intent),'observed_epoch':int(time.time()),'pid':os.getpid()}));f.flush();os.fsync(f.fileno())
from verify_prepared_public_ranges import verify
started=time.monotonic()
result=verify(base/'plan.json',intent['plan_sha256'],intent['revision'],root/'.ovllm-cache/production-preparation-v1',output,maximum_attempts=6)
write_json(output/'measurement.json',{'verification_sha256':digest(result),'elapsed_ms':int((time.monotonic()-started)*1000),'scope':'complete public71GBbyte verification only; no raw reconstruction or replay'})
print('Complete public prepared ranges',result['result'],digest(result),flush=True)
