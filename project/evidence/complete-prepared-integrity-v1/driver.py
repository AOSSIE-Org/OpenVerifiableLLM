"""Once-only full local prepared-data integrity check; no paid resources."""
from pathlib import Path
import sys,time
sys.path.insert(0,'src')
from ovl_pipeline.canonical import digest,file_hash,read_json,verify_inventory,write_json

root=Path.cwd();base=root/'.ovllm-cache/complete-prepared-integrity-v1'
intent=read_json(base/'intent.json')
verify_inventory(root,intent['source_files'])
assert file_hash(Path(__file__))==intent['driver_sha256']
fence=base/'started.json'
assert not fence.exists() and not(base/'verification.json').exists()
import os
from ovl_pipeline.canonical import canonical
with fence.open('xb') as stream:
    stream.write(canonical({'intent_sha256':digest(intent),'observed_epoch':int(time.time()),'pid':os.getpid()}))
    stream.flush();os.fsync(stream.fileno())
started=time.monotonic()
try:
    from ovl_pipeline.prepared_verification import verify_prepared
    result=verify_prepared(root/intent['prepared_directory'],intent['preparation_sha256'],intent['source_sha256'])
    write_json(base/'verification.json',result)
    write_json(base/'measurement.json',{'schema':'ovl.full-prepared-integrity-measurement.v1','intent_sha256':digest(intent),
        'verification_sha256':digest(result),'elapsed_ms':int((time.monotonic()-started)*1000),
        'finished_epoch':int(time.time()),'scope':'complete local prepared-artifact integrity and accounting; raw reconstruction and training replay NOT_RUN'})
    print('Complete prepared integrity',result['result'],digest(result),flush=True)
except Exception as error:
    write_json(base/'failure.json',{'intent_sha256':digest(intent),'result':'FAIL','error_type':type(error).__name__,
                                  'elapsed_ms':int((time.monotonic()-started)*1000)})
    raise
