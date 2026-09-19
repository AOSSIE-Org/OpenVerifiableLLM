"""Read-only GraphQL validation: deployment resolver is literally skipped."""
from pathlib import Path
import sys,time,hashlib
sys.path[:0]=['src','scripts']
import probe_provider_deadline as p
from provider_request_receipts import Opener,private_directory,shape
from ovl_pipeline.canonical import read_json,write_json,digest,file_hash
out=Path('.ovllm-cache/provider-schema-only-v1');out.mkdir(exist_ok=False)
private=private_directory(out/'private')
query='''mutation OvlSkippedFeasibilitySchema($input: PodFindAndDeployOnDemandInput!) {
 podFindAndDeployOnDemand(input:$input) @skip(if:true) { id name createdAt gpuCount imageName } }'''
payload=read_json(Path('.ovllm-cache/sustained-feasibility-v1/rental-intent.json'))['payload']
request={'query':query,'variables':{'input':payload},'scope':'Schema validation only; resolver has literal @skip(if:true). No creation, capacity, past-failure or absence evidence.'}
write_json(out/'request.json',request)
capture={'body':bytearray(),'observed_read_bytes':0,'http_status':None,'content_encoding':None,'same_endpoint':None}
original=p.build_opener;p.build_opener=lambda *a,**k:Opener(original(*a,**k),capture)
p.OPERATIONS['schema-only']=query
error=None;data=None;clock=None
try:data,sha,clock=p.request('schema-only',request['variables'])
except Exception as e:error=p.diagnostic(e)
finally:p.build_opener=original
raw=bytes(capture.pop('body'));rawpath=private/'response.bin'
with rawpath.open('xb') as f:
 __import__('os').fchmod(f.fileno(),0o600);f.write(raw)
result={'schema':'ovl.skipped-provider-schema-probe.v1','observed_epoch':int(time.time()),'request_sha256':digest(request),'driver_sha256':file_hash(Path(__file__)),'response_sha256':hashlib.sha256(raw).hexdigest(),'http':capture,'shape':shape(raw),'failure':error,'empty_skipped_data':data=={} if error is None else False,'http_clock':clock,'scope':request['scope']}
write_json(out/'result.json',result)
print(result)
