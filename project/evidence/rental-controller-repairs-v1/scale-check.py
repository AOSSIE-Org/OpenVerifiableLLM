from pathlib import Path
import json,time
from datetime import datetime,timezone
from test_rental_controller import RentalFake
from ovl_pipeline.supervision import rental_plan,Journal
from ovl_pipeline.canonical import digest,write_json
root=Path('.ovllm-cache/rental-scale-v2');f=RentalFake(root)
v=dict(f.i['plan']['input']);v.update(maximum_seconds=86400,allowance_usd='10')
f.i['plan']=rental_plan(v)
deadline=datetime.fromtimestamp(f.i['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
f.i['payload']['terminateAfter']=deadline;f.value['payload']['terminateAfter']=deadline;f.refresh()
def sleep(seconds):
 f.now+=seconds;f.elapsed+=seconds
 assert f.elapsed<90000
f.sleep=sleep;started=time.monotonic();f.run();execution=time.monotonic()-started
started=time.monotonic()
with Journal(f.directory).lease() as j:
 count=len(j.events);last=digest(j.events[-1])
parse=time.monotonic()-started
report={'schema':'ovl.controller-scale-check.v1','scope':'one simulated 24-hour healthy rental at production polling cadence; no provider requests or paid resource',
 'simulated_elapsed_seconds':int(f.elapsed),'event_count':count,'journal_bytes':sum(p.stat().st_size for p in f.directory.glob('event-*.json')),
 'wall_ms':int(execution*1000),'restart_parse_ms':int(parse*1000),'last_event_sha256':last,'create_requests':f.writes,'terminated':not f.alive,
 'provider_billing':'NOT_RUN'}
write_json(root/'scale-report.json',report);print(json.dumps(report))
