"""Actual file/counter/health integration; HTTPS, clock and installer are doubles."""
from datetime import datetime,timezone
import hashlib,json,sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_fetch_runtime as fetcher
import pod_public_setup as setupper
from sustained_health import SustainedHealth
from test_external_watchdog import intent,NOW
from test_workload_health import Clock,JOB,SELECTION
from test_pod_fetch_runtime import setup,Response
from test_pod_public_setup import fixture
from ovl_pipeline.canonical import digest,file_hash,write_json
from ovl_pipeline.supervision import Journal,rental_plan,observe


@pytest.mark.parametrize('connected',[False,True])
def test_actual_bytes_bridge_long_setup_to_unchanged_stall_guard(tmp_path,monkeypatch,connected):
    data=b'x'*(8*1024**2);plan,item,client=setup(tmp_path,data);pin=file_hash(plan)
    report=tmp_path/'evidence/downloads.json';activity=report.parent/'activity.json'
    if connected:monkeypatch.setenv('OVL_ACTIVITY_FILE',str(activity))
    else:monkeypatch.delenv('OVL_ACTIVITY_FILE',raising=False)
    c=Clock();w=intent();w['plan']=rental_plan({**w['plan']['input'],'maximum_seconds':3600})
    w['payload']['terminateAfter']=datetime.fromtimestamp(w['plan']['provider_terminate_epoch'],timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    binding={'schema':'ovl.public-input-binding.v1','plan_sha256':pin,'bytes':len(data)}
    decisions=[]
    with Journal(tmp_path/'journal').lease() as j:
        h=SustainedHealth(j,w,'owned-pod',{}, {JOB:binding},wall=lambda:c.now,
                          clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})
        h.start_job({**SELECTION,'kind':'setup'})
        def check():
            if activity.exists():h.activity(JOB,json.loads(activity.read_bytes()))
            p=w['plan'];v={'schema':'ovl.supervisor-observation.v2','now_epoch':c.now,'observed_epoch':c.now,
              'attributed_pod_ids':['owned-pod'],'active_pod_ids':['owned-pod'],'pod_id':'owned-pod','gpu_count':1,
              'hourly_usd':'0.3','actual_project_spend_usd':'0.068185','outstanding_usd':'0.15',
              'reserved_remaining_usd':'60','account_balance_usd':'99','progress_epoch':h.progress,
              'last_checkpoint_epoch':h.exported,'terminate_after_request_epoch':p['provider_terminate_epoch'],
              'watchdog_observed_epoch':c.now,'watchdog_plan_sha256':digest(p),
              'watchdog_external_terminate_epoch':p['external_terminate_epoch'],'watchdog_state':'ARMED'}
            decisions.append(observe(p,v));return decisions[-1]
        class SlowResponse(Response):
            def read(self,n):
                check();c.advance(50);return super().read(n)
        client.open=lambda request,timeout:SlowResponse(data,request.full_url)
        result=fetcher.fetch(plan,pin,tmp_path/'wheels',report,NOW+600,opener=client,
                             wall=lambda:c.now,monotonic=lambda:c.ms/1000)
        check();assert c.now-NOW==450
        # The measured offline phase performs no invented progress updates.
        c.advance(115);check()
        if connected:
            assert all(d['action']=='CONTINUE' for d in decisions)
            assert h.progress==NOW+400 and h.exported==NOW and not h.complete
            c.advance(136)
            assert 'stalled-or-future-progress' in check()['reasons']
        else:
            assert any('stalled-or-future-progress' in d['reasons'] for d in decisions)
        assert result['files'][0]['sha256']==hashlib.sha256(data).hexdigest()
        assert (tmp_path/'wheels'/item['path']).read_bytes()==data


def test_setup_forwards_selected_activity_only_to_download_child(tmp_path,monkeypatch):
    inputs,config,value=fixture(tmp_path);output=tmp_path/'output';calls=[]
    activity=str(output/'activity.json');monkeypatch.setenv('OVL_ACTIVITY_FILE',activity)
    def execute(argv,**kwargs):
        calls.append(argv)
        if argv[3].endswith('fetch.py'):
            assert kwargs['env']['OVL_ACTIVITY_FILE']==activity
            f=json.loads((inputs/'plan.json').read_bytes())['files'][0]
            write_json(output/'downloads.json',{'schema':'ovl.public-wheel-download-result.v1',
                'plan_sha256':value['wheel_plan_sha256'],'files':[{**f,'result':'COMPLETE_HASH_MATCH'}]})
        else:
            assert 'OVL_ACTIVITY_FILE' not in kwargs['env']
            (output/'offline').mkdir();write_json(output/'offline/setup.json',{
                'schema':'ovl.offline-runtime-setup-result.v1','result':'PASS','config_sha256':value['offline_config_sha256']})
    setupper.setup(config,file_hash(config),inputs,tmp_path/'runtime',output,int(setupper.time.time())+900,execute=execute)
    assert len(calls)==2


@pytest.mark.parametrize('damage',['foreign','existing','symlink'])
def test_activity_destination_rejected_before_network(tmp_path,monkeypatch,damage):
    plan,item,client=setup(tmp_path);report=tmp_path/'evidence/downloads.json'
    report.parent.mkdir();activity=report.parent/'activity.json'
    if damage=='foreign':activity=tmp_path/'other.json'
    elif damage=='existing':activity.write_text('preserved')
    else:activity.symlink_to(tmp_path/'outside')
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(activity))
    with pytest.raises(ValueError,match='activity'):
        fetcher.fetch(plan,file_hash(plan),tmp_path/'wheels',report,int(fetcher.time.time())+60,opener=client)
    assert not client.calls and not (tmp_path/'wheels').exists()


def test_retry_highwater_never_counts_response_bytes_twice(tmp_path,monkeypatch):
    data=b'x'*(3*1024**2);plan,item,client=setup(tmp_path,data)
    report=tmp_path/'evidence/downloads.json';activity=report.parent/'activity.json'
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(activity));calls=[];counts=[]
    original_replace=fetcher.os.replace
    def replace(source,dest):
        original_replace(source,dest)
        if Path(dest)==activity:counts.append(json.loads(activity.read_bytes())['received_bytes'])
    monkeypatch.setattr(fetcher.os,'replace',replace)
    class Interrupted(Response):
        def read(self,n):
            if self.tell():raise TimeoutError('synthetic read interruption after actual bytes')
            return super().read(n)
    def open(request,timeout):
        calls.append(request.full_url)
        return (Interrupted if len(calls)==1 else Response)(data,request.full_url)
    client.open=open
    result=fetcher.fetch(plan,file_hash(plan),tmp_path/'wheels',report,int(fetcher.time.time())+60,opener=client)
    assert len(calls)==2 and counts==sorted(counts) and counts[-1]==len(data)
    assert all(0<=n<=len(data) for n in counts)
    assert result['files'][0]['result']=='COMPLETE_HASH_MATCH'
    assert len(list((report.parent/'downloads.json.attempts').glob('*.partial')))==1


def test_parallel_workers_publish_one_monotonic_bounded_counter(tmp_path,monkeypatch):
    data=b'x'*(2*1024**2);plan,item,client=setup(tmp_path,data)
    files=[{**item,'path':f'fixture-{n}-py3-none-any.whl'} for n in range(8)]
    plan.write_text(json.dumps({'schema':'ovl.public-wheel-download.v1','files':files}))
    report=tmp_path/'evidence/downloads.json';activity=report.parent/'activity.json'
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(activity));counts=[];identities=set()
    original_replace=fetcher.os.replace
    def replace(source,dest):
        original_replace(source,dest)
        if Path(dest)==activity:
            v=json.loads(activity.read_bytes());counts.append(v['received_bytes']);identities.add((v['process_instance'],v['pid']))
    monkeypatch.setattr(fetcher.os,'replace',replace)
    fetcher.fetch(plan,file_hash(plan),tmp_path/'wheels',report,int(fetcher.time.time())+60,opener=client)
    assert counts==sorted(counts) and counts[-1]==8*len(data) and len(identities)==1
    assert not activity.with_name('activity.json.pending').exists()


def test_setup_rejects_unselected_activity_before_child_execution(tmp_path,monkeypatch):
    inputs,config,value=fixture(tmp_path);monkeypatch.setenv('OVL_ACTIVITY_FILE',str(tmp_path/'other.json'))
    def execute(*args,**kwargs):raise AssertionError('unselected path must not launch child')
    with pytest.raises(ValueError,match='activity path'):
        setupper.setup(config,file_hash(config),inputs,tmp_path/'runtime',tmp_path/'output',
                       int(setupper.time.time())+900,execute=execute)
