from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import pytest

from ovl_pipeline.canonical import EvidenceError, digest, read_json, write_json
from ovl_pipeline.lifecycle_guard import supervise, validate_intent


def intent(now):
    return {"schema": "ovl.lifecycle-guard.v1",
            "identity": {"name": "synthetic-resource", "gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1, "cloud": "SECURE"},
            "created_not_before": now, "terminate_at": now+4, "grace_seconds": 120,
            "hourly_usd": "1", "rental_ceiling_usd": "0.10", "prior_upper_usd": "60",
            "stop_usd": "120", "cap_usd": "130"}


def test_independent_process_guard_survives_coordinator_death(tmp_path):
    now = int(time.time())
    plan = intent(now)
    write_json(tmp_path / "intent.json", plan)
    sleeper = tmp_path / "sleep.py"
    sleeper.write_text("import time\ntime.sleep(60)\n")
    resource = subprocess.Popen([sys.executable, str(sleeper)])
    coordinator = subprocess.Popen([sys.executable, str(sleeper)])
    try:
        start_ticks = Path(f"/proc/{resource.pid}/stat").read_text().split()[21]
        identity = {**plan["identity"], "id": str(resource.pid), "created_at": now, "start_ticks": start_ticks}
        write_json(tmp_path / "resource.json", identity)
        guard = tmp_path / "guard.py"
        guard.write_text('''from pathlib import Path
import os,signal,sys,time
from ovl_pipeline.canonical import read_json,digest,EvidenceError
from ovl_pipeline.lifecycle_guard import supervise
root=Path(sys.argv[1]);r=read_json(root/"resource.json");plan=read_json(root/"intent.json")
class Provider:
 def list(self, **kwargs):
  try: stat=Path("/proc/"+r["id"]+"/stat").read_text().split()
  except FileNotFoundError: return []
  if stat[21]!=r["start_ticks"]: raise EvidenceError("PID reused")
  return [] if stat[2]=="Z" else [r]
 def terminate(self, ident, **kwargs):
  assert ident==r["id"] and self.list()
  os.kill(int(ident),signal.SIGTERM)
supervise(plan,digest(plan),root/"guard-state",Provider(),interval=0.1)
''')
        with (tmp_path / "guard.log").open("w") as log:
            watchdog = subprocess.Popen([sys.executable, str(guard), str(tmp_path)], stdout=log, stderr=log)
        try:
            limit = time.time()+3
            state = tmp_path / "guard-state/guard.json"
            while not state.exists() and time.time() < limit:
                time.sleep(0.05)
            assert state.exists(), (tmp_path / "guard.log").read_text()
            coordinator.kill()
            coordinator.wait(timeout=3)
            assert watchdog.wait(timeout=10) == 0, (tmp_path / "guard.log").read_text()
            assert resource.wait(timeout=3) == -signal.SIGTERM
            receipt = read_json(state)
            assert receipt["status"] == "CLOSED" and receipt["resource_id"] == str(resource.pid)
        finally:
            if watchdog.poll() is None:
                watchdog.terminate()
                watchdog.wait(timeout=3)
    finally:
        for proc in (coordinator, resource):
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=3)


def test_guard_reconciles_lost_delete_response_and_never_extends_deadline(tmp_path):
    from ovl_pipeline.lifecycle import RetryableRead
    now = [100]
    p = intent(100)
    resource = {**p["identity"], "id": "synthetic", "created_at": 100}
    class Provider:
        resources = [resource]
        deletes = 0
        def list(self, **kwargs):
            return self.resources
        def terminate(self, ident, **kwargs):
            assert ident == "synthetic"
            self.deletes += 1
            self.resources = []
            raise RetryableRead("response lost after DELETE")
    provider = Provider()
    def sleep(n):
        now[0] += n
    result = supervise(p, digest(p), tmp_path, provider, clock=lambda: now[0], monotonic=lambda:now[0], sleep=sleep, interval=1)
    assert result["status"] == "CLOSED" and provider.deletes == 1 and now[0] == 104
    changed = {**p, "terminate_at": 106}
    with pytest.raises(EvidenceError, match="recovery identity"):
        supervise(changed, digest(changed), tmp_path, provider, clock=lambda: now[0], sleep=sleep)


def test_empty_listing_before_creation_keeps_guard_armed(tmp_path):
    p = intent(100)
    now = [100]
    class Provider:
        calls = 0
        def list(self, **kwargs):
            self.calls += 1
            return []
    provider = Provider()
    def sleep(n):
        now[0] += n
    result = supervise(p, digest(p), tmp_path, provider, clock=lambda: now[0], monotonic=lambda:now[0], sleep=sleep, interval=1)
    assert result["status"] == "CLOSED" and now[0] == 104 and provider.calls == 5


@pytest.mark.parametrize("change", [{"prior_upper_usd": "120"}, {"rental_ceiling_usd": "0.001"},
                                    {"grace_seconds": 121}, {"hourly_usd": "NaN"}])
def test_guard_budget_rejects_invalid_allowance(change):
    p = {**intent(100), **change}
    with pytest.raises(EvidenceError):
        validate_intent(p, digest(p))


def test_guard_cleans_up_attributed_wrong_hardware(tmp_path):
    p = intent(100)
    now = [100]
    class Provider:
        resources = [{**p["identity"], "id":"synthetic", "created_at":100, "gpu":"incorrect-provisioning"}]
        def list(self, **kwargs): return self.resources
        def terminate(self, ident, **kwargs):
            assert ident == "synthetic"
            self.resources=[]
    def sleep(n): now[0]+=n
    result=supervise(p,digest(p),tmp_path,Provider(),clock=lambda:now[0],monotonic=lambda:now[0],sleep=sleep,interval=1)
    assert result["status"]=="CLOSED" and result["provisioning_match"] is False and now[0]==101


def test_guard_disk_failure_does_not_block_owned_shutdown(tmp_path, monkeypatch):
    import ovl_pipeline.lifecycle_guard as guard
    p=intent(100)
    now=[104]
    class Provider:
        resources=[{**p["identity"],"id":"synthetic","created_at":100}]
        deletes=0
        def list(self, **kwargs): return self.resources
        def terminate(self,ident, **kwargs):
            self.deletes+=1
            self.resources=[]
    provider=Provider()
    def fail(*_): raise OSError("synthetic disk full")
    monkeypatch.setattr(guard,"write_json",fail)
    def sleep(n): now[0]+=n
    with pytest.raises(EvidenceError,match="absence verified"):
        supervise(p,digest(p),tmp_path,provider,clock=lambda:now[0],monotonic=lambda:now[0],sleep=sleep,interval=1)
    assert provider.deletes==1 and not provider.resources


def test_wall_clock_rollback_does_not_extend_deadline(tmp_path):
    p=intent(100)
    wall=[100]
    mono=[0]
    class Provider:
        resources=[{**p["identity"],"id":"synthetic","created_at":100}]
        stopped_at=None
        def list(self, **kwargs): return self.resources
        def terminate(self,ident, **kwargs):
            self.stopped_at=mono[0]
            self.resources=[]
    provider=Provider()
    def sleep(n):
        mono[0]+=n
        wall[0]-=10
    supervise(p,digest(p),tmp_path,provider,clock=lambda:wall[0],monotonic=lambda:mono[0],sleep=sleep,interval=1)
    assert provider.stopped_at==4


def test_transient_inventory_wait_stops_at_original_deadline(tmp_path):
    from ovl_pipeline.lifecycle import RetryableRead
    plan=intent(100);now=[100]
    class Provider:
        resources=[{**plan['identity'],'id':'synthetic','created_at':100}]
        stopped_at=None
        def list(self, **kwargs):
            if now[0]==103:
                raise RetryableRead('synthetic read outage just before deadline')
            return self.resources
        def terminate(self, ident, **kwargs):
            assert ident=='synthetic'
            self.stopped_at=now[0];self.resources=[]
            return True
    provider=Provider()
    def sleep(seconds):now[0]+=seconds
    result=supervise(plan,digest(plan),tmp_path,provider,clock=lambda:now[0],
                     monotonic=lambda:now[0],sleep=sleep,interval=3)
    assert result['status']=='CLOSED' and provider.stopped_at==104


def test_unresolved_post_cannot_disarm_guard_on_empty_deadline_listing(tmp_path):
    from ovl_pipeline.lifecycle_creation import update
    p=intent(100)
    update(tmp_path,digest(p),claim=('a'*64, {'synthetic':'request'}))
    now=[100]
    class Provider:
        deleted=False
        def list(self,**kwargs):
            if now[0] < 107 or self.deleted:
                return []
            return [{**p['identity'],'id':'late','created_at':106}]
        def terminate(self,ident,**kwargs):
            assert ident=='late' and now[0]>=107
            self.deleted=True
            return True
    provider=Provider()
    def sleep(n): now[0]+=n
    result=supervise(p,digest(p),tmp_path,provider,clock=lambda:now[0],monotonic=lambda:now[0],sleep=sleep,interval=1)
    assert result['status']=='CLOSED' and provider.deleted and now[0]==108


def test_accepted_but_not_yet_visible_resource_is_not_closed(tmp_path):
    from ovl_pipeline.lifecycle_creation import update
    p=intent(100)
    request={'synthetic':'request'}
    update(tmp_path,digest(p),claim=('a'*64, request))
    update(tmp_path,digest(p),accepted=('a'*64, request, 'late'))
    now=[100]
    class Provider:
        deleted=False
        def list(self,**kwargs):
            if now[0] < 107 or self.deleted: return []
            return [{**p['identity'],'id':'late','created_at':106}]
        def terminate(self,ident,**kwargs):
            assert ident=='late'
            if now[0]<107: return False
            self.deleted=True
            return True
    provider=Provider()
    def sleep(n): now[0]+=n
    result=supervise(p,digest(p),tmp_path,provider,clock=lambda:now[0],monotonic=lambda:now[0],sleep=sleep,interval=1)
    assert result['status']=='CLOSED' and provider.deleted and now[0]>=107


def test_two_journals_cannot_share_once_only_creation_claim(tmp_path):
    import concurrent.futures
    from ovl_pipeline.lifecycle_creation import update
    from ovl_pipeline.lifecycle import Pending
    p=intent(100)
    def claim(op):
        try:
            update(tmp_path,digest(p),claim=(op,{'synthetic':'request'}))
            return 'claimed'
        except Pending:
            return 'blocked'
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results=list(pool.map(claim,['a'*64,'b'*64]))
    assert sorted(results)==['blocked','claimed']
    update(tmp_path,digest(p),close=True)
    with pytest.raises(EvidenceError,match='closed'):
        update(tmp_path,digest(p),claim=('c'*64,{'synthetic':'request'}))


@pytest.mark.parametrize('reboot',[False,True])
def test_restart_preserves_elapsed_bound_and_reboot_cleans_immediately(tmp_path,reboot):
    p=intent(100);wall=[100];mono=[0]
    class Provider:
        resources=[{**p['identity'],'id':'synthetic','created_at':100}]
        stopped=None
        def list(self,**kwargs):return self.resources
        def terminate(self,ident,**kwargs):
            self.stopped=mono[0];self.resources=[];return True
    provider=Provider()
    def interrupted(event):
        if event=='armed':raise InterruptedError('synthetic supervisor loss')
    with pytest.raises(InterruptedError):
        supervise(p,digest(p),tmp_path,provider,clock=lambda:wall[0],monotonic=lambda:mono[0],event=interrupted)
    if reboot:
        state=read_json(tmp_path/'guard.json');state['boot_id']='previous-boot';write_json(tmp_path/'guard.json',state)
    wall[0]=20;mono[0]=2
    def sleep(n):mono[0]+=n
    result=supervise(p,digest(p),tmp_path,provider,clock=lambda:wall[0],monotonic=lambda:mono[0],sleep=sleep,interval=1)
    assert result['status']=='CLOSED' and provider.stopped==(2 if reboot else 4)


def test_poll_interval_cannot_cross_frozen_termination_time(tmp_path):
    p=intent(100);now=[100]
    class Provider:
        resources=[{**p['identity'],'id':'synthetic','created_at':100}]
        stopped=None
        def list(self,**kwargs):return self.resources
        def terminate(self,ident,**kwargs):self.stopped=now[0];self.resources=[];return True
    provider=Provider()
    def sleep(n):now[0]+=n
    supervise(p,digest(p),tmp_path,provider,clock=lambda:now[0],monotonic=lambda:now[0],sleep=sleep,interval=30)
    assert provider.stopped==104


def test_reboot_cleanup_time_advances_when_wall_clock_is_behind(tmp_path):
    from ovl_pipeline.lifecycle import RetryableRead,Pending
    p=intent(100);now=[0]
    state={'intent_sha256':digest(p),'resource_id':'synthetic','status':'ARMED','last_observed':100,
           'provisioning_match':True,'resource_seen':True,'deletion_confirmed':False,
           'boot_id':'previous-boot','stop_monotonic_ms':104000}
    write_json(tmp_path/'guard.json',state)
    class Provider:
        def list(self,**kwargs):raise RetryableRead('synthetic unavailable')
        def terminate(self,ident,**kwargs):raise RetryableRead('synthetic unavailable')
    def sleep(n):now[0]+=n;assert now[0]<=120
    with pytest.raises(Pending,match='authenticate'):
        supervise(p,digest(p),tmp_path,Provider(),clock=lambda:20,monotonic=lambda:now[0],sleep=sleep,interval=30)
    assert now[0]==120
