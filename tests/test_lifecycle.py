from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

import pytest

from ovl_pipeline.canonical import EvidenceError, read_json, write_json
from ovl_pipeline.lifecycle import (Journal, Observation, Pending, RetryableRead,
                                    exclusive, reconcile_billing)
from ovl_pipeline.lifecycle_fixture import FixtureLifecycle, check_snapshot, snapshot


FIXTURE = Path(__file__).parent / "fixtures" / "pipeline"


class Effect:
    def __init__(self):
        self.calls = 0
        self.result = None
        self.lost_response = False
        self.read_failures = 0
        self.error = None

    def observe(self, operation, request):
        if self.error:
            raise self.error
        if self.read_failures:
            self.read_failures -= 1
            raise RetryableRead("synthetic unavailable read")
        return Observation("complete", self.result) if self.result else Observation("absent")

    def submit(self, operation, request):
        self.calls += 1
        self.result = {"operation": operation, "request": request}
        if self.lost_response:
            raise ConnectionError("synthetic response lost after remote commit")

    def validate(self, operation, request, result):
        if result != {"operation": operation, "request": request}:
            raise EvidenceError("effect identity mismatch")


def test_lost_response_reconciles_without_resubmission(tmp_path):
    adapter = Effect()
    adapter.lost_response = True
    with exclusive(tmp_path):
        j = Journal(tmp_path, {"fixture": 1})
        with pytest.raises(ConnectionError):
            j.effect("create", {"resource": "synthetic"}, adapter, deadline=time.time()+10)
    with exclusive(tmp_path):
        recovered = Journal(tmp_path, {"fixture": 1})
        recovered.effect("create", {"resource": "synthetic"}, adapter, deadline=time.time()+10)
    assert adapter.calls == 1
    assert read_json(tmp_path / "journal.json")["operations"][0]["status"] == "complete"


def test_uncertain_absence_does_not_create_again(tmp_path):
    adapter = Effect()
    with exclusive(tmp_path):
        j = Journal(tmp_path, {"fixture": 1})
        item = j.operation("create", {})
        item["status"] = "sent"
        j.save()
        with pytest.raises(Pending, match="uncertain"):
            j.effect("create", {}, adapter, deadline=time.time()+10)
    assert adapter.calls == 0


def test_remote_acknowledged_mutation_survives_process_death(tmp_path):
    child = tmp_path / "effect.py"
    child.write_text('''from pathlib import Path
import os,sys,time
from ovl_pipeline.canonical import read_json,write_json,EvidenceError
from ovl_pipeline.lifecycle import Journal,Observation,exclusive
root=Path(sys.argv[1]);remote=root/"remote.json"
class Adapter:
 def observe(self,op,request):
  return Observation("complete",read_json(remote)) if remote.exists() else Observation("absent")
 def submit(self,op,request):
  with (root/"submissions").open("a") as f: f.write(op+"\\n"); f.flush(); os.fsync(f.fileno())
  write_json(remote,{"operation":op,"request":request})
  os._exit(74)
 def validate(self,op,request,result):
  if result!={"operation":op,"request":request}: raise EvidenceError("wrong identity")
with exclusive(root/"private"):
 Journal(root/"private",{}).effect("create",{"synthetic":True},Adapter(),deadline=time.time()+10)
''')
    assert launch(child, tmp_path).returncode == 74
    second = launch(child, tmp_path)
    assert second.returncode == 0, second.stderr
    assert len((tmp_path / "submissions").read_text().splitlines()) == 1


@pytest.mark.parametrize("error", [PermissionError("remote descriptor denied"), EvidenceError("wrong identity"),
                                  EvidenceError("invalid signature"), OSError("unclassified transport")])
def test_strict_errors_are_not_transient(tmp_path, error):
    adapter = Effect()
    adapter.error = error
    with exclusive(tmp_path):
        with pytest.raises(type(error), match=str(error)):
            Journal(tmp_path, {}).effect("create", {}, adapter, deadline=time.time()+10)
    assert adapter.calls == 0


def test_transient_reads_and_deadline(tmp_path):
    now = [100]
    def wait(n):
        now[0] += n
    adapter = Effect()
    adapter.read_failures = 2
    with exclusive(tmp_path):
        j = Journal(tmp_path, {})
        j.effect("create", {}, adapter, deadline=110, clock=lambda: now[0], sleep=wait)
        assert adapter.calls == 1 and now[0] == 103
        adapter.read_failures = 99
        with pytest.raises(Pending, match="window exhausted"):
            j.effect("create", {}, adapter, deadline=110, clock=lambda: now[0], sleep=wait)
    assert adapter.calls == 1 and now[0] < 110


def test_cumulative_billing_is_not_added_twice():
    caps = [{"id": "synthetic-a", "ceiling": "1.20"}, {"id": "synthetic-b", "ceiling": "0.30"}]
    rows = [{"id": "synthetic-a", "cumulative": "0.70"}]
    first = reconcile_billing(caps, {}, rows)
    second = reconcile_billing(caps, first["by_resource"], rows)
    assert first == second
    assert first["posted"] == "0.70" and first["reserved"] == "0.80"
    for bad in ([{"id": "other", "cumulative": "0.1"}], rows+rows,
                [{"id": "synthetic-a", "cumulative": "1.21"}],
                [{"id": "synthetic-a", "cumulative": "0.69"}]):
        with pytest.raises(EvidenceError):
            reconcile_billing(caps, first["by_resource"], bad)


def launch(script, *args):
    # CI uses its native project interpreter. Local callers may supply the
    # installed numeric scheduler as an argv prefix, preserving this interpreter.
    runner = os.environ.get("OVL_TEST_NUMERIC_RUNNER")
    cmd = [runner, "run", "--python", sys.executable] if runner else [sys.executable]
    return subprocess.run(cmd + [str(script), *map(str, args)], capture_output=True, text=True)


@pytest.mark.parametrize("seam", ["prepare:built", "prepare:renamed", "initialize:result", "record:boundary",
                                  "record:result", "replay:renamed", "export:built", "download:renamed", "verify:result"])
def test_actual_process_crash_full_recovery(tmp_path, seam):
    child = tmp_path / "child.py"
    child.write_text('''from pathlib import Path
import os,sys
from ovl_pipeline.lifecycle_fixture import FixtureLifecycle
def event(name):
    if name == sys.argv[3]: os._exit(73)
FixtureLifecycle(Path(sys.argv[1]),Path(sys.argv[2]),event=event).run()
''')
    run = tmp_path / "run"
    first = launch(child, FIXTURE, run, seam)
    assert first.returncode == 73, first.stderr + first.stdout
    second = launch(child, FIXTURE, run, "no-crash")
    assert second.returncode == 0, second.stderr + second.stdout
    verifier = tmp_path / "verify.py"
    verifier.write_text('''from pathlib import Path
import sys
from ovl_pipeline.fixture import verify_fixture
from ovl_pipeline.lifecycle_fixture import FixtureLifecycle
t=FixtureLifecycle(Path(sys.argv[1]),Path(sys.argv[2]))
r=verify_fixture(t.object("download"),t.policy())
assert r["result"]=="PASS" and r["updates"]>0
assert r["production_acceptance"]=="NOT_RUN" and not r["independent_third_party"]
''')
    fresh = launch(verifier, FIXTURE, run)
    assert fresh.returncode == 0, fresh.stderr + fresh.stdout
    journal = read_json(run / "private/journal.json")
    assert [o["name"] for o in journal["operations"]] == ["prepare", "initialize", "record", "replay", "export", "download", "verify", "close"]
    assert all(o["status"] == "complete" for o in journal["operations"])


def test_lifecycle_identity_and_sidecar_isolation(tmp_path):
    task = FixtureLifecycle(FIXTURE, tmp_path / "run")
    task.run()
    before = snapshot(task.object("export"))
    (task.private / "transfer.log").write_text("synthetic transport observation")
    task.run()
    assert snapshot(task.object("export")) == before
    (task.object("export") / "sidecar.log").write_text("unlisted")
    with pytest.raises(EvidenceError, match="extra or missing"):
        task.run()


def test_completed_lifecycle_rejects_policy_change_and_partial_export(tmp_path):
    task = FixtureLifecycle(FIXTURE, tmp_path / "run")
    task.run()
    original = read_json(task.policy_path)
    write_json(task.policy_path, {**original, "run_public_key_hex": "00"*32})
    with pytest.raises(EvidenceError):
        task.run()
    write_json(task.policy_path, original)
    (task.object("export") / "chat/model.safetensors").unlink()
    with pytest.raises(EvidenceError, match="missing"):
        task.run()


def test_single_owner_lock_and_changed_request(tmp_path):
    with exclusive(tmp_path):
        with pytest.raises(Pending, match="owner"):
            with exclusive(tmp_path):
                pass
        j = Journal(tmp_path, {})
        j.operation("one", {"input": 1})
        with pytest.raises(EvidenceError, match="input changed"):
            j.operation("one", {"input": 2})


def test_observation_must_not_admit_submission_after_deadline(tmp_path):
    now = [100]
    class Slow(Effect):
        def observe(self, operation, request):
            now[0] = 111
            return Observation("absent")
    adapter = Slow()
    with exclusive(tmp_path):
        with pytest.raises(Pending):
            Journal(tmp_path, {}).effect("create", {}, adapter, deadline=110, clock=lambda: now[0])
    assert adapter.calls == 0


def test_observed_running_is_durable_acceptance(tmp_path):
    class Disappears(Effect):
        reads = 0
        def observe(self, operation, request):
            self.reads += 1
            return Observation("running" if self.reads == 1 else "absent")
    adapter = Disappears()
    with exclusive(tmp_path):
        with pytest.raises(Pending):
            Journal(tmp_path, {}).effect("create", {}, adapter, deadline=time.time()+5, sleep=lambda _: None)
    assert adapter.calls == 0
    assert read_json(tmp_path / "journal.json")["operations"][0]["status"] == "sent"


def test_missing_journal_cannot_adopt_changed_sources(tmp_path):
    source = tmp_path / "inputs"
    shutil.copytree(FIXTURE, source)
    run = tmp_path / "run"
    FixtureLifecycle(source, run).run()
    (run / "private/journal.json").rename(run / "private/retained-journal.json")
    p = source / "conversations.json"
    p.write_text(p.read_text().replace("Alpha", "Gamma"))
    with pytest.raises(EvidenceError, match="journal"):
        FixtureLifecycle(source, run).run()


@pytest.mark.parametrize("chain_write", [1, 3])
def test_crash_inside_atomic_chain_write(tmp_path, chain_write):
    child = tmp_path / "chain-crash.py"
    child.write_text('''from pathlib import Path
import os,sys
from ovl_pipeline.lifecycle_fixture import FixtureLifecycle
original=os.replace
counter=0
def replace(src,dst,*a,**kw):
 global counter
 if str(dst).endswith("objects/record/chain.json"):
  counter+=1
  if counter==int(sys.argv[3]): os._exit(75)
 return original(src,dst,*a,**kw)
os.replace=replace
FixtureLifecycle(Path(sys.argv[1]),Path(sys.argv[2])).run()
''')
    run = tmp_path / "run"
    first = launch(child, FIXTURE, run, chain_write)
    assert first.returncode == 75, first.stderr
    second = launch(child, FIXTURE, run, 9999)
    assert second.returncode == 0, second.stderr
    assert list((run / "private/recovery").rglob(".pending-*"))
    assert not list((run / "objects/download").rglob(".pending-*"))
