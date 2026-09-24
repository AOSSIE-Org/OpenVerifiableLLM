"""Synthetic pre-workload storage recovery; no provider or training credit."""
import pytest
from pathlib import Path
from test_workload_health import Clock,JOB,SELECTION
from test_external_watchdog import NOW,intent
from ovl_pipeline.canonical import digest,write_json,inventory,EvidenceError
from ovl_pipeline.supervision import Journal
from workload_health import Health


def fixture(tmp_path):
    clock=Clock();watchdog=intent()
    selection={'schema':'ovl.storage-recovery-retention.v1','pod_id':'owned-pod','watchdog_intent_sha256':digest(watchdog),
               'reports':['probe','inspected','consolidated'],'deadline_epoch':NOW+150}
    directory=tmp_path/'export';directory.mkdir();write_json(directory/'observed.json',{'synthetic':'retained observation'})
    files=inventory(directory,['observed.json'])
    return clock,watchdog,selection,directory,files


def health(j,c,w):return Health(j,w,'owned-pod',wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})


def test_retained_receipt_handoff_reuses_journal_without_workload_or_completion_credit(tmp_path):
    c,w,s,d,files=fixture(tmp_path);path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=health(j,c,w);c.advance(10)
        assert h.retained_preflight(s,'probe',d,files,deadline=s['deadline_epoch'])
        assert h.progress==h.exported==NOW+10 and not h.jobs and not h.complete
        c.advance(100);assert not h.retained_preflight(s,'probe',d,files,deadline=s['deadline_epoch'])
        assert h.exported==NOW+10
        with pytest.raises(EvidenceError,match='jobs'):h.finish(d,files)
    with Journal(path).lease() as j:
        h=health(j,c,w);assert h.exported==NOW+10 and not h.jobs
        h.start_job(SELECTION);assert set(h.jobs)=={JOB}
        with pytest.raises(EvidenceError,match='outside selected recovery'):
            h.retained_preflight(s,'inspected',d,files,deadline=s['deadline_epoch'])


@pytest.mark.parametrize('damage',['pod','intent','deadline','expired','unselected','extra-file','changed-hash','symlink','too-many'])
def test_changed_or_unbounded_preflight_evidence_cannot_renew_health(tmp_path,damage):
    c,w,s,d,files=fixture(tmp_path);name='probe';deadline=s['deadline_epoch']
    if damage=='pod':s['pod_id']='unrelated-pod'
    elif damage=='intent':s['watchdog_intent_sha256']='a'*64
    elif damage=='deadline':deadline+=1
    elif damage=='expired':c.advance(151)
    elif damage=='unselected':name='not-selected'
    elif damage=='extra-file':(d/'extra').write_bytes(b'synthetic')
    elif damage=='changed-hash':(d/'observed.json').write_bytes(b'changed')
    elif damage=='symlink':(d/'observed.json').unlink();(d/'observed.json').symlink_to('/not-a-real-source')
    else:s['reports']=[f'item-{n}' for n in range(65)]
    with Journal(tmp_path/'journal').lease() as j:
        h=health(j,c,w);before=len(j.events)
        with pytest.raises(EvidenceError):h.retained_preflight(s,name,d,files,deadline=deadline)
        assert len(j.events)==before and h.exported==NOW


def test_same_bytes_or_changed_selection_cannot_renew_export_age(tmp_path):
    c,w,s,d,files=fixture(tmp_path)
    with Journal(tmp_path/'journal').lease() as j:
        h=health(j,c,w);h.retained_preflight(s,'probe',d,files,deadline=s['deadline_epoch'])
        with pytest.raises(EvidenceError,match='repeated receipt'):
            h.retained_preflight(s,'inspected',d,files,deadline=s['deadline_epoch'])
        changed={**s,'deadline_epoch':s['deadline_epoch']+1}
        with pytest.raises(EvidenceError,match='selection changed'):
            h.retained_preflight(changed,'inspected',d,files,deadline=changed['deadline_epoch'])

@pytest.mark.parametrize('damage',['receipt','manifest','removed'])
def test_adoption_rehashes_preserved_recovery_evidence(tmp_path,damage):
    c,w,s,d,files=fixture(tmp_path);path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=health(j,c,w);h.retained_preflight(s,'probe',d,files,deadline=s['deadline_epoch'])
    if damage=='receipt':(d/'observed.json').write_bytes(b'changed synthetic receipt')
    elif damage=='removed':(d/'observed.json').unlink()
    else:next((path/'preflight-manifests').iterdir()).write_bytes(b'{}')
    with Journal(path).lease() as j:
        with pytest.raises((EvidenceError,FileNotFoundError)):health(j,c,w)


def test_production_health_adopts_preflight_without_resetting_clocks(tmp_path):
    from production_run_health import ProductionRunHealth
    c,w,s,d,files=fixture(tmp_path);path=tmp_path/'journal'
    with Journal(path).lease() as j:
        h=health(j,c,w);c.advance(10);h.retained_preflight(s,'probe',d,files,deadline=s['deadline_epoch'])
    c.advance(20)
    with Journal(path).lease() as j:
        h=ProductionRunHealth(j,w,'owned-pod',None,{}, {},wall=lambda:c.now,clock=lambda:{'boot_id':c.boot,'boottime_ms':c.ms})
        h.write(tmp_path/'health.json')
        assert h.progress==h.exported==NOW+10 and not h.jobs and not h.complete
        (d/'observed.json').write_bytes(b'changed after journal adoption')
        with pytest.raises(EvidenceError,match='mismatch|wrong-size'):h.start_job({**SELECTION,'kind':'setup'})
        assert not h.jobs


def test_unhashable_report_selection_is_rejected(tmp_path):
    c,w,s,d,files=fixture(tmp_path);s['reports']=[{}]
    with Journal(tmp_path/'journal').lease() as j:
        with pytest.raises(EvidenceError):health(j,c,w).retained_preflight(s,'probe',d,files,deadline=s['deadline_epoch'])
