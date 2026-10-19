"""Every publishing entry point refuses before side effects while publication is paused."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
import publish_evidence_archive
import publish_models
import publish_production_registration
import publish_progress_boundary
import publish_raw_archive
import publish
from ovl_pipeline.canonical import EvidenceError


@pytest.fixture
def paused(tmp_path,monkeypatch):
    monkeypatch.setattr(Path,'home',lambda:tmp_path/'home')
    marker=tmp_path/'home/.local/state/openverifiablellm/publication-paused.json'
    marker.parent.mkdir(parents=True);marker.write_text('{}')


def entry_points(t):
    out=t/'out'
    return {
        'evidence upload':lambda:publish_evidence_archive.upload(t/'plan.json',t/'stage',out),
        'model commit':lambda:publish_models.commit_pending(out,None),
        'model publish':lambda:publish_models.publish('base',None,None,None,None,out),
        'registration commit':lambda:publish_production_registration.request_commit(None,None,out),
        'registration publish':lambda:publish_production_registration.publish(None,None,None,None,out,1),
        'progress commit':lambda:publish_progress_boundary.request_commit(None,None,out),
        'progress publish':lambda:publish_progress_boundary.publish(None,None,None,None,None,None,None,None,None,out,1),
        'raw upload':lambda:publish_raw_archive.upload(t/'plan.json',t/'stage',out),
        'huggingface upload':lambda:publish.publish_huggingface('owner/repo',str(t/'model')),
    }


@pytest.mark.parametrize('name',sorted(entry_points(Path('.'))))
def test_paused_entry_point_refuses_before_side_effects(paused,tmp_path,name):
    with pytest.raises(EvidenceError,match='Publication paused'):entry_points(tmp_path)[name]()
    assert not (tmp_path/'out').exists()


def test_huggingface_dry_run_stays_available_while_paused(paused,tmp_path,capsys):
    assert publish.publish_huggingface('owner/repo',str(tmp_path),dry_run=True)==0
    assert 'owner/repo' in capsys.readouterr().out
