"""Synthetic key selection must fail before any transport or publication."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pytest
from ovl_pipeline.canonical import EvidenceError,digest,write_json,inventory
from ovl_pipeline.run_key import create
from production_run_inputs import validate_run_key
import run_production_lifetime as lifetime


def selected(tmp_path):
    directory=tmp_path/'private';public=create(directory,'synthetic-admission')
    template={'run_id':public['run_id'],'run_public_key':public['public_key']}
    return directory,public,template


def test_descriptor_digest_is_rejected_without_changing_key(tmp_path):
    directory,public,template=selected(tmp_path)
    before={p.name:p.read_bytes() for p in directory.iterdir()}
    assert validate_run_key(directory,[template,deepcopy(template)])==public
    with pytest.raises(EvidenceError,match='external identity pin'):
        validate_run_key(directory,[{**template,'run_public_key':digest(public)}])
    assert {p.name:p.read_bytes() for p in directory.iterdir()}==before


def test_unselected_variant_and_changed_seed_are_not_silently_adopted(tmp_path):
    directory,public,template=selected(tmp_path)
    other=create(tmp_path/'other','synthetic-admission')
    with pytest.raises(EvidenceError,match='different run keys'):
        validate_run_key(directory,[template,{**template,'run_public_key':other['public_key']}])
    (directory/'seed.key').write_bytes((tmp_path/'other/seed.key').read_bytes())
    with pytest.raises(EvidenceError,match='private run key differs'):
        validate_run_key(directory,[template])
    with pytest.raises(EvidenceError,match='at least one'):
        validate_run_key(directory,[])


@pytest.mark.parametrize('optimized',[False,True])
def test_lifetime_rejects_wrong_pin_before_transport_or_phase_work(tmp_path,monkeypatch,optimized):
    directory,public,template=selected(tmp_path)
    path=tmp_path/'registration.json';write_json(path,{**template,'run_public_key':digest(public)})
    names='rental controller watchdog profile key known_hosts worker output health qualification initialization source_statement source_bundle source_policy preparation source_checkout static_files'.split()
    spec={name:str(tmp_path/name) for name in names}
    spec.update(schema='ovl.production-lifetime-invocation.v2' if optimized else 'ovl.production-lifetime-invocation.v1',
                selection={'schema':'ovl.production-run-selection.v2' if optimized else 'ovl.production-run-selection.v1'},
                registration_template=str(path),run_key=str(directory),input_inventory={'directory':str(tmp_path),'files':inventory(tmp_path,[path.name])})
    if optimized:
        good=tmp_path/'good.json';write_json(good,template)
        spec.update(initialization={'baseline':{},'candidate':{}},optimization={},registration_template={'baseline':str(good),'candidate':str(path)})
        spec['input_inventory']['files']=inventory(tmp_path,[path.name,good.name])
    def unexpected(*args,**kwargs):raise AssertionError('transport reached before key validation')
    monkeypatch.setattr(lifetime,'transport',unexpected)
    with pytest.raises(EvidenceError,match='external identity pin|different run keys'):
        lifetime.run(spec,digest(spec))
    assert not Path(spec['output']).exists()


def test_registration_change_after_entry_is_rejected_before_publication(tmp_path,monkeypatch):
    from types import SimpleNamespace
    directory,public,template=selected(tmp_path)
    path=tmp_path/'template.json';write_json(path,template)
    profile=tmp_path/'profile.json';write_json(profile,{'remote_root':'/explicit-fixture'})
    rental=tmp_path/'rental.json';write_json(rental,{'explicit-fixture':True})
    output=tmp_path/'output';output.mkdir()
    names='controller watchdog key known_hosts worker health qualification initialization source_statement source_bundle source_policy preparation source_checkout static_files'.split()
    spec={name:str(tmp_path/name) for name in names}
    spec.update(schema='ovl.production-lifetime-invocation.v1',selection={'schema':'ovl.production-run-selection.v1','timing':{'registration_seconds':30}},
                rental=str(rental),profile=str(profile),output=str(output),registration_template=str(path),run_key=str(directory),
                input_inventory={'directory':str(tmp_path),'files':inventory(tmp_path,[path.name,profile.name,rental.name])})
    class Owner:
        health=SimpleNamespace(complete=False,now=lambda:10)
        rental={}
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def phase(self,*args):return {'explicit-checked-parent':True}
        def register(self,*args):raise AssertionError('publication reached with changed registration key')
    monkeypatch.setattr(lifetime,'transport',lambda *args:None)
    monkeypatch.setattr(lifetime,'Run',Owner)
    monkeypatch.setattr(lifetime,'qualified_runtime',lambda *args:'/explicit-runtime')
    monkeypatch.setattr(lifetime,'qualified_volume',lambda *args:None)
    monkeypatch.setattr(lifetime,'selected_phases',lambda *args:{name:({},'0'*64,tmp_path,tmp_path) for name in ('qualification','initialization')})
    monkeypatch.setattr(lifetime,'registration',lambda *args,**kwargs:({**template,'run_public_key':digest(public)},{'explicit-fixture':True}))
    with pytest.raises(EvidenceError,match='external identity pin'):lifetime.run(spec,digest(spec))
    assert not (output/'registration.json').exists()


@pytest.mark.parametrize('optimized',[False,True])
def test_template_must_belong_to_exact_inventory(tmp_path,optimized):
    directory,public,template=selected(tmp_path)
    path=tmp_path/'template.json';write_json(path,template)
    unrelated=tmp_path/'unrelated.json';write_json(unrelated,{'explicit-fixture':True})
    spec={'schema':'ovl.production-lifetime-invocation.v2' if optimized else 'ovl.production-lifetime-invocation.v1',
          'registration_template':{'baseline':str(unrelated),'candidate':str(path)} if optimized else str(path),
          'input_inventory':{'directory':str(tmp_path),'files':inventory(tmp_path,[unrelated.name])}}
    with pytest.raises(EvidenceError,match='missing from pinned inventory'):lifetime.registration_templates(spec)


def test_empty_or_consumed_iterator_cannot_skip_key_load(tmp_path):
    with pytest.raises(EvidenceError,match='list or tuple'):
        validate_run_key(tmp_path/'missing',iter([]))


def test_coordinated_template_and_key_replacement_invalidates_original_pin(tmp_path):
    directory,public,template=selected(tmp_path)
    path=tmp_path/'template.json';write_json(path,template)
    spec={'schema':'ovl.production-lifetime-invocation.v1','registration_template':str(path),
          'input_inventory':{'directory':str(tmp_path),'files':inventory(tmp_path,[path.name])}}
    original=lifetime.registration_templates(spec);assert validate_run_key(directory,list(original.values()))==public
    other=create(tmp_path/'other','synthetic-admission')
    for name in ('seed.key','public.json'):(directory/name).write_bytes((tmp_path/'other'/name).read_bytes())
    write_json(path,{**template,'run_public_key':other['public_key']})
    with pytest.raises(EvidenceError,match='hash mismatch'):lifetime.registration_templates(spec)
    with pytest.raises(EvidenceError,match='external identity pin'):validate_run_key(directory,list(original.values()))
