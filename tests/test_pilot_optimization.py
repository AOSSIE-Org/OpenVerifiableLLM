"""Synthetic selection arithmetic and adversarial parents; no GPU credit."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pytest
from ovl_pipeline.canonical import EvidenceError,digest
from ovl_pipeline.phase_timing import Collector
from test_production_parents import parents
from pilot_optimization import screen,choose,policy


def fixture(*,candidate_ms=800000,numerical_percent=80):
    r,p=parents();baseline={'pilot_records':p['pilot_records'],'pilot_replays':p['pilot_replays'],'profiles':{}}
    counts=deepcopy(r['coverage'])
    for phase in counts:
        for mode,key in [('record','pilot_records'),('replay','pilot_replays')]:
            report=baseline[key][phase];report['setup_including_warmup_ms']=1000
            if mode=='replay':report['record_sha256']=digest(baseline['pilot_records'][phase])
            profiler=Collector();profiler.phase='measured'
            wall=report['measured_ms']*1000000
            profiler.add_wall('numerical_update',wall*numerical_percent//100)
            profiler.entries[('measured','numerical_update')]['calls']=100
            profiler.entries[('measured','numerical_update')]['cuda_calls']=100  # Explicit event-counter fixture, no actual GPU.
            profiler.add_wall('durable_delivery_wait',wall*(100-numerical_percent)//100)
            baseline['profiles'][phase+'-'+mode]=profiler.report(report,scope='operator-pilot-phase-timing')
        census=counts[phase]
        census.update(updates=10000,full_batch_updates=10000,target_bearing_windows=10000*census['batch_size'])
    candidate=deepcopy(baseline);candidate.pop('profiles')
    other=deepcopy(counts)
    for phase in counts:
        recorded=candidate['pilot_records'][phase];replayed=candidate['pilot_replays'][phase]
        recipe=deepcopy(recorded['settings']['recipe']);recipe['batch_size']*=2;recipe['boundary_every']//=2
        recorded['settings']['recipe']=recipe;recorded['measured_ms']=candidate_ms
        recorded['settings']['checkpoint_every']=(recorded['settings']['checkpoint_every']+1)//2
        replayed['measured_ms']=candidate_ms;replayed['record_sha256']=digest(recorded)
        other[phase].update(batch_size=recipe['batch_size'],recipe_sha256=digest(recipe),updates=5000,full_batch_updates=5000)
    candidate['prepared_qualification_sha256']=digest(baseline)
    selection={'schema':'ovl.single-batch-optimization.v1','minimum_gain_percent':5,'minimum_saved_seconds':900,
        'payback_multiplier':2,'coverage':{'baseline':counts,'candidate':other}}
    return baseline,candidate,selection


def test_choose_only_end_to_end_improvement_with_qualification_payback():
    baseline,candidate,p=fixture()
    assert screen(baseline,p)['decision']=='TRY_ONE_BATCH_DOUBLING'
    result=choose(baseline,candidate,p,3600000)
    assert result['selected']=='candidate'
    assert result['candidate_projected_training_replay_ms']<result['baseline_projected_training_replay_ms']
    # Substantial measured speedup is insufficient if acquiring it costs more.
    assert choose(baseline,candidate,p,7*86400000)['selected']=='baseline'
    baseline,candidate,p=fixture(candidate_ms=1200000)
    assert choose(baseline,candidate,p,100000)['selected']=='baseline'


def test_checkpoint_dominated_profile_does_not_speculate_on_compute():
    baseline,candidate,p=fixture(numerical_percent=20)
    result=screen(baseline,p)
    assert result['decision']=='KEEP_BASELINE' and result['largest_projected_wall_category']=='durable_delivery_wait'
    with pytest.raises(EvidenceError,match='bottleneck'):choose(baseline,candidate,p,1000)


@pytest.mark.parametrize('damage',['missing-profile','timing-parent','sampled-profile','overlapping-timing','sampled-replay','changed-corpus','changed-candidate-parent','extra-recipe-change'])
def test_missing_or_rehashed_conflicting_selection_fails_closed(damage):
    baseline,candidate,p=fixture()
    if damage=='missing-profile':baseline['profiles'].pop('wikipedia-record')
    elif damage=='timing-parent':baseline['profiles']['wikipedia-record']['parent_sha256']='0'*64
    elif damage=='sampled-profile':baseline['profiles']['wikipedia-record']['measurements'][1]['calls']=1
    elif damage=='overlapping-timing':baseline['profiles']['wikipedia-record']['measurements'][1]['wall_ns']*=10
    elif damage=='sampled-replay':candidate['pilot_replays']['wikipedia']['updates_recomputed']=1
    elif damage=='changed-corpus':p['coverage']['candidate']['wikipedia']['targets']+=1
    elif damage=='changed-candidate-parent':candidate['prepared_qualification_sha256']='0'*64
    else:
        candidate['pilot_records']['wikipedia']['settings']['recipe']['seed']+=1
    with pytest.raises(EvidenceError):choose(baseline,candidate,p,1000)


def test_coordinator_persists_one_decision_and_original_duration(tmp_path):
    from types import SimpleNamespace
    from production_run_coordinator import Run
    from ovl_pipeline.canonical import read_json,write_json
    baseline,candidate,p=fixture();now=[1000];calls=[]
    owner=SimpleNamespace(baseline=baseline,candidate=None,output=tmp_path,selection={'optimization_policy':p},health=SimpleNamespace(now=lambda:now[0]))
    def phase(name,*args):
        assert name=='optimization';calls.append(args);owner.candidate=candidate;now[0]=1100
        return candidate
    owner.phase=phase
    assert Run.optimize(owner,('plan','hash','inputs','output'))[0]=='candidate'
    original={f.name:f.read_bytes() for f in tmp_path.glob('*.json')}
    now[0]=10000
    assert Run.optimize(owner,('plan','hash','inputs','output'))[0]=='candidate'
    assert original=={f.name:f.read_bytes() for f in tmp_path.glob('*.json')}
    assert read_json(tmp_path/'optimization-decision.json')['actual_candidate_qualification_ms']==100000
    changed=read_json(tmp_path/'optimization-start.json');changed['screen_sha256']='0'*64
    write_json(tmp_path/'optimization-start.json',changed)
    with pytest.raises(EvidenceError,match='start selection'):Run.optimize(owner,('plan','hash','inputs','output'))


def test_coordinator_checkpoint_dominance_never_launches_candidate(tmp_path):
    from types import SimpleNamespace
    from production_run_coordinator import Run
    baseline,candidate,p=fixture(numerical_percent=20)
    owner=SimpleNamespace(baseline=baseline,candidate=None,output=tmp_path,selection={'optimization_policy':p},
        phase=lambda *args:pytest.fail('unjustified candidate launch'))
    assert Run.optimize(owner,())[0]=='baseline' and owner.qualified==baseline
    assert not (tmp_path/'optimization-start.json').exists()
