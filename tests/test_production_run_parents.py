"""Parent selection helpers; synthetic report fixture supplies no GPU credit."""
from pathlib import Path
import sys
from copy import deepcopy
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import production_run_parents as m
from test_production_parents import parents
from ovl_pipeline.canonical import EvidenceError,digest,inventory,read_json,write_json
from ovl_pipeline.production_anchoring import PACKET_FILES


def selected(tmp_path):
    r,p=parents();bundle=tmp_path/'source.sigstore.json';write_json(bundle,{'explicit-test-signature-substitute':True})
    r['source_bundle_sha256']=digest(read_json(bundle))
    q={'pilot_records':p['pilot_records'],'pilot_replays':p['pilot_replays']}
    initial={'record':p['initial_record'],'verification':p['initial_verification']}
    return r,p,bundle,q,initial


def test_exact_packet_uses_both_qualified_pairs_and_regenerated_initialization(tmp_path):
    r,p,bundle,q,initial=selected(tmp_path)
    check=m.packet(r,p['source'],bundle,p['source_policy'],p['prepared'],q,initial,tmp_path/'packet')
    assert check['production_admission']=='NOT_RUN' and check['assertion_truth_established'] is False
    assert {p.name for p in (tmp_path/'packet').iterdir()}==PACKET_FILES
    assert read_json(tmp_path/'packet/registration.json')==r
    with pytest.raises(EvidenceError):m.packet(r,p['source'],bundle,p['source_policy'],p['prepared'],q,initial,tmp_path/'packet')


@pytest.mark.parametrize('damage',['runtime','recipe','initial-state','incomplete-replay','source-bundle','budget'])
def test_wrong_parents_cannot_create_a_registration_packet(tmp_path,damage):
    r,p,bundle,q,initial=selected(tmp_path)
    if damage=='runtime':initial['verification']['environment']={'compatible':{'changed':True}}
    elif damage=='recipe':q['pilot_records']['wikipedia']['settings']['recipe']['seed']+=1
    elif damage=='initial-state':initial['verification']['initial_state_sha256']='0'*64
    elif damage=='incomplete-replay':q['pilot_replays']['conversation']['updates_recomputed']-=1
    elif damage=='source-bundle':write_json(bundle,{'other':'bundle'})
    else:r['forecast_input']['spent_usd']='130'
    with pytest.raises(EvidenceError):m.packet(r,p['source'],bundle,p['source_policy'],p['prepared'],q,initial,tmp_path/'packet')
    assert not(tmp_path/'packet').exists()
