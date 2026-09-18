"""Real file/hash measurement with explicitly substituted nvidia-smi only."""
from pathlib import Path
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_diagnostic as m


def test_diagnostic_preserves_complete_roundtrip_and_labels_scope(tmp_path):
    source=tmp_path/'payload'
    with source.open('wb') as f:f.truncate(64*1024**2)
    def smi(argv,**kwargs):
        assert argv[0]=='/usr/bin/nvidia-smi' and kwargs['timeout']==30
        return subprocess.CompletedProcess(argv,0,b'Fixture GPU, Fixture UUID, 000, 1, 00:00\n',b'')
    v=m.run(source,tmp_path/'output','Fixture GPU',execute=smi)
    assert v['hash_read_bytes']==1024**3 and v['hash_nanoseconds']>0
    assert (tmp_path/'output/roundtrip.bin').read_bytes()==source.read_bytes()
    assert v['production_admission']=='NOT_RUN'
    with pytest.raises(FileExistsError):m.run(source,tmp_path/'output','Fixture GPU',execute=smi)


def test_wrong_gpu_refuses_without_success_report(tmp_path):
    source=tmp_path/'payload'
    with source.open('wb') as f:f.truncate(64*1024**2)
    def smi(argv,**kwargs):return subprocess.CompletedProcess(argv,0,b'Other GPU, Fake UUID, 000, 1, 00:00\n',b'')
    with pytest.raises(ValueError,match='selected GPU'):m.run(source,tmp_path/'output','Selected GPU',execute=smi)
    assert not(tmp_path/'output/diagnostic.json').exists()
