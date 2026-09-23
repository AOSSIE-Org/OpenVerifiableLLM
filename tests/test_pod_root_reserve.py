from types import SimpleNamespace

import pytest

from ovl_pipeline.canonical import read_json,write_json
from test_pod_job_worker import fixture,m


@pytest.mark.parametrize('running',[False,True])
def test_root_reserve_is_independent_of_large_workspace(tmp_path,monkeypatch,running):
    job,_,root,worker=fixture(tmp_path,'import time\ntime.sleep(30)\n')
    launch=job/'launch';launch.mkdir()
    write_json(launch/'intent.json',{'schema':'ovl.pod-job-launch-intent.v1','job_sha256':root,'worker_sha256':worker})
    reads=[]
    def disk(path):
        if str(path)=='/':
            reads.append(path)
            return SimpleNamespace(free=(8 if running and len(reads)==1 else 3)*1024**3)
        return SimpleNamespace(free=128*1024**3)
    monkeypatch.setattr(m.shutil,'disk_usage',disk)
    if running:
        result=m.run(job,root,worker)
        assert result['exit_code']<0
        assert read_json(job/'status.json')['stop_reason']=='storage-bound'
        assert len(reads)>=2
    else:
        with pytest.raises(m.Refusal,match='pod root reserve'):m.run(job,root,worker)
        assert not(launch/'child.json').exists()
        assert read_json(job/'exit.json')['exit_code']==125
