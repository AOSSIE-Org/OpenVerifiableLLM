from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from ovl_pipeline.canonical import EvidenceError,digest,write_json
import production_run_inputs as m


def selected_plan(directory,roots):
    stages=[]
    for index,root in enumerate(roots):
        job={'argv':['/usr/bin/python3','-I','-S','/inputs/pod_runtime_setup.py','setup','--runtime',root]}
        path=f'job-{index}.json';write_json(directory/path,job)
        stages.append({'template_path':path,'template_sha256':digest(job)})
    return {'stages':stages}


def test_runtime_placement_comes_from_all_exact_selected_job_bytes(tmp_path):
    plan=selected_plan(tmp_path,['/opt/ovllm-runtime','/opt/ovllm-runtime'])
    before=deepcopy(plan)
    assert m.qualified_runtime(plan,tmp_path)=='/opt/ovllm-runtime'
    assert plan==before
    write_json(tmp_path/'job-1.json',{'argv':['/usr/bin/python3','-I','-S','/inputs/pod_runtime_setup.py','setup','--runtime','/elsewhere']})
    with pytest.raises(EvidenceError,match='template changed'):m.qualified_runtime(plan,tmp_path)


@pytest.mark.parametrize('roots',[[],['/a','/b'],['/'],['relative'],['/a/../b'],['/a/'],['/a//b']])
def test_missing_ambiguous_or_noncanonical_runtime_is_rejected(tmp_path,roots):
    with pytest.raises(EvidenceError):m.qualified_runtime(selected_plan(tmp_path,roots),tmp_path)


@pytest.mark.parametrize('args',[
    ['/bin/python','--runtime'],
    ['/bin/python','--runtime','/a','--runtime','/a'],
    ['/bin/python','setup'],
])
def test_missing_or_repeated_runtime_option_is_not_inferred(tmp_path,args):
    job={'argv':['/usr/bin/python3','-I','-S','/inputs/pod_runtime_setup.py','launch',*args[1:]]}
    write_json(tmp_path/'job.json',job)
    with pytest.raises(EvidenceError):
        m.qualified_runtime({'stages':[{'template_path':'job.json','template_sha256':digest(job)}]},tmp_path)


@pytest.mark.parametrize('option',['--runtime=/other','--runt=/other','--r=/other'])
@pytest.mark.parametrize('canonical_too',[False,True])
def test_alternative_argparse_spelling_cannot_hide_conflicting_runtime(tmp_path,option,canonical_too):
    plan=selected_plan(tmp_path,['/opt/runtime','/other'])
    job={'argv':['/usr/bin/python3','-I','-S','/inputs/pod_runtime_setup.py','launch',option]}
    if canonical_too:job['argv'][5:5]=['--runtime','/opt/runtime']
    write_json(tmp_path/'job-1.json',job);plan['stages'][1]['template_sha256']=digest(job)
    with pytest.raises(EvidenceError,match='explicit runtime option'):m.qualified_runtime(plan,tmp_path)


def test_module_arguments_are_not_mistaken_for_launcher_runtime(tmp_path):
    plan=selected_plan(tmp_path,['/opt/runtime'])
    job={'argv':['/usr/bin/python3','-I','-S','/inputs/pod_runtime_setup.py','launch','--runtime','/opt/runtime','--','--runtime','/module-option']}
    write_json(tmp_path/'job-0.json',job);plan['stages'][0]['template_sha256']=digest(job)
    assert m.qualified_runtime(plan,tmp_path)=='/opt/runtime'


@pytest.mark.parametrize('script',['pod_public_setup.py','pod_runtime_setup.py',
                                   'pod_sustained_pilot.py','pod_initialization.py'])
def test_all_audited_workflow_entrypoints_retain_the_runtime(tmp_path,script):
    job={'argv':['/usr/bin/python3','-I','-S','/inputs/'+script,'--runtime','/opt/runtime']}
    write_json(tmp_path/'job.json',job)
    assert m.qualified_runtime({'stages':[{'template_path':'job.json','template_sha256':digest(job)}]},tmp_path)=='/opt/runtime'


def test_template_escape_and_symlink_are_rejected(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    job={'argv':['/bin/python','--runtime','/opt/runtime']};write_json(tmp_path/'outside.json',job)
    (inputs/'link.json').symlink_to(tmp_path/'outside.json')
    for name in ('../outside.json','link.json'):
        with pytest.raises(EvidenceError,match='outside selected inputs'):
            m.qualified_runtime({'stages':[{'template_path':name,'template_sha256':digest(job)}]},inputs)


@pytest.mark.parametrize('kind',['production-record','full-replay'])
def test_production_and_replay_use_qualified_local_runtime_with_full_parents(tmp_path,kind):
    from test_production_run_job import selected
    import production_run_job
    _,registration,packet,bundle,policy,source,_,_=selected(tmp_path,kind)
    config=tmp_path/'config.json';write_json(config,{'explicit-offline-config-fixture':True})
    pp=tmp_path/'policy.json';sp=tmp_path/'source-policy.json';write_json(pp,policy);write_json(sp,source)
    base='/selected';runtime='/opt/ovllm-runtime';profile={'remote_root':base}
    static=[{'path':base+'/prepared/'+p+'/stream.json','bytes':1,'sha256':registration['coverage'][p]['stream_sha256']} for p in registration['coverage']]
    executable={'path':runtime+'/public-python/python/bin/python3.12','bytes':7,'sha256':'5'*64}
    static.append(executable)
    kwargs={'record_files':[] if kind=='full-replay' else None,'runtime_root':runtime}
    job=m.production_job(kind,profile,config,static,packet,bundle,pp,sp,**kwargs)
    assert job['argv'][0]==executable['path']
    assert job['argv'][job['argv'].index('--runtime')+1]==runtime
    assert executable in job['required_files']
    job['deadline_epoch']=1000;job['argv']=['1000' if x==m.DEADLINE else x for x in job['argv']]
    remote=base+('-production-record' if kind=='production-record' else '-production-replay')
    assert production_run_job.command(job,registration,packet,bundle,policy,source,SimpleNamespace(profile={'remote_root':remote}),kind)['--output']==remote
    for damaged in (static[:-1],static+[executable]):
        with pytest.raises(EvidenceError,match='uniquely hash bound'):
            m.production_job(kind,profile,config,damaged,packet,bundle,pp,sp,**kwargs)
