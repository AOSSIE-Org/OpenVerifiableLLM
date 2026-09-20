"""Only wrapper sequencing/refusal; actual numerical CUDA evidence is separate."""
import sys,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import pod_tiny_cuda_probe as m
from ovl_pipeline.canonical import write_json,file_hash,read_json


@pytest.mark.parametrize('fault',[None,'measured-duration','faster-setup','record-fails','replay-fails','resume-mismatch','resume-short','resume-middle','wrong-resume','incomplete-replay','wrong-record-parent','launcher-tamper','monotonic-expired','missing-activity','wrong-schema'])
def test_bounded_fresh_process_probe_preserves_failures(tmp_path,fault,monkeypatch):
    for n in ('setup','config','recipe','kernel'):(tmp_path/n).write_text('explicit double')
    for n in ('inputs','runtime','stream'):(tmp_path/n).mkdir()
    output=tmp_path/'output';calls=[];setup_sha=file_hash(tmp_path/'setup')
    monkeypatch.setenv('OVL_ACTIVITY_FILE',str(output/'activity.json'))
    if fault=='missing-activity':monkeypatch.delenv('OVL_ACTIVITY_FILE')
    clock=[100.0]
    if fault in ('monotonic-expired','measured-duration'):monkeypatch.setattr(m.time,'monotonic',lambda:clock[0])
    if fault=='launcher-tamper':(tmp_path/'setup').write_bytes(b'altered')
    def execute(argv,**kwargs):
        calls.append(argv);assert argv[:3]==[sys.executable,'-I','-S'] and kwargs['check'] and 0<kwargs['timeout']<=240
        if fault=='measured-duration':clock[0]+=150
        assert 'OVL_ACTIVITY_FILE' not in kwargs['env']  # no mixing child numerical sequences
        if len(calls)>1:
            observed=read_json(output/'activity.json');assert len(observed['completed'])==len(calls)-1
        mode=argv[argv.index('--')+1];args=argv[argv.index('--')+1:];dest=Path(args[args.index('--output')+1]);dest.mkdir()
        if mode=='record':
            if fault=='record-fails':raise RuntimeError('actual subprocess would fail')
            write_json(dest/'record.json',{'schema':'future' if fault=='wrong-schema' else 'ovl.gpu-pilot-record.v1',
                'result':'RECORDED_NOT_REPLAYED','updates':8,'eligible_duration_for_forecast':False,'boundaries':[0,1,2]})
            if fault=='monotonic-expired':clock[0]+=721
        else:
            resume='--resume-from' in args
            if not resume and fault=='replay-fails':raise RuntimeError('actual subprocess would fail')
            value={'schema':'ovl.gpu-pilot-replay.v1','result':'PASS','scope':'training-resume-continuation-probe' if resume else 'fresh-initialization-continuous-pilot-replay',
                   'updates_recomputed':4 if resume else 8,'record_sha256':file_hash(output/'record/record.json'),
                   'initial_state_regenerated':True,'independent_third_party':False,'compared':[0,1,2],'resume_from':1 if resume else None}
            if fault=='resume-short' and resume:value['compared']=[2]
            if fault=='resume-middle' and resume:value['compared'][1]=99
            if fault=='wrong-resume' and resume:value['resume_from']=2
            if fault=='incomplete-replay' and not resume:value['updates_recomputed']=7
            if fault=='wrong-record-parent':value['record_sha256']='0'*64
            if fault=='resume-mismatch' and resume:value['compared'][-1]=3
            write_json(dest/'verification.json',value)
    args=[tmp_path/'setup',setup_sha,tmp_path/'config',file_hash(tmp_path/'config'),tmp_path/'inputs',tmp_path/'runtime',tmp_path/'stream',tmp_path/'recipe',tmp_path/'kernel',output,int(time.time())+500]
    if fault=='faster-setup':args[-1]=int(time.time())+1020
    if fault not in (None,'measured-duration','faster-setup'):
        with pytest.raises((ValueError,RuntimeError,TimeoutError)):m.probe(*args,execute=execute)
        assert not(output/'probe.json').exists()
        if fault=='record-fails':assert len(calls)==1
        if fault=='replay-fails':assert len(calls)==2
        if fault in ('launcher-tamper','missing-activity'):assert not calls
        if fault in ('incomplete-replay','wrong-record-parent'):assert len(calls)==2
        if fault=='monotonic-expired':assert len(calls)==1 and (output/'record/record.json').exists()
    else:
        result=m.probe(*args,execute=execute);assert result['throughput_forecast']=='NOT_RUN' and len(calls)==3
        assert [p['phase'] for p in read_json(output/'activity.json')['completed']]==['record','replay','resume']
