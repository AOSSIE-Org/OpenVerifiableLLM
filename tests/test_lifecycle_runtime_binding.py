from pathlib import Path
import pytest
from ovl_pipeline.canonical import EvidenceError,write_json
from ovl_pipeline.lifecycle import exclusive
from ovl_pipeline.runtime_launch import launch
from test_lifecycle_process import audited_request
from ovl_pipeline.lifecycle_process import validate


def test_dependency_lock_and_allowance_bytes_must_be_pinned(tmp_path):
    request=audited_request(tmp_path)
    missing={**request,'inputs':request['inputs'][:-1]}
    with pytest.raises(EvidenceError,match='dependency selection'):validate(missing)
    lock=Path(request['inputs'][-1]['path']);lock.write_text('different legitimate package selection at the same path')
    with pytest.raises(EvidenceError,match='input bytes differ'):validate(request)
    request=audited_request(tmp_path)
    allowance=tmp_path/'allowed.json';write_json(allowance,{})
    request['arguments'][0:0]=['--allowed-generated',str(allowance)]
    with pytest.raises(EvidenceError,match='dependency selection'):validate(request)


def test_actual_runtime_selection_must_match_owned_request_before_auditing(tmp_path):
    request=audited_request(tmp_path);job=tmp_path/'job';job.mkdir()
    write_json(job/'request.json',request)
    args=request['arguments'];split=args.index('--');fields=dict(zip(args[:split:2],args[1:split:2]))
    with exclusive(job/'lease') as fd:
        with pytest.raises(EvidenceError,match='actual audited launch differs'):
            launch(Path(fields['--lock']),tmp_path/'different-wheels',Path(fields['--venv']),
                   Path(fields['--source']),Path(fields['--output']),fields['--module'],args[split+1:],
                   interpreter_archive=Path(fields['--interpreter-archive']),interpreter_sha256=fields['--interpreter-sha256'],
                   interpreter_root=Path(fields['--interpreter-root']),lease_fd=fd,lease_path=job/'lease/owner.lock',
                   lifecycle_request=job/'request.json',execute=lambda *a,**k:pytest.fail('must not launch'))
