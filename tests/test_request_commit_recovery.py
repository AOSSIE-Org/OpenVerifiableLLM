"""Real local Git recovery with installed hooks; no external publication."""
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from publish_progress_boundary import command, recover_request_commit
from ovl_pipeline.canonical import EvidenceError, file_hash, write_json


def fixture(tmp_path):
    clone=tmp_path/'checkout';clone.mkdir()
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=clone,stderr=subprocess.DEVNULL).decode().strip()
    git('init','-b','synthetic-publication')
    (clone/'README').write_text('Synthetic publication recovery fixture.\n')
    git('add','README')
    git('-c','user.name=Test','-c','user.email=test@example.org','commit','-m','Initialize synthetic publication')
    return clone,git('rev-parse','HEAD'),{'schema':'synthetic-public-request.v1','digest':'a'*64},git


@pytest.mark.parametrize('seam',['staged','before-commit','after-commit'])
def test_interrupted_request_commit_adopts_original_bytes_once(tmp_path,seam):
    clone,parent,request,git=fixture(tmp_path);name='requests/synthetic.json';interrupted=[];commits=[]
    def execute(args,**kwargs):
        committing='commit' in args
        if committing and seam=='before-commit' and not interrupted:
            interrupted.append(1);raise InterruptedError('synthetic pre-commit interruption')
        result=command(args,**kwargs)
        if committing:commits.append(git('rev-parse','HEAD'))
        if not interrupted and ((seam=='staged' and args[:2]==['git','add'])
                                or (seam=='after-commit' and committing)):
            interrupted.append(1);raise InterruptedError('synthetic lost local reply')
        return result
    with pytest.raises(InterruptedError):
        recover_request_commit(request,clone,name,parent,'Commit synthetic request',execute=execute)
    revision=recover_request_commit(request,clone,name,parent,'Commit synthetic request',execute=execute)
    assert recover_request_commit(request,clone,name,parent,'Commit synthetic request',execute=execute)==revision
    assert len(commits)==1 and git('rev-list','--count','HEAD')=='2'
    assert git('status','--porcelain')==''


@pytest.mark.parametrize('damage',['staged','working','unrelated-staged','untracked','extra-committed'])
def test_request_recovery_preserves_conflicting_content(tmp_path,damage):
    clone,parent,request,git=fixture(tmp_path);name='requests/synthetic.json';target=clone/name
    write_json(target,request);git('add',name)
    if damage=='staged':
        write_json(target,{'different':'synthetic bytes'});git('add',name);write_json(target,request)
    elif damage=='working':write_json(target,{'different':'synthetic bytes'})
    else:
        (clone/'unrelated.txt').write_text('Synthetic unrelated edit.\n')
        if damage!='untracked':git('add','unrelated.txt')
        if damage=='extra-committed':
            git('-c','user.name=Test','-c','user.email=test@example.org','commit','-m','Synthetic conflicting commit')
    original_head=git('rev-parse','HEAD');original_index=git('write-tree');original_bytes=file_hash(target)
    with pytest.raises(EvidenceError):
        recover_request_commit(request,clone,name,parent,'Commit synthetic request')
    assert git('rev-parse','HEAD')==original_head and git('write-tree')==original_index
    assert file_hash(target)==original_bytes
