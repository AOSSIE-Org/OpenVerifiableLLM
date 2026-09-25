"""The external verifier must not import the target numerical installation."""
from pathlib import Path
import os
import subprocess
import sys


def test_process_selection_does_not_import_numerical_packages(tmp_path):
    source=Path(__file__).resolve().parents[1]
    script=tmp_path/'verifier.py'
    script.write_text('''import importlib.abc,sys,time
from pathlib import Path
class RejectNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','nacl','model'}:
   raise RuntimeError('numerical import before external audit: '+fullname)
sys.meta_path.insert(0,RejectNumerical())
from ovl_pipeline.canonical import digest,file_hash
from ovl_pipeline.lifecycle_process import validate
source=Path(sys.argv[1]);input_file=source/'src/model.py'
base=source/'src/ovl_pipeline'
root=digest([{'path':'ovl_pipeline/'+p.name,'sha256':file_hash(p)} for p in sorted(base.glob('*.py'))]
            +[{'path':'model.py','sha256':file_hash(input_file)}])
validate({'schema':'ovl.lifecycle-process.v1','module':'ovl_pipeline.lifecycle_fixture',
          'arguments':['run'],'source_root':str(source),'source_sha256':root,
          'deadline':int(time.time())+60,'output':str(Path(sys.argv[2])/'result'),
          'inputs':[{'path':str(input_file),'bytes':input_file.stat().st_size,'sha256':file_hash(input_file)}]})
assert not {'torch','numpy','nacl','model'} & set(sys.modules)
''')
    result=subprocess.run([sys.executable,str(script),str(source),str(tmp_path)],
                          env={**os.environ,'PYTHONPATH':str(source/'src')},capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
