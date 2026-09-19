"""Recheck the separately extracted, selected complete closed-rental evidence.

Usage: .venv/bin/python THIS_SCRIPT EXTRACTED_COST_INPUTS FRESH_REPORT.json
Use the trusted repository revision containing this script and its named checker.
No provider calls, rental mutation or training verification is performed.
"""
from pathlib import Path
import importlib.util,sys
sys.path[:0]=['src','scripts']
from ovl_pipeline.canonical import EvidenceError,read_json,write_json,file_hash,digest
checker=Path('project/evidence/closed-rental-reconciliation-v1/reconcile.py')
if file_hash(checker)!='9b7196e9ab0a78f69bff5c288662311ab7d7a44e470373dcac5432a1c8186b05':raise EvidenceError('trusted checker source changed')
spec=importlib.util.spec_from_file_location('closed_charge',checker);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
value=m.check(Path(sys.argv[1]))
if value['pod_id']!='i3qz59mirj7969' or value['intent_sha256']!='ed12cae273a934f21c8b5789bad8760777b692d293d754311ec34249bf3f3e13':raise EvidenceError('closed rental identity differs')
output=Path(sys.argv[2])
if output.exists():raise EvidenceError('fresh verification output required')
write_json(output,value);print(digest(value))
