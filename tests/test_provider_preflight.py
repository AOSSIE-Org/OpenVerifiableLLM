import importlib.util
import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location("provider_preflight",Path(__file__).parents[1]/"scripts/provider_preflight.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


class Response:
    url=mod.ENDPOINT
    status=200
    headers={}
    def __init__(self,raw):self.raw=raw
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self,n):return self.raw[:n]


def test_fixed_query_uses_header_and_exact_tls_endpoint():
    def open(req,timeout):
        assert req.full_url==mod.ENDPOINT and timeout==30
        assert req.get_header("Authorization")=="Bearer test-secret"
        assert b"test-secret" not in req.data
        assert json.loads(req.data)=={"query":mod.QUERIES["schema"]}
        return Response(b'{"data":{"ok":true}}')
    data,h=mod.query("schema","test-secret",opener=SimpleNamespace(open=open))
    assert data=={"ok":True} and len(h)==64
    with pytest.raises(mod.Refused):mod.query("mutation","test-secret")


@pytest.mark.parametrize("raw",[b'{"data":null}',b'{"data":{},"errors":[{"message":"test-secret"}]}',
    b'{"data":{},"data":{}}',b'{"data":{"n":NaN}}',b'x'*(1024*1024+1)])
def test_invalid_or_error_responses_fail_without_reflecting_secrets(raw):
    with pytest.raises(mod.Refused) as e:
        mod.query("account","test-secret",opener=SimpleNamespace(open=lambda *a,**k:Response(raw)))
    assert 'test-secret' not in str(e.value)


def test_redirect_never_forwards_authorization():
    with pytest.raises(mod.Refused):
        mod.NoRedirect().redirect_request(None,None,302,'',{},'https://other.invalid')


@pytest.mark.parametrize("value",[True,None,-1,Decimal('NaN'),Decimal('Infinity'),Decimal('1e-1000000'),Decimal('1e1000000')])
def test_money_rejects_missing_nonfinite_and_allocation_attacks(value):
    with pytest.raises(mod.Refused):mod.amount(value)


def test_preflight_reports_observation_without_deadline_or_execution_credit(monkeypatch):
    schema={'pod':{'fields':[{'name':'id'},{'name':'desiredStatus'}]},
            'creation':{'inputFields':[{'name':'terminateAfter','type':{'name':'DateTime','kind':'SCALAR','ofType':None}}]}}
    account={'myself':{'clientBalance':Decimal('100.000001'),'currentSpendPerHr':Decimal('0.001092'),
                      'isAutoPayEnabled':False,'pods':[],'networkVolumes':[]}}
    monkeypatch.setattr(mod,'query',lambda op,key:({'schema':schema,'account':account}[op],'a'*64))
    r=mod.check('not-reported')
    assert r['account_balance_usd']=='100.000001'
    assert r['deadline_readback_fields_present']==[]
    assert r['execution_admission']==r['provider_deadline_behavior']=='NOT_RUN'
    assert r['project_spend_attribution']=='NOT_RUN'
    assert 'not-reported' not in json.dumps(r)
    def no_introspection(op,key):
        if op=='schema':raise mod.Refused('provider read failed: HTTP status 400')
        return account,'a'*64
    monkeypatch.setattr(mod,'query',no_introspection)
    r=mod.check('not-reported')
    assert r['schema_observation_status']=='UNAVAILABLE'
    assert r['pod_read_fields'] is r['deadline_readback_fields_present'] is None
    assert r['execution_admission']=='NOT_RUN' and r['account_balance_usd']=='100.000001'


def test_local_configuration_absent_empty_and_environment_precedence(tmp_path,monkeypatch):
    monkeypatch.delenv('RUNPOD_API_KEY',raising=False)
    monkeypatch.setattr(mod.Path,'home',lambda:tmp_path)
    with pytest.raises(mod.Refused):mod.credential()
    p=tmp_path/'.runpod';p.mkdir();(p/'config.toml').write_text('apikey=""\n')
    with pytest.raises(mod.Refused):mod.credential()
    (p/'config.toml').write_text('apikey="test-config-key"\n')
    (p/'config.toml').chmod(0o600)
    assert mod.credential()=='test-config-key'
    monkeypatch.setenv('RUNPOD_API_KEY','test-environment-key')
    assert mod.credential()=='test-environment-key'


def test_provider_precision_rounds_funds_down_costs_up_for_policy():
    from ovl_pipeline.budget import money
    assert mod.policy_amount('100.5079216339',balance=True)=='100.507921'
    assert mod.policy_amount('0.0000000001',balance=False)=='0.000001'
    assert mod.policy_amount(Decimal('0.24'),balance=False)=='0.240000'
    assert money(mod.policy_amount('100.5079216339',balance=True))==100507921


def test_publicly_readable_credentials_refused(tmp_path,monkeypatch):
    monkeypatch.delenv('RUNPOD_API_KEY',raising=False);monkeypatch.setattr(mod.Path,'home',lambda:tmp_path)
    p=tmp_path/'.runpod';p.mkdir();f=p/'config.toml';f.write_text('apikey="test-only"\n');f.chmod(0o644)
    with pytest.raises(mod.Refused):mod.credential()
