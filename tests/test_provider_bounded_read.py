"""Real owned subprocess deadline/reaping; no provider or billable requests."""
import json
import os
from pathlib import Path
import time
from decimal import Decimal
import pytest
import provider_bounded_read as m
from probe_provider_deadline import ProviderFailure,diagnostic,transient_read_grace
from ovl_pipeline.canonical import EvidenceError


def call(execute,seconds=1,operation='account',variables=None):
    return m.call(operation,variables,execute,ProviderFailure,diagnostic,seconds=seconds)


def test_exact_financial_values_response_hash_and_clock_survive_process_boundary():
    # Artificial precision sentinel; not an observed account balance.
    result=call(lambda *a:({'myself':{'clientBalance':Decimal('123.456789012345678901'),'pods':[]}},'a'*64,
                          {'server_epoch':100,'request_started_epoch':99,'request_completed_epoch':101}))
    assert result==({'myself':{'clientBalance':Decimal('123.456789012345678901'),'pods':[]}},'a'*64,
                    {'server_epoch':100,'request_started_epoch':99,'request_completed_epoch':101})


def test_provider_dictionary_cannot_spoof_a_transport_type_tag():
    value={'kind':'decimal','value':'1.25','nested':[{'kind':'array','value':[]}]}
    assert call(lambda *a:(value,'a'*64,{}))[0]==value


def test_whole_call_deadline_kills_and_reaps_slow_owned_read(tmp_path):
    pidfile=tmp_path/'pid'
    def slow(*args):
        pidfile.write_text(str(os.getpid()));time.sleep(10)
        return {},'a'*64,{}
    started=time.monotonic()
    with pytest.raises(ProviderFailure) as error:call(slow,.15)
    elapsed=time.monotonic()-started
    assert error.value.transient and elapsed<1
    pid=int(pidfile.read_text());assert not Path(f'/proc/{pid}').exists()
    with pytest.raises(ChildProcessError):os.waitpid(pid,os.WNOHANG)
    assert transient_read_grace(error.value,28,0,28,0,100,False)
    assert transient_read_grace(error.value,50,0,50,0,200,False)
    assert not transient_read_grace(error.value,120,0,120,0,200,False)


@pytest.mark.parametrize('category,status,transient',[('transport',None,True),('http',503,True),('http',401,False),('invalid-response-Refused',None,False)])
def test_error_classification_unchanged_without_reflected_data(category,status,transient):
    def failed(*a):raise ProviderFailure(category,status=status,transient=transient)
    with pytest.raises(ProviderFailure) as error:call(failed)
    assert diagnostic(error.value)=={'error_type':'ProviderFailure','category':category,'http_status':status,'transient':transient}


def test_unexpected_exception_never_reflects_credential():
    def failed(*a):raise ValueError('private-test-token-never-public')
    with pytest.raises(ProviderFailure) as error:call(failed)
    assert not error.value.transient and 'private' not in str(error.value)


@pytest.mark.parametrize('operation,variables',[('create',None),('terminate',{}),('account',{'unexpected':'input'})])
def test_mutations_or_read_arguments_refused_before_child(operation,variables,tmp_path):
    marker=tmp_path/'called'
    with pytest.raises(EvidenceError):call(lambda *a:marker.touch(),operation=operation,variables=variables)
    assert not marker.exists()


def test_malformed_and_oversized_child_results_fail_closed():
    for result in [('not-a-three-item-response',),({'huge':'x'*(m.LIMIT+1)},'a'*64,{})]:
        with pytest.raises(ProviderFailure) as error:call(lambda *a:result)
        assert not error.value.transient
