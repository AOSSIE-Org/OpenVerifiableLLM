"""Bounded full-stream preflight liveness for selected development pilots only.

The dispatcher supplies exact stream roots/counts from its independently pinned
plan. These observations authorize neither training nor durable export credit.
The existing rental clock, useful-progress and export-age limits are unchanged.
"""
import re

from ovl_pipeline.canonical import EvidenceError,digest,require_digest
from ovl_pipeline.schema import fields,integer
from workload_health import Health
from inventory_health import InventoryHealth


class PilotHealth(Health):
    def __init__(self,journal,watchdog_intent,pod_id,bindings,**kwargs):
        self.bindings=bindings
        self.validations={}
        self.inventory=InventoryHealth(self)
        super().__init__(journal,watchdog_intent,pod_id,**kwargs)

    def contract(self,job):
        require_digest(job)
        if job not in self.bindings:raise EvidenceError('pilot lacks selected stream binding')
        b=self.bindings[job]
        fields(b,'schema stream_sha256 documents','pilot validation binding')
        if b['schema']!='ovl.pilot-validation-binding.v1':raise EvidenceError('wrong pilot validation binding')
        require_digest(b['stream_sha256']);integer(b['documents'],1,2**53-1,'bound stream documents')
        return b

    def start_job(self,selection):
        if selection.get('kind')!='pilot':return super().start_job(selection)
        fields(selection,'schema job_sha256 pod_id kind','selected pilot job')
        job=selection['job_sha256'];contract=self.contract(job)
        if selection['schema']!='ovl.selected-workload-job.v1' or selection['pod_id']!=self.pod:
            raise EvidenceError('selected pilot identity differs')
        selected={**selection,'validation_contract_sha256':digest(contract)}
        if job in self.jobs:
            if self.jobs[job]['selection']!=selected:raise EvidenceError('retained pilot contract changed')
            return False
        if any(not x['finished'] for x in self.jobs.values()):raise EvidenceError('another workload stage is active')
        self.event('job-start',selected);return True

    def _numeric_transition(self,job,observation):
        self.inventory.transition(job,observation)
        if self.jobs[job]['selection'].get('validation_contract_sha256')!=digest(self.contract(job)):
            raise EvidenceError('selected validation contract changed')
        previous=self.validations.get(job)
        if previous and (any(previous[k]!=observation[k] for k in ('process_instance','pid'))
                         or observation['sequence']<=previous['sequence']):
            raise EvidenceError('numerical activity changed validation process or sequence')
        # Polling may miss the final preflight snapshot. Source code still gates
        # all updates on successful complete validation; telemetry is not proof.

    def _validation(self,job,value):
        self.inventory.transition(job,value)
        self.active(job)
        b=self.contract(job)
        if self.jobs[job]['selection'].get('validation_contract_sha256')!=digest(b):
            raise EvidenceError('selected validation contract changed')
        if self.jobs[job]['selection']['kind']!='pilot' or job in self.activities or job in self.phase_activities:
            raise EvidenceError('validation activity after numerical work or in another protocol')
        fields(value,'schema process_instance pid sequence stream_sha256 documents completed_documents complete scope','stream validation activity')
        if (value['schema']!='ovl.runtime-stream-validation.v1'
            or value['scope']!='operator-supervision-only-not-training-verification'
            or type(value['process_instance']) is not str or not re.fullmatch('[0-9a-f]{32}',value['process_instance'])):
            raise EvidenceError('unsupported stream validation activity')
        integer(value['pid'],1,2**31-1,'validation PID');integer(value['sequence'],1,2**53-1,'validation sequence')
        integer(value['documents'],1,2**53-1,'validation document total')
        integer(value['completed_documents'],0,b['documents'],'validated prefix')
        if value['stream_sha256']!=b['stream_sha256'] or value['documents']!=b['documents']:
            raise EvidenceError('validation stream differs from selected job')
        if type(value['complete']) is not bool or value['complete'] and value['completed_documents']!=b['documents']:
            raise EvidenceError('invalid validation completion')
        previous=self.validations.get(job)
        if previous:
            if any(previous[k]!=value[k] for k in ('process_instance','pid')):
                raise EvidenceError('validation process changed')
            if value['sequence']<previous['sequence'] or value['completed_documents']<previous['completed_documents']:
                raise EvidenceError('validation sequence or prefix regressed')
            if value['sequence']==previous['sequence'] and value!=previous:
                raise EvidenceError('same validation sequence changed')
            if previous['complete'] and value!=previous:raise EvidenceError('completed validation changed')
        return (value['completed_documents']>0 if previous is None else
                value['completed_documents']>previous['completed_documents'] or (value['complete'] and not previous['complete']))

    def _apply(self,body):
        if self.inventory.apply(body):return
        kind=body.get('kind');d=body.get('detail',{})
        if kind in ('activity','pilot-phases'):self.inventory.transition(d['job_sha256'],d['observation'])
        if kind=='job-start' and d.get('kind')=='pilot':
            if d.get('validation_contract_sha256')!=digest(self.contract(d['job_sha256'])):
                raise EvidenceError('retained pilot validation contract changed')
        if kind=='activity' and self.jobs[d['job_sha256']]['selection']['kind']=='pilot':
            self._numeric_transition(d['job_sha256'],d['observation'])
        if kind!='stream-validation':return super()._apply(body)
        fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','validation cost event')
        fields(d,'job_sha256 observation','validation cost detail')
        advances=self._validation(d['job_sha256'],d['observation'])
        integer(body['observed_epoch'],self.plan['input']['now_epoch'],self.plan['external_terminate_epoch'],'validation event clock')
        if (body['schema']!='ovl.cost-activity-event.v2' or body['advances_progress'] is not advances
            or body['advances_export'] is not False or body['completes'] is not False):
            raise EvidenceError('invalid validation cost decision')
        self.validations[d['job_sha256']]=d['observation']
        if advances:self.progress=max(self.progress,body['observed_epoch'])

    def activity(self,job,observation):
        self.active(job)
        if self.jobs[job]['selection']['kind']!='pilot':return super().activity(job,observation)
        if type(observation) is dict and observation.get('schema')=='ovl.runtime-inventory-read.v1':
            return self.inventory.activity(job,observation)
        if type(observation) is dict and observation.get('schema')=='ovl.runtime-stream-validation.v1':
            advances=self._validation(job,observation)
            if self.validations.get(job)==observation:return False
            self.event('stream-validation',{'job_sha256':job,'observation':observation},progress=advances)
            return advances
        if type(observation) is not dict or observation.get('schema')!='ovl.runtime-activity.v1':
            raise EvidenceError('selected pilot requires its numerical process activity')
        self._numeric_transition(job,observation)
        return super().activity(job,observation)

    def inventory_contract(self,job):
        b=self.contract(job)
        if (self.jobs[job]['selection']['kind']!='pilot'
                or self.jobs[job]['selection'].get('validation_contract_sha256')!=digest(b)):
            raise EvidenceError('inventory selected pilot contract changed')
        return {(b['stream_sha256'],b['documents'])},1

    def _phase_activity(self,job,observation):
        if job in self.bindings:raise EvidenceError('selected numerical pilot forbids wrapper phase protocol')
        return super()._phase_activity(job,observation)
