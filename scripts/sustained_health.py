"""Pinned public-input download liveness alongside single-process pilot health."""
import re

from ovl_pipeline.canonical import EvidenceError,digest,require_digest
from ovl_pipeline.schema import fields,integer
from pilot_health import PilotHealth


class SustainedHealth(PilotHealth):
    def __init__(self,journal,watchdog_intent,pod_id,bindings,download_bindings,**kwargs):
        self.download_bindings=download_bindings;self.download_processes={};self.setup_audits={}
        if set(bindings)&set(download_bindings):raise EvidenceError('job cannot mix download and numerical contracts')
        super().__init__(journal,watchdog_intent,pod_id,bindings,**kwargs)

    def download_contract(self,job):
        require_digest(job)
        if job not in self.download_bindings:raise EvidenceError('missing selected public input contract')
        b=self.download_bindings[job];fields(b,'schema plan_sha256 bytes','public input contract')
        if b['schema']!='ovl.public-input-binding.v1':raise EvidenceError('wrong public input contract')
        require_digest(b['plan_sha256']);integer(b['bytes'],1,64*1024**3,'selected public input bytes')
        return b

    def start_job(self,selection):
        job=selection.get('job_sha256')
        if job not in self.download_bindings:return super().start_job(selection)
        fields(selection,'schema job_sha256 pod_id kind','selected public input job')
        if selection['schema']!='ovl.selected-workload-job.v1' or selection['pod_id']!=self.pod or selection['kind']!='setup':
            raise EvidenceError('public download requires a distinct selected setup job')
        selected={**selection,'download_contract_sha256':digest(self.download_contract(job))}
        if job in self.jobs:
            if self.jobs[job]['selection']!=selected:raise EvidenceError('retained download contract changed')
            return False
        if any(not x['finished'] for x in self.jobs.values()):raise EvidenceError('another stage is active')
        self.event('job-start',selected);return True

    def _apply(self,body):
        kind=body.get('kind');d=body.get('detail',{})
        if kind=='job-start' and 'download_contract_sha256' in d:
            if d['kind']!='setup' or d['download_contract_sha256']!=digest(self.download_contract(d['job_sha256'])):
                raise EvidenceError('historical download contract changed')
        if kind=='setup-audit':
            fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','setup audit cost event')
            fields(d,'job_sha256 observation','setup audit detail')
            advances=self._setup_audit(d['job_sha256'],d['observation'])
            integer(body['observed_epoch'],self.plan['input']['now_epoch'],self.plan['external_terminate_epoch'],'setup audit clock')
            if (body['schema']!='ovl.cost-activity-event.v2' or body['advances_progress'] is not advances
                or body['advances_export'] is not False or body['completes'] is not False):raise EvidenceError('invalid setup audit cost decision')
            self.setup_audits[d['job_sha256']]=d['observation']
            if advances:self.progress=max(self.progress,body['observed_epoch'])
            return
        if kind!='public-download-process':return super()._apply(body)
        fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','public download process event')
        fields(d,'job_sha256 process_instance pid','selected public download process')
        self.active(d['job_sha256']);self.download_contract(d['job_sha256'])
        integer(body['observed_epoch'],self.plan['input']['now_epoch'],self.plan['external_terminate_epoch'],'download identity clock')
        if (body['schema']!='ovl.cost-activity-event.v2' or any(body[k] is not False for k in ('advances_progress','advances_export','completes'))
            or type(d['process_instance']) is not str or not re.fullmatch('[0-9a-f]{32}',d['process_instance'])):
            raise EvidenceError('invalid public download identity event')
        integer(d['pid'],1,2**31-1,'public download PID')
        if d['job_sha256'] in self.download_processes:raise EvidenceError('download process identity duplicated')
        self.download_processes[d['job_sha256']]=d

    def _setup_audit(self,job,value):
        self.active(job);binding=self.download_contract(job)
        if self.jobs[job]['selection'].get('download_contract_sha256')!=digest(binding):raise EvidenceError('selected setup contract changed')
        fields(value,'schema process_instance pid completed copied_bytes scope','setup audit activity')
        if (self.jobs[job]['selection']['kind']!='setup' or value['schema']!='ovl.runtime-setup-activity.v1'
            or value['scope']!='operator-supervision-only-not-input-verification'
            or type(value['process_instance']) is not str or not re.fullmatch('[0-9a-f]{32}',value['process_instance'])):
            raise EvidenceError('unsupported setup audit activity')
        integer(value['pid'],1,2**31-1,'setup audit PID');integer(value['copied_bytes'],0,min(binding['bytes'],4*1024**3),'setup copy bytes')
        phases=['source','selection','interpreter','installation','installed','launch-interpreter','launch-wheels','launch-installed','inspection']
        completed=value['completed']
        if type(completed) is not list or not 1<=len(completed)<=len(phases) or completed!=phases[:len(completed)]:
            raise EvidenceError('bounded ordered setup audit prefix required')
        if len(completed)>1 and value['copied_bytes']!=binding['bytes']:raise EvidenceError('setup selection requires complete bounded copy')
        previous=self.setup_audits.get(job)
        if previous:
            if any(value[k]!=previous[k] for k in ('process_instance','pid')):raise EvidenceError('setup audit process changed')
            if len(completed)<len(previous['completed']) or value['copied_bytes']<previous['copied_bytes']:
                raise EvidenceError('setup audit progress regressed')
            if len(previous['completed'])>1 and value['copied_bytes']!=previous['copied_bytes']:
                raise EvidenceError('setup copy changed after selection')
            return len(completed)>len(previous['completed']) or value['copied_bytes']-previous['copied_bytes']>=1024**2
        return True  # At least the complete source identity check has finished.

    def activity(self,job,value):
        if job not in self.download_bindings:return super().activity(job,value)
        self.active(job);b=self.download_contract(job)
        if type(value) is dict and value.get('schema')=='ovl.runtime-setup-activity.v1':
            advances=self._setup_audit(job,value)
            if self.setup_audits.get(job)==value:return False
            # Preserve the credited high water; tiny repeated reads cannot renew liveness.
            if not advances:return False
            self.event('setup-audit',{'job_sha256':job,'observation':value},progress=advances);return advances
        if job in self.setup_audits:raise EvidenceError('download activity after offline setup')
        if self.jobs[job]['selection'].get('download_contract_sha256')!=digest(b):raise EvidenceError('selected download contract changed')
        fields(value,'schema process_instance pid plan_sha256 total_bytes received_bytes scope','public download activity')
        if (value['schema']!='ovl.public-input-transfer.v1' or value['scope']!='operator-supervision-only-not-input-verification'
            or value['plan_sha256']!=b['plan_sha256'] or value['total_bytes']!=b['bytes']
            or type(value['process_instance']) is not str or not re.fullmatch('[0-9a-f]{32}',value['process_instance'])):
            raise EvidenceError('public transfer differs from selected input contract')
        integer(value['pid'],1,2**31-1,'download PID');integer(value['total_bytes'],1,64*1024**3,'download total')
        integer(value['received_bytes'],0,b['bytes'],'download received bytes')
        identity={'job_sha256':job,'process_instance':value['process_instance'],'pid':value['pid']}
        prior=self.download_processes.get(job)
        if prior is not None and prior!=identity:raise EvidenceError('public input process changed within selected job')
        operation=digest({'job_sha256':job,'public_input_plan_sha256':b['plan_sha256'],'operation':'actual-public-response-bytes'})
        if operation in self.transfers and value['received_bytes']<self.transfers[operation]['bytes_received']:
            raise EvidenceError('public input byte count regressed')
        if prior is None:self.event('public-download-process',identity)
        return self.bytes(operation,{'bytes_sent':0,'bytes_received':value['received_bytes']},total=b['bytes'])
