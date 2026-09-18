"""State-aware cost health for dedicated production/replay dispatchers.

Callers must supply the independently authenticated registration and exact selected
jobs/transports. This class never authenticates publisher identity on their behalf
or authorizes an update. The numerical record/replay drivers retain those gates.
"""
from pathlib import Path
import re
from ovl_pipeline.canonical import EvidenceError,digest,read_json,require_digest,write_json
from ovl_pipeline.schema import fields,integer
from pod_checkpoint_handoff import state_check
from production_retention import verify as verify_retention
from publication_activity import validate as validate_publication
from workload_health import Health,terminal_status
from ovl_pipeline.production_observation import MAX_PASSES


class ProductionHealth(Health):
    def __init__(self,journal,watchdog_intent,pod_id,registration,bindings,**kwargs):
        self.registration=registration;self.registration_root=digest(registration)
        self.bindings=bindings;self.publications={};self.scans={}
        super().__init__(journal,watchdog_intent,pod_id,**kwargs)

    def contract(self,job):
        if digest(self.registration)!=self.registration_root:raise EvidenceError('selected production registration changed')
        require_digest(job)
        if job not in self.bindings:raise EvidenceError('production job lacks independently selected retention bindings')
        b=self.bindings[job];fields(b,'job_file worker_sha256 control transports','production retention binding')
        require_digest(b['worker_sha256']);value=read_json(Path(b['job_file']))
        if digest(value)!=job or value['kind'] not in ('production-record','full-replay'):raise EvidenceError('production job selection changed')
        from production_retention import roots
        selected=roots(b['control'],b['transports'],value,job)
        if b['control'].profile['pod_id']!=self.pod:raise EvidenceError('production retention selects another pod')
        return {'registration_sha256':self.registration_root,'job_sha256':job,'kind':value['kind'],
                'worker_sha256':b['worker_sha256'],
                'outputs':[{'root':root,'profile_sha256':digest(t.profile)} for t,_,root in selected]}

    def start_job(self,selection):
        if selection.get('kind') not in ('production-record','full-replay'):return super().start_job(selection)
        fields(selection,'schema job_sha256 pod_id kind','selected production job')
        job=selection['job_sha256'];contract=self.contract(job)
        if selection['schema']!='ovl.selected-workload-job.v1' or selection['pod_id']!=self.pod or selection['kind']!=contract['kind']:
            raise EvidenceError('selected production identity differs')
        selected={**selection,'retention_contract_sha256':digest(contract)}
        if job in self.jobs:
            if self.jobs[job]['selection']!=selected:raise EvidenceError('retained production contract changed')
            return False
        if any(not x['finished'] for x in self.jobs.values()):raise EvidenceError('another workload stage is active')
        self.event('job-start',selected);return True

    def _apply(self,body):
        kind=body.get('kind');detail=body.get('detail',{})
        if kind=='job-start' and detail.get('kind') in ('production-record','full-replay'):
            if detail.get('retention_contract_sha256')!=digest(self.contract(detail['job_sha256'])):
                raise EvidenceError('historical production retention contract differs')
        if kind=='production-scan':
            fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','production scan cost event')
            fields(detail,'job_sha256 observation','production scan cost detail')
            advances=self._scan(detail['job_sha256'],detail['observation'])
            integer(body['observed_epoch'],self.plan['input']['now_epoch'],self.plan['external_terminate_epoch'],'production scan clock')
            if (body['schema']!='ovl.cost-activity-event.v2' or body['advances_progress'] is not advances
                or body['advances_export'] is not False or body['completes'] is not False):
                raise EvidenceError('invalid production scan cost decision')
            self.scans[detail['job_sha256']]=detail['observation']
            if advances:self.progress=max(self.progress,body['observed_epoch'])
            return
        if kind=='activity' and detail['job_sha256'] in self.scans:
            self._scan_transition(detail['job_sha256'],detail['observation'])
        if body.get('kind')!='publication':return super()._apply(body)
        fields(body,'schema kind observed_epoch detail advances_progress advances_export completes','publication cost event')
        integer(body['observed_epoch'],self.plan['input']['now_epoch'],self.plan['external_terminate_epoch'],'publication event clock')
        if (body['schema']!='ovl.cost-activity-event.v2' or body['advances_progress'] is not True
            or body['advances_export'] is not False or body['completes'] is not False):raise EvidenceError('invalid publication liveness decision')
        d=body['detail'];fields(d,'job_sha256 observation','publication cost detail');value=d['observation']
        validate_publication(value,self.registration_root,value['boundary_sha256'],value['deadline_epoch'])
        key=(d['job_sha256'],value['boundary_sha256'],value['stage'])
        if key in self.publications:raise EvidenceError('duplicate durable publication credit')
        self.publications[key]=value;self.progress=max(self.progress,body['observed_epoch'])

    def _scan_transition(self,job,value):
        previous=self.scans.get(job)
        if previous and (any(value[k]!=previous[k] for k in ('process_instance','pid')) or value['sequence']<=previous['sequence']):
            raise EvidenceError('numerical activity changed production scan process/sequence')

    def _scan(self,job,value):
        self.active(job);contract=self.contract(job)
        if self.jobs[job]['selection'].get('retention_contract_sha256')!=digest(contract):
            raise EvidenceError('selected production contract changed')
        if job in self.activities or job in self.phase_activities:raise EvidenceError('production scan follows numerical activity')
        fields(value,'schema process_instance pid sequence pass_index operation stream_sha256 documents completed_documents complete scope','production scan observation')
        if (value['schema']!='ovl.runtime-production-scan.v1' or value['scope']!='operator-supervision-only-not-training-verification'
            or value['operation'] not in ('stream-validation','coverage-census','boundary-cursor-census')
            or type(value['process_instance']) is not str or not re.fullmatch('[0-9a-f]{32}',value['process_instance'])):
            raise EvidenceError('unsupported production scan observation')
        integer(value['pid'],1,2**31-1,'production scan PID');integer(value['sequence'],1,2**53-1,'production scan sequence')
        integer(value['pass_index'],1,MAX_PASSES,'bounded production scan pass')
        selected=[c for c in self.registration['coverage'].values() if c['stream_sha256']==value['stream_sha256']]
        if len(selected)!=1 or selected[0]['documents']!=value['documents']:raise EvidenceError('production scan differs from registered stream')
        integer(value['documents'],1,2**53-1,'production scan document total')
        integer(value['completed_documents'],0,value['documents'],'production scanned prefix')
        if type(value['complete']) is not bool or value['complete'] and value['completed_documents']!=value['documents']:
            raise EvidenceError('production scan completion differs')
        previous=self.scans.get(job)
        if previous:
            if any(value[k]!=previous[k] for k in ('process_instance','pid')):raise EvidenceError('production scan process changed')
            if value['sequence']<previous['sequence'] or value['pass_index']<previous['pass_index']:
                raise EvidenceError('production scan sequence/pass regressed')
            if value['sequence']==previous['sequence'] and value!=previous:raise EvidenceError('same production scan sequence changed')
            if value['pass_index']==previous['pass_index']:
                if any(value[k]!=previous[k] for k in ('operation','stream_sha256','documents')):
                    raise EvidenceError('production scan contract changed within pass')
                if value['completed_documents']<previous['completed_documents']:raise EvidenceError('production scan prefix regressed')
                if previous['complete'] and value!=previous:raise EvidenceError('completed production scan changed')
                return value['completed_documents']>previous['completed_documents'] or value['complete'] and not previous['complete']
        # A new pass alone is not progress. An actual checked prefix is required;
        # polling may miss earlier/final snapshots of a fast nested scan.
        return value['completed_documents']>0

    def activity(self,job,value):
        self.active(job)
        if type(value) is dict and value.get('schema')=='ovl.runtime-production-scan.v1':
            advances=self._scan(job,value)
            if self.scans.get(job)==value:return False
            self.event('production-scan',{'job_sha256':job,'observation':value},progress=advances)
            return advances
        if job in self.scans:self._scan_transition(job,value)
        return super().activity(job,value)

    def publication(self,job,snapshot,deadline,value):
        self.active(job);contract=self.contract(job)
        if contract['kind']!='production-record':raise EvidenceError('replay does not publish a new recorded boundary')
        integer(deadline,1,self.plan['provider_terminate_epoch'],'selected publication deadline')
        if self.now()>=deadline:raise EvidenceError('publication deadline reached')
        validate_publication(value,self.registration_root,value['boundary_sha256'],deadline)
        key=(job,value['boundary_sha256'],value['stage'])
        prior=self.publications.get(key)
        if prior is not None:
            if prior!=value:raise EvidenceError('publication identity changed after credited stage')
            return False
        chain,body,waiting,checked=state_check(self.registration,self.registration_root,Path(snapshot))
        validate_publication(value,self.registration_root,waiting['boundary_sha256'],deadline)
        others=[v for (j,b,s),v in self.publications.items() if j==job and b==value['boundary_sha256']]
        if others and any(v['deadline_epoch']!=deadline for v in others):raise EvidenceError('publication deadline changed within boundary')
        self.event('publication',{'job_sha256':job,'observation':value},progress=True);return True

    def terminal_retained(self,job,proof):
        self.active(job);contract=self.contract(job);b=self.bindings[job]
        proof=Path(proof)
        if proof.is_symlink():raise EvidenceError('terminal retention proof symlink')
        value=read_json(proof)
        receipts=verify_retention(value,b['control'],b['transports'],b['job_file'],job,b['worker_sha256'])
        for receipt in receipts:self.exported_files(job,Path(receipt['files_directory']),receipt['files'])
        selected={'proof_path':str(proof.resolve()),'proof_sha256':digest(value),'contract_sha256':digest(contract)}
        directory=self.journal.directory/'production-retention';directory.mkdir(mode=0o700,exist_ok=True)
        path=directory/(job+'.json')
        if path.exists():
            if read_json(path)!=selected:raise EvidenceError('terminal retention proof selection changed')
        else:write_json(path,selected)
        return selected

    def retained(self,job):
        selected=read_json(self.journal.directory/'production-retention'/(job+'.json'))
        fields(selected,'proof_path proof_sha256 contract_sha256','retained production proof selection')
        if selected['contract_sha256']!=digest(self.contract(job)):raise EvidenceError('production retention contract changed')
        proof=Path(selected['proof_path'])
        if proof.is_symlink():raise EvidenceError('retention proof replaced by symlink')
        value=read_json(proof)
        if digest(value)!=selected['proof_sha256']:raise EvidenceError('retention proof changed')
        b=self.bindings[job];verify_retention(value,b['control'],b['transports'],b['job_file'],job,b['worker_sha256'])
        return selected,value

    def terminal(self,job,status,state):
        if self.jobs[job]['selection']['kind'] not in ('production-record','full-replay'):
            return (super().job_exit if state=='EXITED' else super().abandon_job)(job,status)
        self.active(job)
        if terminal_status(status,job)!=state:raise EvidenceError('incorrect production terminal mode')
        selected,value=self.retained(job)
        if value['terminal']!=status:raise EvidenceError('terminal status differs from complete retained outputs')
        self.event('job-exit' if state=='EXITED' else 'job-abandon',{'job_sha256':job,'exit':status,
                   'retention_sha256':digest(selected)},progress=True)

    def job_exit(self,job,status):return self.terminal(job,status,'EXITED')
    def abandon_job(self,job,status):return self.terminal(job,status,'ABANDONED')

    def _check_final(self,directory,expected):
        super()._check_final(directory,expected)
        for job,record in self.jobs.items():
            if record['selection']['kind'] not in ('production-record','full-replay'):continue
            selected,value=self.retained(job)
            if record['selection']['retention_contract_sha256']!=selected['contract_sha256']:raise EvidenceError('started production contract differs')
            terminal=[e['body']['detail'] for e in self.journal.events if e['body'].get('kind') in ('job-exit','job-abandon') and e['body']['detail']['job_sha256']==job]
            if len(terminal)!=1 or terminal[0].get('retention_sha256')!=digest(selected):raise EvidenceError('production terminal omitted complete-output retention')
